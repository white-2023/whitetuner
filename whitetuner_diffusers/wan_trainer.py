"""
WAN I2V LoKr 训练器

基于 musubi-tuner 的 WAN 实现
- 支持 I2V (Image to Video) 模式
- 使用 LoKr (Low-Rank Kronecker) 高效微调
- 支持 FP8 量化和多种内存优化
- 支持多种时间步采样方法
"""

import os
import sys
import gc
import math
import json
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from PIL.ImageOps import exif_transpose
from tqdm import tqdm
import numpy as np
from accelerate import init_empty_weights
from safetensors.torch import load_file, save_file

from base_trainer import (
    BaseTrainer,
    BaseTrainerConfig,
    fix_windows_encoding,
    sample_timesteps,
    TIMESTEP_TYPES,
    TimestepType,
)
from lokr import LokrModule, apply_lokr_to_transformer


class WanModelConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


WAN_I2V_21_CONFIG = WanModelConfig(
    i2v=True,
    is_fun_control=False,
    flf2v=False,
    v2_2=False,
    t5_dtype=torch.bfloat16,
    clip_dtype=torch.float16,
    text_len=512,
    vae_stride=(4, 8, 8),
    patch_size=(1, 2, 2),
    dim=5120,
    ffn_dim=13824,
    freq_dim=256,
    in_dim=36,
    out_dim=16,
    num_heads=40,
    num_layers=40,
    window_size=(-1, -1),
    qk_norm=True,
    cross_attn_norm=True,
    eps=1e-6,
    sample_shift=5.0,
    sample_neg_prompt="",
    boundary=None,
)

WAN_I2V_22_CONFIG = WanModelConfig(
    i2v=True,
    is_fun_control=False,
    flf2v=False,
    v2_2=True,
    t5_dtype=torch.bfloat16,
    text_len=512,
    vae_stride=(4, 8, 8),
    patch_size=(1, 2, 2),
    dim=5120,
    ffn_dim=13824,
    freq_dim=256,
    in_dim=36,
    out_dim=16,
    num_heads=40,
    num_layers=40,
    window_size=(-1, -1),
    qk_norm=True,
    cross_attn_norm=True,
    eps=1e-6,
    sample_shift=5.0,
    sample_neg_prompt="",
    boundary=0.900,
)


def verify_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False


def load_video_frames(video_path: str, num_frames: int) -> List[np.ndarray]:
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames


def load_image_folder_as_video(folder_path: str, num_frames: int) -> List[np.ndarray]:
    supported_exts = ('.jpg', '.jpeg', '.png', '.webp')
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(supported_exts)])
    frames = []
    for f in files[:num_frames]:
        img = Image.open(os.path.join(folder_path, f)).convert('RGB')
        frames.append(np.array(img))
    return frames


class WanConfig(BaseTrainerConfig):
    
    def __init__(
        self,
        video_folder: str,
        output_dir: str,
        dit_path: str,
        vae_path: str,
        t5_path: str,
        clip_path: str = None,
        wan_version: str = "2.2",
        dit_high_noise_path: str = None,
        timestep_boundary: float = None,
        offload_inactive_dit: bool = True,
        min_timestep: int = None,
        max_timestep: int = None,
        fp8_scaled: bool = True,
        blocks_to_swap: int = 0,
        gradient_checkpointing_cpu_offload: bool = False,
        use_pinned_memory: bool = False,
        full_matrix: bool = True,
        lora_dim: int = 10000,
        lora_alpha: int = 1,
        lokr_factor: int = 4,
        decompose_both: bool = False,
        num_train_steps: int = 5000,
        checkpoint_every_n_steps: int = 500,
        checkpoints_total_limit: int = 3,
        batch_size: int = 1,
        learning_rate: float = 1e-4,
        lr_warmup_steps: int = 100,
        resolution: int = 480,
        num_frames: int = 17,
        timestep_type: TimestepType = "shift",
        sigmoid_scale: float = 1.0,
        shift_scale: float = 5.0,
        lognorm_alpha: float = 0.75,
        use_caption: bool = True,
        caption_ext: str = ".txt",
        default_caption: str = "",
        cache_dir: str = None,  # Default: {video_folder}/_wan_cache
        use_tensorboard: bool = True,
        tensorboard_dir: str = None,
        gradient_accumulation_steps: int = 1,
        mixed_precision: str = "bf16",
        seed: int = 42,
        max_grad_norm: float = 1.0,
        resume_from_checkpoint: str = None,
        **kwargs,
    ):
        super().__init__(
            model_id=dit_path,
            output_dir=output_dir,
            num_train_steps=num_train_steps,
            checkpoint_every_n_steps=checkpoint_every_n_steps,
            checkpoints_total_limit=checkpoints_total_limit,
            batch_size=batch_size,
            learning_rate=learning_rate,
            lr_warmup_steps=lr_warmup_steps,
            resolution=resolution,
            quantize_transformer=False,
            quantize_text_encoder=False,
            use_tensorboard=use_tensorboard,
            tensorboard_dir=tensorboard_dir,
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            seed=seed,
            max_grad_norm=max_grad_norm,
            resume_from_checkpoint=resume_from_checkpoint,
        )
        
        self.video_folder = video_folder
        self.dit_path = dit_path
        self.vae_path = vae_path
        self.t5_path = t5_path
        self.clip_path = clip_path
        self.wan_version = wan_version
        
        self.dit_high_noise_path = dit_high_noise_path
        self.offload_inactive_dit = offload_inactive_dit
        self.min_timestep = min_timestep
        self.max_timestep = max_timestep
        
        self.fp8_scaled = fp8_scaled
        self.blocks_to_swap = blocks_to_swap
        self.gradient_checkpointing_cpu_offload = gradient_checkpointing_cpu_offload
        self.use_pinned_memory = use_pinned_memory
        
        self.full_matrix = full_matrix
        self.lora_dim = lora_dim
        self.lora_alpha = lora_alpha
        self.lokr_factor = lokr_factor
        self.decompose_both = decompose_both
        
        has_low = dit_path is not None and dit_path.strip()
        has_high = dit_high_noise_path is not None and dit_high_noise_path.strip()
        
        if has_low and has_high:
            self.high_low_training = True
            self.single_model_type = None
        elif has_low:
            self.high_low_training = False
            self.single_model_type = "low"
        elif has_high:
            self.high_low_training = False
            self.single_model_type = "high"
            self.dit_path = dit_high_noise_path
            self.model_id = dit_high_noise_path
        else:
            raise ValueError("At least one DiT model path is required")
        
        if timestep_boundary is not None:
            if timestep_boundary > 1:
                timestep_boundary = timestep_boundary / 1000.0
            self.timestep_boundary = timestep_boundary
        elif wan_version == "2.2" and self.high_low_training:
            self.timestep_boundary = 0.900
        else:
            self.timestep_boundary = None
        
        self.num_frames = num_frames
        self.timestep_type = timestep_type
        self.sigmoid_scale = sigmoid_scale
        self.shift_scale = shift_scale
        self.lognorm_alpha = lognorm_alpha
        self.use_caption = use_caption
        self.caption_ext = caption_ext
        self.default_caption = default_caption
        
        # Cache directory defaults to {video_folder}/_wan_cache
        if cache_dir is None:
            self.cache_dir = os.path.join(video_folder, "_wan_cache")
        else:
            dataset_name = os.path.basename(os.path.normpath(video_folder))
            self.cache_dir = os.path.join(cache_dir, dataset_name)
        
        if wan_version == "2.2":
            self.wan_config = WAN_I2V_22_CONFIG
            self.use_clip = False
        else:
            self.wan_config = WAN_I2V_21_CONFIG
            self.use_clip = True
            if clip_path is None:
                raise ValueError("WAN 2.1 requires CLIP path")


class WanI2VDataset(Dataset):
    
    def __init__(
        self,
        video_folder: str,
        resolution: int = 480,
        num_frames: int = 17,
        use_caption: bool = True,
        caption_ext: str = ".txt",
        default_caption: str = "",
        dtype: torch.dtype = torch.bfloat16,
        verbose: bool = True,
    ):
        super().__init__()
        self.video_folder = video_folder
        self.resolution = resolution
        self.num_frames = num_frames
        self.use_caption = use_caption
        self.caption_ext = caption_ext
        self.default_caption = default_caption
        self.dtype = dtype
        
        video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
        image_exts = ('.jpg', '.jpeg', '.png', '.webp')
        
        self.samples = []
        
        for item in os.listdir(video_folder):
            item_path = os.path.join(video_folder, item)
            base_name = os.path.splitext(item)[0]
            
            if item.lower().endswith(video_exts):
                caption = default_caption
                if use_caption:
                    caption_path = os.path.join(video_folder, base_name + caption_ext)
                    if os.path.exists(caption_path):
                        with open(caption_path, 'r', encoding='utf-8') as f:
                            caption = f.read().strip()
                
                self.samples.append({
                    'type': 'video',
                    'path': item_path,
                    'caption': caption,
                })
            
            elif os.path.isdir(item_path):
                image_files = [f for f in os.listdir(item_path) if f.lower().endswith(image_exts)]
                if len(image_files) >= num_frames:
                    caption = default_caption
                    if use_caption:
                        caption_path = os.path.join(video_folder, base_name + caption_ext)
                        if not os.path.exists(caption_path):
                            caption_path = os.path.join(item_path, "caption.txt")
                        if os.path.exists(caption_path):
                            with open(caption_path, 'r', encoding='utf-8') as f:
                                caption = f.read().strip()
                    
                    self.samples.append({
                        'type': 'image_folder',
                        'path': item_path,
                        'caption': caption,
                    })
        
        if verbose:
            print(f"Found {len(self.samples)} video samples")
            video_count = sum(1 for s in self.samples if s['type'] == 'video')
            folder_count = sum(1 for s in self.samples if s['type'] == 'image_folder')
            print(f"  - Video files: {video_count}")
            print(f"  - Image folders: {folder_count}")
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid video samples found in: {video_folder}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        if sample['type'] == 'video':
            frames = load_video_frames(sample['path'], self.num_frames)
        else:
            frames = load_image_folder_as_video(sample['path'], self.num_frames)
        
        if len(frames) < self.num_frames:
            last_frame = frames[-1] if frames else np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
            while len(frames) < self.num_frames:
                frames.append(last_frame)
        
        h, w = frames[0].shape[:2]
        scale = self.resolution / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        new_h = (new_h // 16) * 16
        new_w = (new_w // 16) * 16
        
        processed_frames = []
        for frame in frames[:self.num_frames]:
            img = Image.fromarray(frame)
            img = img.resize((new_w, new_h), Image.BICUBIC)
            frame_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            processed_frames.append(frame_tensor)
        
        video_tensor = torch.stack(processed_frames, dim=1)
        first_frame = processed_frames[0]
        
        return {
            'video': video_tensor,
            'first_frame': first_frame,
            'caption': sample['caption'],
            'sample_idx': idx,
        }


class WanTrainer(BaseTrainer):
    
    def __init__(self, config: WanConfig):
        super().__init__(config)
        self.config: WanConfig = config
        
        self.t5_cache = {}
        self.clip_cache = {}
        self.latent_cache = {}
        self.image_latent_cache = {}
        
        self.t5_encoder = None
        self.clip_model = None
        self.vae = None
        self.dit = None
        
        self.dit_inactive_state_dict = None
        self.current_model_is_high_noise = False
        self.next_model_is_high_noise = False
    
    def _check_stop(self, stage: str = None) -> bool:
        return self.check_stop(stage)
    
    def create_dataset(self):
        if self.accelerator.is_main_process:
            print("\nCreating WAN I2V Dataset")
            print("=" * 60)
        
        self.dataset = WanI2VDataset(
            video_folder=self.config.video_folder,
            resolution=self.config.resolution,
            num_frames=self.config.num_frames,
            use_caption=self.config.use_caption,
            caption_ext=self.config.caption_ext,
            default_caption=self.config.default_caption,
            dtype=self.config.dtype,
            verbose=self.accelerator.is_main_process,
        )
        
        def collate_fn(batch):
            videos = torch.stack([item['video'] for item in batch])
            first_frames = torch.stack([item['first_frame'] for item in batch])
            captions = [item['caption'] for item in batch]
            sample_indices = [item['sample_idx'] for item in batch]
            
            return {
                'video': videos,
                'first_frame': first_frames,
                'caption': captions,
                'sample_indices': sample_indices,
            }
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            collate_fn=collate_fn,
        )
        
        if self.accelerator.is_main_process:
            print(f"DataLoader created, batch_size={self.config.batch_size}")
    
    def load_models(self):
        if self._check_stop():
            return
        
        self._load_encoders()
        if self._check_stop():
            return
        
        self._cache_embeddings()
        if self._check_stop():
            return
        
        self._load_dit()
        if self._check_stop():
            return
        
        self._prepare_for_ddp()
    
    def _load_encoders(self):
        if self.accelerator.is_main_process:
            version_str = "WAN 2.2" if self.config.wan_version == "2.2" else "WAN 2.1"
            print(f"\nPhase 1: Loading {version_str} encoders (T5, VAE" + (", CLIP" if self.config.use_clip else "") + ")")
            print("=" * 60)
        
        if self._check_stop():
            return
        
        from wan_modules.t5 import T5EncoderModel
        from wan_modules.vae import WanVAE
        
        if self.accelerator.is_main_process:
            print(">>> Loading T5 Encoder...")
        
        self.t5_encoder = T5EncoderModel(
            text_len=self.config.wan_config.text_len,
            dtype=self.config.wan_config.t5_dtype,
            device=self.accelerator.device,
            weight_path=self.config.t5_path,
        )
        
        if self.accelerator.is_main_process:
            print("T5 Encoder loaded")
        
        if self._check_stop():
            return
        
        if self.config.use_clip:
            from wan_modules.clip import CLIPModel
            
            if self.accelerator.is_main_process:
                print(">>> Loading CLIP (WAN 2.1 mode)...")
            
            self.clip_model = CLIPModel(
                dtype=self.config.wan_config.clip_dtype,
                device=self.accelerator.device,
                weight_path=self.config.clip_path,
            )
            self.clip_model.model.to(self.accelerator.device)
            
            if self.accelerator.is_main_process:
                print("CLIP loaded")
        else:
            self.clip_model = None
            if self.accelerator.is_main_process:
                print(">>> Skipping CLIP (WAN 2.2 mode - not needed)")
        
        if self._check_stop():
            return
        
        if self.accelerator.is_main_process:
            print(">>> Loading VAE...")
        
        self.vae = WanVAE(
            vae_path=self.config.vae_path,
            device="cpu",
            dtype=self.config.dtype,
        )
        self.vae.to(self.accelerator.device)
        
        if self.accelerator.is_main_process:
            print("VAE loaded")
        
        self.flush_memory()
    
    def _cache_embeddings(self):
        if self.accelerator.is_main_process:
            print("\nPhase 2: Caching Embeddings")
            print("=" * 60)
        
        if self._check_stop():
            return
        
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            os.makedirs(self.config.cache_dir, exist_ok=True)
        
        self.accelerator.wait_for_everyone()
        
        samples_to_encode = []
        for idx in range(len(self.dataset)):
            cache_file = os.path.join(self.config.cache_dir, f"sample_{idx}.pt")
            if os.path.exists(cache_file):
                cached = torch.load(cache_file, map_location='cpu')
                self.t5_cache[idx] = cached['t5_embeds']
                self.clip_cache[idx] = cached['clip_embeds']
                self.latent_cache[idx] = cached['latents']
                self.image_latent_cache[idx] = cached['image_latents']
            else:
                samples_to_encode.append(idx)
        
        num_processes = self.accelerator.num_processes
        process_index = self.accelerator.process_index
        
        if self.accelerator.is_main_process:
            if len(self.t5_cache) == 0:
                print(f"No cache found, creating cache ({len(samples_to_encode)} samples)")
            else:
                print(f"Loaded {len(self.t5_cache)} cached samples, need to encode {len(samples_to_encode)} samples")
        
        if len(samples_to_encode) > 0:
            my_samples = []
            for i, idx in enumerate(samples_to_encode):
                if i % num_processes == process_index:
                    my_samples.append(idx)
            
            pbar = None
            if self.accelerator.is_main_process:
                pbar = tqdm(total=len(samples_to_encode), desc="Caching embeddings")
            
            with torch.no_grad():
                for idx in my_samples:
                    if self._check_stop():
                        break
                    
                    sample = self.dataset[idx]
                    
                    caption = sample['caption']
                    t5_embeds = self.t5_encoder([caption], self.accelerator.device)
                    t5_embeds = t5_embeds[0].cpu()
                    
                    clip_embeds = None
                    if self.config.use_clip and self.clip_model is not None:
                        first_frame = sample['first_frame'].unsqueeze(0).unsqueeze(2)
                        first_frame = first_frame * 2 - 1
                        first_frame = first_frame.to(self.accelerator.device)
                        
                        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                            clip_embeds = self.clip_model.visual([first_frame.squeeze(2).transpose(1, 2)])
                        clip_embeds = clip_embeds.cpu()
                    
                    video = sample['video'].unsqueeze(0)
                    video = video * 2 - 1
                    video = video.to(self.accelerator.device, self.config.dtype)
                    
                    latents = self.vae.encode([video.squeeze(0)])[0].cpu()
                    
                    lat_f, lat_h, lat_w = latents.shape[1:4]
                    num_channels = latents.shape[0]
                    frame_count = self.config.num_frames
                    
                    first_frame_for_latent = sample['first_frame'].unsqueeze(0).unsqueeze(2)
                    first_frame_for_latent = first_frame_for_latent * 2 - 1
                    padding_frames = frame_count - 1
                    h, w = first_frame_for_latent.shape[3:5]
                    padded_frames = torch.cat([
                        first_frame_for_latent,
                        torch.zeros(1, 3, padding_frames, h, w)
                    ], dim=2).to(self.accelerator.device, self.config.dtype)
                    
                    first_frame_latent = self.vae.encode([padded_frames.squeeze(0)])[0]
                    first_frame_latent = first_frame_latent[:, :lat_f].cpu()
                    
                    msk = torch.ones(frame_count, lat_h, lat_w)
                    msk[1:] = 0
                    msk = torch.cat([msk[0:1].repeat(4, 1, 1), msk[1:]], dim=0)
                    msk = msk.view(msk.shape[0] // 4, 4, lat_h, lat_w)
                    msk = msk.transpose(0, 1)
                    
                    image_latents = torch.cat([msk, first_frame_latent], dim=0)
                    
                    cache_data = {
                        't5_embeds': t5_embeds,
                        'clip_embeds': clip_embeds,
                        'latents': latents,
                        'image_latents': image_latents,
                        'wan_version': self.config.wan_version,
                    }
                    
                    cache_file = os.path.join(self.config.cache_dir, f"sample_{idx}.pt")
                    torch.save(cache_data, cache_file)
                    
                    self.t5_cache[idx] = t5_embeds
                    self.clip_cache[idx] = clip_embeds
                    self.latent_cache[idx] = latents
                    self.image_latent_cache[idx] = image_latents
                    
                    if pbar is not None:
                        pbar.update(num_processes)
            
            if pbar is not None:
                pbar.close()
        
        self.accelerator.wait_for_everyone()
        
        if self._check_stop():
            return
        
        for idx in samples_to_encode:
            if idx not in self.t5_cache:
                cache_file = os.path.join(self.config.cache_dir, f"sample_{idx}.pt")
                if os.path.exists(cache_file):
                    cached = torch.load(cache_file, map_location='cpu')
                    self.t5_cache[idx] = cached['t5_embeds']
                    self.clip_cache[idx] = cached['clip_embeds']
                    self.latent_cache[idx] = cached['latents']
                    self.image_latent_cache[idx] = cached['image_latents']
        
        if self.accelerator.is_main_process:
            print(f"Caching complete, {len(self.t5_cache)} samples cached")
        
        mem_before = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        del self.t5_encoder, self.clip_model, self.vae
        self.t5_encoder = None
        self.clip_model = None
        self.vae = None
        torch.cuda.empty_cache()
        gc.collect()
        mem_after = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        freed_memory = mem_before - mem_after
        if self.accelerator.is_main_process:
            print(f"Freed {freed_memory:.2f}GB VRAM after unloading encoders")
    
    def _load_dit(self):
        if self.accelerator.is_main_process:
            print("\nPhase 3: Loading DiT (Transformer)")
            print("=" * 60)
            if self.config.high_low_training:
                print(f">>> WAN 2.2 Dual Model Training Mode")
                print(f">>> timestep_boundary: {self.config.timestep_boundary}")
            elif self.config.single_model_type == "high":
                print(f">>> Single Model Mode: HIGH-noise only")
            else:
                print(f">>> Single Model Mode: LOW-noise only")
            if self.config.fp8_scaled:
                print(f">>> FP8 Scaled: Enabled")
            if self.config.blocks_to_swap > 0:
                print(f">>> Block Swap: {self.config.blocks_to_swap} blocks")
            if self.config.gradient_checkpointing_cpu_offload:
                print(f">>> Activation CPU Offloading: Enabled")
        
        if self._check_stop():
            return
        
        from wan_modules.model import load_wan_model
        
        blocks_to_swap = self.config.blocks_to_swap
        loading_device = "cpu" if blocks_to_swap > 0 else self.accelerator.device
        dit_weight_dtype = None if self.config.fp8_scaled else self.config.dtype
        
        model_type = "High-noise" if self.config.single_model_type == "high" else "Low-noise"
        if self.accelerator.is_main_process:
            print(f">>> Loading {model_type} DiT from {self.config.dit_path}...")
        
        self.dit = load_wan_model(
            config=self.config.wan_config,
            device=self.accelerator.device,
            dit_path=self.config.dit_path,
            attn_mode="torch",
            split_attn=False,
            loading_device=loading_device,
            dit_weight_dtype=dit_weight_dtype,
            fp8_scaled=self.config.fp8_scaled,
        )
        
        if self._check_stop():
            return
        
        if self.config.high_low_training:
            if self.accelerator.is_main_process:
                print(f">>> Loading High-noise DiT from {self.config.dit_high_noise_path}...")
            
            dit_high_noise = load_wan_model(
                config=self.config.wan_config,
                device=self.accelerator.device,
                dit_path=self.config.dit_high_noise_path,
                attn_mode="torch",
                split_attn=False,
                loading_device="cpu",
                dit_weight_dtype=dit_weight_dtype,
                fp8_scaled=self.config.fp8_scaled,
            )
            
            self.dit_inactive_state_dict = dit_high_noise.state_dict()
            del dit_high_noise
            gc.collect()
            
            self.current_model_is_high_noise = False
            self.next_model_is_high_noise = False
            
            if self.accelerator.is_main_process:
                print(f"High-noise model loaded to CPU (inactive)")
        
        if blocks_to_swap > 0:
            if self.accelerator.is_main_process:
                print(f">>> Enabling block swap: {blocks_to_swap} blocks")
            self.dit.enable_block_swap(
                blocks_to_swap,
                self.accelerator.device,
                supports_backward=True,
                use_pinned_memory=self.config.use_pinned_memory
            )
        
        self.dit.requires_grad_(False)
        
        if self.accelerator.is_main_process:
            print(f"\n>>> Applying LoKr to DiT...")
        
        self.lokr_modules, self.lokr_module_names = apply_lokr_to_transformer(
            self.dit,
            lora_dim=self.config.lora_dim,
            alpha=self.config.lora_alpha,
            factor=self.config.lokr_factor,
            full_matrix=self.config.full_matrix,
            decompose_both=self.config.decompose_both,
            verbose=self.accelerator.is_main_process,
        )
        
        # Move LoKr parameters to GPU and set requires_grad
        device = self.accelerator.device
        lokr_dtype = self.config.dtype  # Use training dtype (bf16)
        
        if self.accelerator.is_main_process:
            print(f">>> Moving LoKr parameters to {device} with dtype {lokr_dtype}...")
        
        for lokr in self.lokr_modules:
            # Move all parameters to GPU with correct dtype
            for name, param in lokr.named_parameters():
                param.data = param.data.to(device=device, dtype=lokr_dtype)
                param.requires_grad = True
        
        if self.accelerator.is_main_process:
            print(f">>> LoKr parameters moved to GPU")
        
        self.dit.train()
        
        if hasattr(self.dit, 'enable_gradient_checkpointing'):
            self.dit.enable_gradient_checkpointing(self.config.gradient_checkpointing_cpu_offload)
            if self.accelerator.is_main_process:
                gc_info = "with Activation CPU Offloading" if self.config.gradient_checkpointing_cpu_offload else ""
                print(f"Gradient Checkpointing enabled {gc_info}")
        
        self.transformer = self.dit
        
        if self.accelerator.is_main_process:
            total_params = sum(p.numel() for p in self.dit.parameters())
            lokr_params = sum(p.numel() for module in self.lokr_modules for p in module.parameters())
            print(f"DiT loaded with LoKr")
            print(f"  - Base model params: {total_params:,}")
            print(f"  - LoKr trainable params: {lokr_params:,}")
            print(f"  - LoKr modules: {len(self.lokr_modules)}")
            if self.config.high_low_training:
                print(f"  - Dual model mode: HIGH >= {self.config.timestep_boundary:.3f}, LOW < {self.config.timestep_boundary:.3f}")
            else:
                print(f"  - Single model mode: {model_type}")
    
    def _prepare_for_ddp(self):
        if self.accelerator.is_main_process:
            print("\nPhase 4: Preparing for distributed training")
            print("=" * 60)
        
        blocks_to_swap = self.config.blocks_to_swap
        
        if blocks_to_swap > 0:
            self.dit.move_to_device_except_swap_blocks(self.accelerator.device)
            self.dit.prepare_block_swap_before_forward()
            if self.accelerator.is_main_process:
                if self.accelerator.num_processes > 1:
                    print("Block Swap + Multi-GPU: Skip DDP wrapper, use manual gradient sync")
                print(f"DiT prepared (Block Swap mode, manual gradient sync)")
        else:
            self.dit = self.accelerator.prepare(self.dit)
            if self.accelerator.is_main_process:
                print(f"DiT prepared for DDP (num_processes: {self.accelerator.num_processes})")
        
        self.transformer = self.dit
    
    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        return [p for module in self.lokr_modules for p in module.parameters() if p.requires_grad]
    
    def swap_high_low_weights(self):
        if not self.config.high_low_training:
            return
        
        if self.current_model_is_high_noise == self.next_model_is_high_noise:
            return
        
        if self.config.offload_inactive_dit:
            self.dit.to("cpu", non_blocking=True)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        current_state_dict = self.dit.state_dict()
        
        self.dit.load_state_dict(self.dit_inactive_state_dict, strict=True, assign=True)
        
        if self.config.offload_inactive_dit:
            self.dit.to(self.accelerator.device, non_blocking=True)
            torch.cuda.synchronize()
        
        self.dit_inactive_state_dict = current_state_dict
        self.current_model_is_high_noise = self.next_model_is_high_noise
        
        if self.accelerator.is_main_process:
            model_type = "HIGH-noise" if self.current_model_is_high_noise else "LOW-noise"
            print(f"  [Switched to {model_type} model]")
    
    def sample_timesteps_for_high_low(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        first_timestep, _ = sample_timesteps(
            1,
            num_train_timesteps=1000,
            device=device,
            timestep_type=self.config.timestep_type,
            sigmoid_scale=self.config.sigmoid_scale,
            shift=self.config.shift_scale,
            lognorm_alpha=self.config.lognorm_alpha,
        )
        
        is_high_noise = (first_timestep[0].item() / 1000.0) >= self.config.timestep_boundary
        self.next_model_is_high_noise = is_high_noise
        
        timesteps_list = []
        weights_list = []
        max_attempts = 100
        
        for _ in range(batch_size):
            for attempt in range(max_attempts):
                ts, w = sample_timesteps(
                    1,
                    num_train_timesteps=1000,
                    device=device,
                    timestep_type=self.config.timestep_type,
                    sigmoid_scale=self.config.sigmoid_scale,
                    shift=self.config.shift_scale,
                    lognorm_alpha=self.config.lognorm_alpha,
                )
                ts_normalized = ts[0].item() / 1000.0
                
                if is_high_noise and ts_normalized >= self.config.timestep_boundary:
                    timesteps_list.append(ts)
                    weights_list.append(w)
                    break
                elif not is_high_noise and ts_normalized < self.config.timestep_boundary:
                    timesteps_list.append(ts)
                    weights_list.append(w)
                    break
            else:
                if is_high_noise:
                    ts = torch.tensor([int(self.config.timestep_boundary * 1000 + 50)], device=device)
                else:
                    ts = torch.tensor([int(self.config.timestep_boundary * 1000 - 50)], device=device)
                timesteps_list.append(ts)
                weights_list.append(torch.ones(1, device=device))
        
        timesteps = torch.cat(timesteps_list, dim=0)
        weights = torch.cat(weights_list, dim=0)
        
        return timesteps, weights, is_high_noise
    
    def train_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        device = self.accelerator.device
        sample_indices = batch['sample_indices']
        batch_size = len(sample_indices)
        
        latents_list = []
        t5_embeds_list = []
        clip_embeds_list = []
        image_latents_list = []
        
        for idx in sample_indices:
            latents_list.append(self.latent_cache[idx])
            t5_embeds_list.append(self.t5_cache[idx])
            if self.config.use_clip:
                clip_embeds_list.append(self.clip_cache[idx])
            image_latents_list.append(self.image_latent_cache[idx])
        
        latents = torch.stack(latents_list, dim=0).to(device, self.config.dtype)
        
        t5_embeds = [e.to(device, self.config.dtype) for e in t5_embeds_list]
        
        clip_embeds = None
        if self.config.use_clip and len(clip_embeds_list) > 0:
            clip_embeds = torch.stack(clip_embeds_list, dim=0).to(device, self.config.dtype)
        
        image_latents = torch.stack(image_latents_list, dim=0).to(device, self.config.dtype)
        
        if self.config.blocks_to_swap > 0:
            self.dit.prepare_block_swap_before_forward()
        
        if self.config.high_low_training:
            timesteps, timestep_weights, is_high_noise = self.sample_timesteps_for_high_low(batch_size, device)
            self.swap_high_low_weights()
        else:
            timesteps, timestep_weights = sample_timesteps(
                batch_size,
                num_train_timesteps=1000,
                device=device,
                timestep_type=self.config.timestep_type,
                sigmoid_scale=self.config.sigmoid_scale,
                shift=self.config.shift_scale,
                lognorm_alpha=self.config.lognorm_alpha,
            )
        
        noise = torch.randn_like(latents)
        
        sigmas = timesteps.float() / 1000.0
        sigmas = sigmas.view(-1, 1, 1, 1, 1)
        noisy_latents = (1 - sigmas) * latents + sigmas * noise
        
        lat_f, lat_h, lat_w = latents.shape[2:5]
        
        noisy_input = [noisy_latents[i] for i in range(batch_size)]
        
        seq_len = lat_f * lat_h * lat_w // (self.config.wan_config.patch_size[0] * self.config.wan_config.patch_size[1] * self.config.wan_config.patch_size[2])
        
        with self.accelerator.autocast():
            model_pred = self.dit(
                noisy_input,
                t=timesteps,
                context=t5_embeds,
                clip_fea=clip_embeds,
                seq_len=seq_len,
                y=image_latents,
            )
        
        model_pred = torch.stack(model_pred, dim=0)
        
        target = noise - latents
        
        per_sample_loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        per_sample_loss = per_sample_loss.mean(dim=[1, 2, 3, 4])
        
        weighted_loss = per_sample_loss * timestep_weights
        loss = weighted_loss.mean()
        
        return loss
    
    def save_checkpoint(self, step: int):
        output_dir = self.config.output_dir or os.path.join(self.script_dir, "wan_output")
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        lokr_state_dict = {}
        for idx, (module, layer_name) in enumerate(zip(self.lokr_modules, self.lokr_module_names)):
            # Use diffusion_model.{layer_name} format for ComfyUI compatibility
            key_prefix = f"diffusion_model.{layer_name}"
            
            if hasattr(module, 'alpha'):
                lokr_state_dict[f"{key_prefix}.alpha"] = module.alpha.cpu()
            
            for param_name, param in module.named_parameters():
                lokr_state_dict[f"{key_prefix}.{param_name}"] = param.cpu()
        
        save_file(lokr_state_dict, os.path.join(checkpoint_dir, "lokr_weights.safetensors"))
        
        lokr_config = {
            "lora_dim": self.config.lora_dim,
            "lora_alpha": self.config.lora_alpha,
            "lokr_factor": self.config.lokr_factor,
            "full_matrix": self.config.full_matrix,
            "decompose_both": self.config.decompose_both,
            "num_modules": len(self.lokr_modules),
        }
        with open(os.path.join(checkpoint_dir, "lokr_config.json"), "w") as f:
            json.dump(lokr_config, f, indent=2)
        
        self.save_accelerate_state(checkpoint_dir, step)
        self._save_config(checkpoint_dir)
    
    def _save_config(self, checkpoint_dir: str):
        if not self.accelerator.is_main_process:
            return
        
        config_dict = {
            "wan_version": self.config.wan_version,
            "dit_path": self.config.dit_path,
            "dit_high_noise_path": self.config.dit_high_noise_path,
            "high_low_training": self.config.high_low_training,
            "timestep_boundary": self.config.timestep_boundary,
            "vae_path": self.config.vae_path,
            "t5_path": self.config.t5_path,
            "clip_path": self.config.clip_path,
            "video_folder": self.config.video_folder,
            "output_dir": self.config.output_dir,
            "num_train_steps": self.config.num_train_steps,
            "learning_rate": self.config.learning_rate,
            "resolution": self.config.resolution,
            "num_frames": self.config.num_frames,
            "timestep_type": self.config.timestep_type,
            "shift_scale": self.config.shift_scale,
            "training_method": "lokr",
            "lora_dim": self.config.lora_dim,
            "lora_alpha": self.config.lora_alpha,
            "lokr_factor": self.config.lokr_factor,
            "full_matrix": self.config.full_matrix,
            "decompose_both": self.config.decompose_both,
        }
        config_path = os.path.join(checkpoint_dir, "training_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def load_checkpoint(self, checkpoint_dir: str):
        if not os.path.exists(checkpoint_dir):
            raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        if self.accelerator.is_main_process:
            print(f"Resuming from checkpoint: {checkpoint_dir}")
        
        state_path = os.path.join(checkpoint_dir, "training_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                state_info = json.load(f)
            self.resume_step = state_info.get("step", 0)
        else:
            folder_name = os.path.basename(checkpoint_dir)
            if folder_name.startswith("checkpoint-"):
                try:
                    self.resume_step = int(folder_name.split("-")[1])
                except:
                    self.resume_step = 0
        
        if self.accelerator.is_main_process:
            print(f"  - Resume step: {self.resume_step}")
        
        lokr_weights_path = os.path.join(checkpoint_dir, "lokr_weights.safetensors")
        if os.path.exists(lokr_weights_path) and self.lokr_modules is not None:
            if self.accelerator.is_main_process:
                print(f"  - Loading LoKr weights...")
            
            lokr_state_dict = load_file(lokr_weights_path)
            
            for module, layer_name in zip(self.lokr_modules, self.lokr_module_names):
                key_prefix = f"diffusion_model.{layer_name}"
                
                for param_name, param in module.named_parameters():
                    key = f"{key_prefix}.{param_name}"
                    if key in lokr_state_dict:
                        param.data.copy_(lokr_state_dict[key].to(param.device, param.dtype))
            
            if self.accelerator.is_main_process:
                print(f"  - LoKr weights restored")
        
        accelerate_state_dir = os.path.join(checkpoint_dir, "accelerate_state")
        if os.path.exists(accelerate_state_dir):
            self.accelerator.load_state(accelerate_state_dir)
            if self.accelerator.is_main_process:
                print(f"  - Accelerate state restored")
    
    def save_final_model(self):
        self.accelerator.wait_for_everyone()
        
        if not self.accelerator.is_main_process:
            return
        
        print("\nSaving final model")
        print("=" * 60)
        
        output_dir = self.config.output_dir or os.path.join(self.script_dir, "wan_output")
        os.makedirs(output_dir, exist_ok=True)
        
        lokr_state_dict = {}
        for idx, (module, layer_name) in enumerate(zip(self.lokr_modules, self.lokr_module_names)):
            # Use diffusion_model.{layer_name} format for ComfyUI compatibility
            key_prefix = f"diffusion_model.{layer_name}"
            
            if hasattr(module, 'alpha'):
                lokr_state_dict[f"{key_prefix}.alpha"] = module.alpha.cpu()
            
            for param_name, param in module.named_parameters():
                lokr_state_dict[f"{key_prefix}.{param_name}"] = param.cpu()
        
        metadata = {
            "lora_dim": str(self.config.lora_dim),
            "lora_alpha": str(self.config.lora_alpha),
            "lokr_factor": str(self.config.lokr_factor),
            "full_matrix": str(self.config.full_matrix),
            "model_type": "lokr",
            "base_model": f"WAN{self.config.wan_version}_I2V",
            "ss_network_module": "lycoris.kohya",
            "ss_network_dim": str(self.config.lora_dim),
            "ss_network_alpha": str(self.config.lora_alpha),
            "num_gpus": str(self.accelerator.num_processes),
        }
        
        actual_steps = self.current_step if self.current_step > 0 else self.adjusted_num_train_steps
        equivalent_single_gpu_steps = actual_steps * self.accelerator.num_processes * self.config.gradient_accumulation_steps
        
        model_filename = f"wan{self.config.wan_version}_i2v_lokr_{equivalent_single_gpu_steps}steps.safetensors"
        config_filename = f"lokr_config_{equivalent_single_gpu_steps}steps.json"
        
        save_file(lokr_state_dict, os.path.join(output_dir, model_filename), metadata=metadata)
        
        lokr_config = {
            "lora_dim": self.config.lora_dim,
            "lora_alpha": self.config.lora_alpha,
            "lokr_factor": self.config.lokr_factor,
            "full_matrix": self.config.full_matrix,
            "decompose_both": self.config.decompose_both,
            "num_modules": len(self.lokr_modules),
            "base_model": f"WAN{self.config.wan_version}_I2V",
            "resolution": self.config.resolution,
            "num_frames": self.config.num_frames,
            "format": "comfyui_lokr",
            "trained_steps": actual_steps,
            "equivalent_single_gpu_steps": equivalent_single_gpu_steps,
            "original_num_train_steps": self.config.num_train_steps,
            "stopped_early": self.should_stop,
            "num_gpus": self.accelerator.num_processes,
            "batch_size_per_gpu": self.config.batch_size,
            "effective_batch_size": self.effective_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
        }
        with open(os.path.join(output_dir, config_filename), "w") as f:
            json.dump(lokr_config, f, indent=2)
        
        self.save_checkpoint(actual_steps)
        
        print(f"\nFinal LoKr model saved to: {output_dir}/")
        print(f"  - {model_filename}")
        print(f"  - {config_filename}")
        print(f"Final checkpoint saved to: {checkpoint_dir}/checkpoint-{actual_steps}/")
        
        if self.should_stop:
            print(f"\nTraining stopped at step {actual_steps}")
        else:
            print(f"\nTraining complete!")
        
        print(f"  Dataset: {len(self.dataset)} videos")
        print(f"  GPUs used: {self.accelerator.num_processes}")
        print(f"\nModel format: LyCORIS LoKr")
        print(f"Can be loaded directly in ComfyUI")
    
    def pre_training_hook(self):
        if self.accelerator.is_main_process:
            print(f"\nWAN I2V LoKr Training")
            print(f"Timestep sampling: {self.config.timestep_type}")
            if self.config.timestep_type == "sigmoid":
                print(f"  - sigmoid_scale: {self.config.sigmoid_scale}")
            elif self.config.timestep_type == "shift":
                print(f"  - shift_scale: {self.config.shift_scale}")
            elif self.config.timestep_type == "lognorm_blend":
                print(f"  - lognorm_alpha: {self.config.lognorm_alpha}")
            print(f"LoKr config: dim={self.config.lora_dim}, alpha={self.config.lora_alpha}, factor={self.config.lokr_factor}")
            print(f"  - full_matrix: {self.config.full_matrix}")
            if self.config.fp8_scaled:
                print(f"  - FP8 Scaled: Enabled")
            if self.config.gradient_checkpointing_cpu_offload:
                print(f"  - Activation CPU Offloading: Enabled")
            if self.config.blocks_to_swap > 0:
                print(f"  - Block Swap: {self.config.blocks_to_swap} blocks")
            print(f"\n⏳ 正在初始化训练，第一个 batch 可能需要 1-3 分钟...")
            print(f"   (首次前向传播需要初始化 CUDA kernels 和计算图)\n")


def main():
    import argparse
    
    fix_windows_encoding()
    
    parser = argparse.ArgumentParser(description="WAN I2V Full Trainer")
    
    parser.add_argument("--dit_path", type=str, default=None, help="DiT model path (low-noise for WAN 2.2)")
    parser.add_argument("--dit_high_noise_path", type=str, default=None, help="High-noise DiT model path (WAN 2.2 dual model)")
    parser.add_argument("--vae_path", type=str, required=True, help="VAE model path")
    parser.add_argument("--t5_path", type=str, required=True, help="T5 model path")
    parser.add_argument("--clip_path", type=str, default=None, help="CLIP model path (only for WAN 2.1)")
    parser.add_argument("--video_folder", type=str, required=True, help="Video folder path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--wan_version", type=str, default="2.2", choices=["2.1", "2.2"], help="WAN version (2.1 or 2.2)")
    parser.add_argument("--timestep_boundary", type=float, default=None, help="Timestep boundary for dual model (0.0-1.0, default 0.9 for I2V)")
    parser.add_argument("--offload_inactive_dit", action="store_true", default=True, help="Offload inactive DiT to CPU")
    
    parser.add_argument("--fp8_scaled", action="store_true", help="Use FP8 scaled quantization for DiT")
    parser.add_argument("--blocks_to_swap", type=int, default=0, help="Number of blocks to swap to CPU (0 to disable)")
    parser.add_argument("--gradient_checkpointing_cpu_offload", action="store_true", help="Offload activations to CPU during gradient checkpointing")
    parser.add_argument("--use_pinned_memory", action="store_true", help="Use pinned memory for block swap")
    
    parser.add_argument("--num_train_steps", type=int, default=5000, help="Training steps")
    parser.add_argument("--checkpoint_every_n_steps", type=int, default=500, help="Checkpoint interval")
    parser.add_argument("--checkpoints_total_limit", type=int, default=3, help="Max checkpoints")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--lr_warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--resolution", type=int, default=480, help="Video resolution")
    parser.add_argument("--num_frames", type=int, default=17, help="Number of frames")
    
    parser.add_argument("--timestep_type", type=str, default="shift", choices=TIMESTEP_TYPES, help="Timestep sampling type")
    parser.add_argument("--sigmoid_scale", type=float, default=1.0, help="Sigmoid scale")
    parser.add_argument("--shift_scale", type=float, default=5.0, help="Shift scale")
    parser.add_argument("--lognorm_alpha", type=float, default=0.75, help="Lognorm alpha")
    
    parser.add_argument("--use_caption", action="store_true", default=True, help="Use captions")
    parser.add_argument("--caption_ext", type=str, default=".txt", help="Caption file extension")
    parser.add_argument("--default_caption", type=str, default="", help="Default caption")
    
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume checkpoint")
    
    args = parser.parse_args()
    
    config = WanConfig(
        dit_path=args.dit_path,
        dit_high_noise_path=args.dit_high_noise_path,
        vae_path=args.vae_path,
        t5_path=args.t5_path,
        clip_path=args.clip_path,
        video_folder=args.video_folder,
        output_dir=args.output_dir,
        wan_version=args.wan_version,
        timestep_boundary=args.timestep_boundary,
        offload_inactive_dit=args.offload_inactive_dit,
        fp8_scaled=args.fp8_scaled,
        blocks_to_swap=args.blocks_to_swap,
        gradient_checkpointing_cpu_offload=args.gradient_checkpointing_cpu_offload,
        use_pinned_memory=args.use_pinned_memory,
        num_train_steps=args.num_train_steps,
        checkpoint_every_n_steps=args.checkpoint_every_n_steps,
        checkpoints_total_limit=args.checkpoints_total_limit,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_warmup_steps=args.lr_warmup_steps,
        resolution=args.resolution,
        num_frames=args.num_frames,
        timestep_type=args.timestep_type,
        sigmoid_scale=args.sigmoid_scale,
        shift_scale=args.shift_scale,
        lognorm_alpha=args.lognorm_alpha,
        use_caption=args.use_caption,
        caption_ext=args.caption_ext,
        default_caption=args.default_caption,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        seed=args.seed,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    
    trainer = WanTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()

