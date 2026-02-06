"""
Anima Full Training Trainer

基于 CircleStone Labs Anima 模型的全量训练器
- Anima 是基于 NVIDIA Cosmos Predict2 架构的 2B 参数文生图模型
- 使用 Qwen3 0.6B 作为文本编码器
- 使用 Wan 2.1 VAE (与 Qwen-Image VAE 相同)
- Flow Matching 采样 (shift=3.0)
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
from safetensors.torch import load_file, save_file

from base_trainer import (
    BaseTrainer,
    BaseTrainerConfig,
    fix_windows_encoding,
    sample_timesteps,
    TIMESTEP_TYPES,
    TimestepType,
)


ANIMA_CONFIG = {
    "max_img_h": 240,
    "max_img_w": 240,
    "max_frames": 128,
    "in_channels": 16,
    "out_channels": 16,
    "patch_spatial": 2,
    "patch_temporal": 1,
    "model_channels": 2048,
    "num_blocks": 28,
    "num_heads": 32,
    "mlp_ratio": 4.0,
    "crossattn_emb_channels": 1024,
    "pos_emb_cls": "rope3d",
    "concat_padding_mask": True,
    "sample_shift": 3.0,
    "vae_stride": (1, 8, 8),
    "latent_channels": 16,
}


def verify_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False


class AnimaConfig(BaseTrainerConfig):
    
    def __init__(
        self,
        image_folder: str,
        output_dir: str,
        dit_path: str,
        vae_path: str,
        text_encoder_path: str,
        qwen_tokenizer_path: str = None,
        num_train_steps: int = 5000,
        checkpoint_every_n_steps: int = 500,
        checkpoints_total_limit: int = 3,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        lr_warmup_steps: int = 100,
        resolution: int = 1024,
        timestep_type: TimestepType = "shift",
        sigmoid_scale: float = 1.0,
        shift_scale: float = 3.0,
        lognorm_alpha: float = 0.75,
        use_caption: bool = True,
        caption_ext: str = ".txt",
        default_caption: str = "",
        noise_offset: float = 0.0,
        train_text_encoder: bool = False,
        te_learning_rate: float = 1e-6,
        blocks_to_swap: int = 0,
        use_adafactor: bool = False,
        use_pinned_memory: bool = False,
        cache_dir: str = None,
        use_tensorboard: bool = True,
        tensorboard_dir: str = None,
        gradient_accumulation_steps: int = 1,
        mixed_precision: str = "bf16",
        seed: int = 42,
        max_grad_norm: float = 1.0,
        resume_from_checkpoint: str = None,
        gradient_checkpointing: bool = True,
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
        
        self.image_folder = image_folder
        self.dit_path = dit_path
        self.vae_path = vae_path
        self.text_encoder_path = text_encoder_path
        self.qwen_tokenizer_path = qwen_tokenizer_path
        
        self.timestep_type = timestep_type
        self.sigmoid_scale = sigmoid_scale
        self.shift_scale = shift_scale
        self.lognorm_alpha = lognorm_alpha
        self.use_caption = use_caption
        self.caption_ext = caption_ext
        self.default_caption = default_caption
        self.noise_offset = noise_offset
        
        self.gradient_checkpointing = gradient_checkpointing
        self.train_text_encoder = train_text_encoder
        self.te_learning_rate = te_learning_rate
        self.blocks_to_swap = blocks_to_swap
        self.use_adafactor = use_adafactor
        self.use_pinned_memory = use_pinned_memory
        
        if cache_dir is None:
            self.cache_dir = os.path.join(image_folder, "_anima_cache")
        else:
            dataset_name = os.path.basename(os.path.normpath(image_folder))
            self.cache_dir = os.path.join(cache_dir, dataset_name)


class AnimaImageDataset(Dataset):
    
    def __init__(
        self,
        image_folder: str,
        resolution: int = 1024,
        use_caption: bool = True,
        caption_ext: str = ".txt",
        default_caption: str = "",
        dtype: torch.dtype = torch.bfloat16,
        verbose: bool = True,
    ):
        super().__init__()
        self.image_folder = image_folder
        self.resolution = resolution
        self.use_caption = use_caption
        self.caption_ext = caption_ext
        self.default_caption = default_caption
        self.dtype = dtype
        
        image_exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
        
        self.samples = []
        
        for item in os.listdir(image_folder):
            if item.startswith('.') or item.startswith('_'):
                continue
            
            item_path = os.path.join(image_folder, item)
            base_name = os.path.splitext(item)[0]
            
            if item.lower().endswith(image_exts):
                if not verify_image(item_path):
                    continue
                
                caption = default_caption
                if use_caption:
                    caption_path = os.path.join(image_folder, base_name + caption_ext)
                    if os.path.exists(caption_path):
                        with open(caption_path, 'r', encoding='utf-8') as f:
                            caption = f.read().strip()
                
                self.samples.append({
                    'path': item_path,
                    'caption': caption,
                })
        
        if verbose:
            print(f"Found {len(self.samples)} image samples")
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid image samples found in: {image_folder}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        img = Image.open(sample['path']).convert('RGB')
        img = exif_transpose(img)
        
        h, w = img.size[1], img.size[0]
        scale = self.resolution / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        new_h = (new_h // 16) * 16
        new_w = (new_w // 16) * 16
        
        if new_h < 16:
            new_h = 16
        if new_w < 16:
            new_w = 16
        
        img = img.resize((new_w, new_h), Image.BICUBIC)
        
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        
        return {
            'image': img_tensor,
            'caption': sample['caption'],
            'sample_idx': idx,
            'image_name': os.path.basename(sample['path']),
        }


class AnimaTrainer(BaseTrainer):
    
    def __init__(self, config: AnimaConfig):
        super().__init__(config)
        self.config: AnimaConfig = config
        
        self.text_cache = {}
        self.latent_cache = {}
        self.caption_cache = {}
        
        self.text_encoder = None
        self.tokenizer = None
        self.vae = None
        self.dit = None
    
    def _check_stop(self, stage: str = None) -> bool:
        return self.check_stop(stage)
    
    def create_dataset(self):
        if self.accelerator.is_main_process:
            print("\nCreating Anima Image Dataset")
            print("=" * 60)
        
        self.dataset = AnimaImageDataset(
            image_folder=self.config.image_folder,
            resolution=self.config.resolution,
            use_caption=self.config.use_caption,
            caption_ext=self.config.caption_ext,
            default_caption=self.config.default_caption,
            dtype=self.config.dtype,
            verbose=self.accelerator.is_main_process,
        )
        
        def collate_fn(batch):
            max_h = max(item['image'].shape[1] for item in batch)
            max_w = max(item['image'].shape[2] for item in batch)
            
            max_h = ((max_h + 15) // 16) * 16
            max_w = ((max_w + 15) // 16) * 16
            
            padded_images = []
            for item in batch:
                img = item['image']
                c, h, w = img.shape
                padded = torch.zeros(c, max_h, max_w)
                padded[:, :h, :w] = img
                padded_images.append(padded)
            
            images = torch.stack(padded_images)
            captions = [item['caption'] for item in batch]
            sample_indices = [item['sample_idx'] for item in batch]
            image_names = [item['image_name'] for item in batch]
            
            return {
                'image': images,
                'caption': captions,
                'sample_indices': sample_indices,
                'image_names': image_names,
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
        
        self._prepare_for_training()
    
    def _load_encoders(self):
        if self.accelerator.is_main_process:
            print("\nPhase 1: Loading Encoders (Qwen3, VAE)")
            print("=" * 60)
        
        if self._check_stop():
            return
        
        if self.accelerator.is_main_process:
            print(">>> Loading Qwen3 0.6B Text Encoder...")
        
        from anima_modules.text_encoder import load_qwen3_text_encoder, AnimaTokenizer
        
        self.text_encoder = load_qwen3_text_encoder(
            model_path=self.config.text_encoder_path,
            device=self.accelerator.device,
            dtype=self.config.dtype,
        )
        
        self.tokenizer = AnimaTokenizer(
            qwen_tokenizer_path=self.config.qwen_tokenizer_path,
        )
        
        if self.accelerator.is_main_process:
            print("Qwen3 Text Encoder loaded")
        
        if self._check_stop():
            return
        
        if self.accelerator.is_main_process:
            print(">>> Loading VAE (Wan 2.1 / Qwen-Image VAE)...")
        
        from anima_modules.vae import load_anima_vae
        
        self.vae = load_anima_vae(
            vae_path=self.config.vae_path,
            device=self.accelerator.device,
            dtype=self.config.dtype,
        )
        
        if self.accelerator.is_main_process:
            print("VAE loaded")
        
        self.flush_memory()
    
    def _encode_text(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            device=self.accelerator.device,
        )
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        t5_ids = inputs.t5_ids
        
        with torch.no_grad():
            hidden_states = self.text_encoder(input_ids, attention_mask)
        
        return hidden_states, t5_ids
    
    def _encode_text_train(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            device=self.accelerator.device,
        )
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        t5_ids = inputs.t5_ids
        
        hidden_states = self.text_encoder(input_ids, attention_mask)
        
        return hidden_states, t5_ids
    
    def _cache_embeddings(self):
        if self.accelerator.is_main_process:
            print("\nPhase 2: Caching Embeddings")
            print("=" * 60)
            if self.config.train_text_encoder:
                print("  - Training Text Encoder: ENABLED (caching latents only)")
            else:
                print("  - Training Text Encoder: DISABLED (caching text embeds + latents)")
        
        if self._check_stop():
            return
        
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            os.makedirs(self.config.cache_dir, exist_ok=True)
        
        self.accelerator.wait_for_everyone()
        
        samples_to_encode = []
        cache_suffix = "_te" if self.config.train_text_encoder else ""
        
        for idx in range(len(self.dataset)):
            cache_file = os.path.join(self.config.cache_dir, f"sample_{idx}{cache_suffix}.pt")
            if os.path.exists(cache_file):
                cached = torch.load(cache_file, map_location='cpu', weights_only=True)
                if self.config.train_text_encoder:
                    self.caption_cache[idx] = cached['caption']
                else:
                    text_embeds = cached['text_embeds']
                    t5_ids = cached.get('t5_ids', None)
                    self.text_cache[idx] = (text_embeds, t5_ids)
                self.latent_cache[idx] = cached['latents']
            else:
                samples_to_encode.append(idx)
        
        num_processes = self.accelerator.num_processes
        process_index = self.accelerator.process_index
        
        if self.accelerator.is_main_process:
            cached_count = len(self.latent_cache)
            if cached_count == 0:
                print(f"No cache found, creating cache ({len(samples_to_encode)} samples)")
            else:
                print(f"Loaded {cached_count} cached samples, need to encode {len(samples_to_encode)} samples")
        
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
                    
                    img = sample['image'].unsqueeze(0)
                    img = img * 2 - 1
                    img = img.unsqueeze(2)
                    img = img.to(self.accelerator.device, self.config.dtype)
                    
                    latents = self.vae.encode([img.squeeze(0)])[0].cpu()
                    
                    if self.config.train_text_encoder:
                        cache_data = {
                            'caption': caption,
                            'latents': latents,
                        }
                        self.caption_cache[idx] = caption
                    else:
                        text_embeds, t5_ids = self._encode_text([caption])
                        text_embeds = text_embeds[0].cpu()
                        t5_ids = t5_ids[0].cpu()
                        cache_data = {
                            'text_embeds': text_embeds,
                            't5_ids': t5_ids,
                            'latents': latents,
                        }
                        self.text_cache[idx] = (text_embeds, t5_ids)
                    
                    self.latent_cache[idx] = latents
                    
                    cache_file = os.path.join(self.config.cache_dir, f"sample_{idx}{cache_suffix}.pt")
                    torch.save(cache_data, cache_file)
                    
                    if pbar is not None:
                        pbar.update(num_processes)
            
            if pbar is not None:
                pbar.close()
        
        self.accelerator.wait_for_everyone()
        
        if self._check_stop():
            return
        
        for idx in samples_to_encode:
            if idx not in self.latent_cache:
                cache_file = os.path.join(self.config.cache_dir, f"sample_{idx}{cache_suffix}.pt")
                if os.path.exists(cache_file):
                    cached = torch.load(cache_file, map_location='cpu', weights_only=True)
                    if self.config.train_text_encoder:
                        self.caption_cache[idx] = cached['caption']
                    else:
                        text_embeds = cached['text_embeds']
                        t5_ids = cached.get('t5_ids', None)
                        self.text_cache[idx] = (text_embeds, t5_ids)
                    self.latent_cache[idx] = cached['latents']
        
        if self.accelerator.is_main_process:
            print(f"Caching complete, {len(self.latent_cache)} samples cached")
        
        if self.config.train_text_encoder:
            del self.vae
            self.vae = None
            torch.cuda.empty_cache()
            gc.collect()
            if self.accelerator.is_main_process:
                print("Unloaded VAE (keeping Text Encoder for training)")
        else:
            mem_before = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            del self.text_encoder, self.tokenizer, self.vae
            self.text_encoder = None
            self.tokenizer = None
            self.vae = None
            torch.cuda.empty_cache()
            gc.collect()
            mem_after = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            freed_memory = mem_before - mem_after
            if self.accelerator.is_main_process:
                print(f"Freed {freed_memory:.2f}GB VRAM after unloading encoders")
    
    def _load_dit(self):
        if self.accelerator.is_main_process:
            print("\nPhase 3: Loading DiT (Anima Transformer)")
            print("=" * 60)
        
        if self._check_stop():
            return
        
        blocks_to_swap = self.config.blocks_to_swap
        loading_device = "cpu" if blocks_to_swap > 0 else self.accelerator.device
        
        if self.accelerator.is_main_process:
            print(f">>> Loading Anima DiT from {self.config.dit_path}...")
            if blocks_to_swap > 0:
                print(f">>> Block Swap: {blocks_to_swap} blocks")
            if self.config.use_adafactor:
                print(f">>> Optimizer: Adafactor (fused)")
        
        from anima_modules.model import load_anima_model
        
        self.dit = load_anima_model(
            dit_path=self.config.dit_path,
            device=loading_device,
            dtype=self.config.dtype,
        )
        
        if blocks_to_swap > 0:
            if self.accelerator.is_main_process:
                print(f">>> Enabling block swap: {blocks_to_swap} blocks")
            self.dit.enable_block_swap(
                blocks_to_swap,
                self.accelerator.device,
                supports_backward=True,
                use_pinned_memory=self.config.use_pinned_memory,
            )
        
        self.dit.requires_grad_(True)
        self.dit.train()
        
        if self.config.gradient_checkpointing:
            if hasattr(self.dit, 'enable_gradient_checkpointing'):
                self.dit.enable_gradient_checkpointing()
                if self.accelerator.is_main_process:
                    print("Gradient Checkpointing enabled")
            elif hasattr(self.dit, 'gradient_checkpointing_enable'):
                self.dit.gradient_checkpointing_enable()
                if self.accelerator.is_main_process:
                    print("Gradient Checkpointing enabled")
        
        self.transformer = self.dit
        
        if self.accelerator.is_main_process:
            total_params = sum(p.numel() for p in self.dit.parameters())
            trainable_params = sum(p.numel() for p in self.dit.parameters() if p.requires_grad)
            print(f"DiT loaded")
            print(f"  - Total params: {total_params:,}")
            print(f"  - Trainable params: {trainable_params:,}")
    
    def _prepare_for_training(self):
        if self.accelerator.is_main_process:
            print("\nPhase 4: Preparing for distributed training")
            print("=" * 60)
        
        blocks_to_swap = self.config.blocks_to_swap
        
        if blocks_to_swap > 0:
            self.dit.move_to_device_except_swap_blocks(self.accelerator.device)
            self.dit.prepare_block_swap_before_forward()
            if self.accelerator.is_main_process:
                print(f"DiT blocks prepared for block swap (num_processes: {self.accelerator.num_processes})")
            
            if self.config.train_text_encoder:
                self.text_encoder = self.accelerator.prepare(self.text_encoder)
                if self.accelerator.is_main_process:
                    print(f"Text Encoder prepared for DDP")
        else:
            if self.config.train_text_encoder:
                self.dit, self.text_encoder = self.accelerator.prepare(self.dit, self.text_encoder)
                if self.accelerator.is_main_process:
                    print(f"DiT + Text Encoder prepared for DDP (num_processes: {self.accelerator.num_processes})")
            else:
                self.dit = self.accelerator.prepare(self.dit)
                if self.accelerator.is_main_process:
                    print(f"DiT prepared for DDP (num_processes: {self.accelerator.num_processes})")
        
        self.transformer = self.dit
    
    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        params = [p for p in self.dit.parameters() if p.requires_grad]
        if self.config.train_text_encoder and self.text_encoder is not None:
            params.extend([p for p in self.text_encoder.parameters() if p.requires_grad])
        return params
    
    def create_optimizer(self, trainable_params: List[torch.nn.Parameter]):
        if self.config.use_adafactor:
            from transformers import Adafactor
            from adafactor_fused import patch_adafactor_fused
            
            dit_model = self.accelerator.unwrap_model(self.dit) if hasattr(self.dit, 'module') else self.dit
            dit_params = [p for p in dit_model.parameters() if p.requires_grad]
            
            param_groups = [{"params": dit_params, "lr": self.config.learning_rate}]
            
            if self.config.train_text_encoder and self.text_encoder is not None:
                te_model = self.accelerator.unwrap_model(self.text_encoder) if hasattr(self.text_encoder, 'module') else self.text_encoder
                te_params = [p for p in te_model.parameters() if p.requires_grad]
                param_groups.append({"params": te_params, "lr": self.config.te_learning_rate})
                
                if self.accelerator.is_main_process:
                    print(f"  - DiT params: {sum(p.numel() for p in dit_params):,}, lr={self.config.learning_rate}")
                    print(f"  - TE params: {sum(p.numel() for p in te_params):,}, lr={self.config.te_learning_rate}")
            
            optimizer = Adafactor(
                param_groups,
                lr=None,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
            )
            patch_adafactor_fused(optimizer)
            
            if self.accelerator.is_main_process:
                print(f"  - Optimizer: Adafactor (fused, block swap compatible)")
            
            return optimizer
        else:
            import bitsandbytes as bnb
            
            if self.config.train_text_encoder and self.text_encoder is not None:
                dit_model = self.accelerator.unwrap_model(self.dit) if hasattr(self.dit, 'module') else self.dit
                te_model = self.accelerator.unwrap_model(self.text_encoder) if hasattr(self.text_encoder, 'module') else self.text_encoder
                dit_params = [p for p in dit_model.parameters() if p.requires_grad]
                te_params = [p for p in te_model.parameters() if p.requires_grad]
                
                param_groups = [
                    {"params": dit_params, "lr": self.config.learning_rate},
                    {"params": te_params, "lr": self.config.te_learning_rate},
                ]
                
                if self.accelerator.is_main_process:
                    print(f"  - DiT params: {sum(p.numel() for p in dit_params):,}, lr={self.config.learning_rate}")
                    print(f"  - TE params: {sum(p.numel() for p in te_params):,}, lr={self.config.te_learning_rate}")
                
                return bnb.optim.AdamW8bit(
                    param_groups,
                    betas=(0.9, 0.999),
                    weight_decay=1e-2,
                    eps=1e-6,
                )
            else:
                return bnb.optim.AdamW8bit(
                    trainable_params,
                    lr=self.config.learning_rate,
                    betas=(0.9, 0.999),
                    weight_decay=1e-2,
                    eps=1e-6,
                )
    
    def train_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        device = self.accelerator.device
        sample_indices = batch['sample_indices']
        batch_size = len(sample_indices)
        
        if hasattr(self, 'last_image_name'):
            self.last_image_name = batch['image_names'][0] if batch['image_names'] else None
        
        latents_list = []
        
        for idx in sample_indices:
            latents_list.append(self.latent_cache[idx])
        
        latents = torch.stack(latents_list, dim=0).to(device, self.config.dtype)
        
        if self.config.train_text_encoder:
            captions = [self.caption_cache[idx] for idx in sample_indices]
            text_embeds, t5_ids = self._encode_text_train(captions)
        else:
            text_embeds_list = []
            t5_ids_list = []
            for idx in sample_indices:
                cached = self.text_cache[idx]
                if isinstance(cached, tuple):
                    text_embeds_list.append(cached[0])
                    t5_ids_list.append(cached[1] if cached[1] is not None else torch.tensor([1]))
                else:
                    text_embeds_list.append(cached)
                    t5_ids_list.append(torch.tensor([1]))
            
            max_text_len = max(e.shape[0] for e in text_embeds_list)
            max_t5_len = max(t.shape[0] for t in t5_ids_list)
            padded_embeds = []
            padded_t5_ids = []
            
            for embeds, t5 in zip(text_embeds_list, t5_ids_list):
                pad_len = max_text_len - embeds.shape[0]
                if pad_len > 0:
                    embeds = F.pad(embeds, (0, 0, 0, pad_len))
                padded_embeds.append(embeds)
                
                t5_pad_len = max_t5_len - t5.shape[0]
                if t5_pad_len > 0:
                    t5 = F.pad(t5, (0, t5_pad_len))
                padded_t5_ids.append(t5)
            
            text_embeds = torch.stack(padded_embeds, dim=0).to(device, self.config.dtype)
            t5_ids = torch.stack(padded_t5_ids, dim=0).to(device)
        
        dit_model = self.accelerator.unwrap_model(self.dit) if hasattr(self.dit, 'module') else self.dit
        context = dit_model.preprocess_text_embeds(text_embeds, t5_ids)
        
        if context.shape[1] < 512:
            context = F.pad(context, (0, 0, 0, 512 - context.shape[1]))
        
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
        
        if self.config.noise_offset > 0:
            noise = noise + self.config.noise_offset * torch.randn(
                (noise.shape[0], noise.shape[1], 1, 1, 1), device=noise.device, dtype=noise.dtype
            )
        
        sigmas = timesteps.float() / 1000.0
        sigmas_broadcast = sigmas.view(-1, 1, 1, 1, 1)
        noisy_latents = (1 - sigmas_broadcast) * latents + sigmas_broadcast * noise
        
        with self.accelerator.autocast():
            model_pred = self.dit(
                noisy_latents,
                timesteps=sigmas,
                context=context,
            )
        
        target = noise - latents
        
        per_sample_loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        per_sample_loss = per_sample_loss.mean(dim=[1, 2, 3, 4])
        
        weighted_loss = per_sample_loss * timestep_weights
        loss = weighted_loss.mean()
        
        del latents, noise, noisy_latents, model_pred, target, per_sample_loss, weighted_loss, context
        
        return loss
    
    def save_checkpoint(self, step: int):
        output_dir = self.config.output_dir or os.path.join(self.script_dir, "anima_output")
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        unwrapped_model = self.accelerator.unwrap_model(self.dit)
        state_dict = unwrapped_model.state_dict()
        
        state_dict_cpu = {k: v.cpu() for k, v in state_dict.items()}
        save_file(state_dict_cpu, os.path.join(checkpoint_dir, "dit_weights.safetensors"))
        
        if self.config.train_text_encoder and self.text_encoder is not None:
            unwrapped_te = self.accelerator.unwrap_model(self.text_encoder)
            te_state_dict = unwrapped_te.state_dict()
            te_state_dict_cpu = {}
            for k, v in te_state_dict.items():
                if not k.startswith('model.'):
                    new_key = 'model.' + k
                else:
                    new_key = k
                te_state_dict_cpu[new_key] = v.cpu()
            save_file(te_state_dict_cpu, os.path.join(checkpoint_dir, "text_encoder_weights.safetensors"))
        
        self.save_accelerate_state(checkpoint_dir, step)
        self._save_config(checkpoint_dir)
    
    def _save_config(self, checkpoint_dir: str):
        if not self.accelerator.is_main_process:
            return
        
        config_dict = {
            "dit_path": self.config.dit_path,
            "vae_path": self.config.vae_path,
            "text_encoder_path": self.config.text_encoder_path,
            "image_folder": self.config.image_folder,
            "output_dir": self.config.output_dir,
            "num_train_steps": self.config.num_train_steps,
            "learning_rate": self.config.learning_rate,
            "te_learning_rate": self.config.te_learning_rate,
            "resolution": self.config.resolution,
            "timestep_type": self.config.timestep_type,
            "shift_scale": self.config.shift_scale,
            "train_text_encoder": self.config.train_text_encoder,
            "training_method": "full",
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
        
        dit_weights_path = os.path.join(checkpoint_dir, "dit_weights.safetensors")
        if os.path.exists(dit_weights_path):
            if self.accelerator.is_main_process:
                print(f"  - Loading DiT weights...")
            
            state_dict = load_file(dit_weights_path)
            unwrapped_model = self.accelerator.unwrap_model(self.dit)
            unwrapped_model.load_state_dict(state_dict, strict=True)
            
            if self.accelerator.is_main_process:
                print(f"  - DiT weights restored")
        
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
        
        output_dir = self.config.output_dir or os.path.join(self.script_dir, "anima_output")
        os.makedirs(output_dir, exist_ok=True)
        
        unwrapped_model = self.accelerator.unwrap_model(self.dit)
        state_dict = unwrapped_model.state_dict()
        state_dict_cpu = {k: v.cpu() for k, v in state_dict.items()}
        
        metadata = {
            "model_type": "anima",
            "training_method": "full",
            "base_model": "CircleStone-Labs/Anima",
            "num_gpus": str(self.accelerator.num_processes),
        }
        
        actual_steps = self.current_step if self.current_step > 0 else self.adjusted_num_train_steps
        equivalent_single_gpu_steps = actual_steps * self.accelerator.num_processes * self.config.gradient_accumulation_steps
        
        model_filename = f"anima_full_{equivalent_single_gpu_steps}steps.safetensors"
        config_filename = f"training_config_{equivalent_single_gpu_steps}steps.json"
        
        save_file(state_dict_cpu, os.path.join(output_dir, model_filename), metadata=metadata)
        
        train_config = {
            "resolution": self.config.resolution,
            "training_method": "full",
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
            json.dump(train_config, f, indent=2)
        
        self.save_checkpoint(actual_steps)
        
        print(f"\nFinal model saved to: {output_dir}/")
        print(f"  - {model_filename}")
        print(f"  - {config_filename}")
        print(f"Final checkpoint saved to: {checkpoint_dir}/checkpoint-{actual_steps}/")
        
        if self.should_stop:
            print(f"\nTraining stopped at step {actual_steps}")
        else:
            print(f"\nTraining complete!")
        
        print(f"  Dataset: {len(self.dataset)} images")
        print(f"  GPUs used: {self.accelerator.num_processes}")
    
    def pre_training_hook(self):
        if self.accelerator.is_main_process:
            print(f"\nAnima Full Training")
            if self.config.train_text_encoder:
                print(f"  - Training: DiT + Text Encoder")
            else:
                print(f"  - Training: DiT only")
            print(f"Timestep sampling: {self.config.timestep_type}")
            if self.config.timestep_type == "sigmoid":
                print(f"  - sigmoid_scale: {self.config.sigmoid_scale}")
            elif self.config.timestep_type == "shift":
                print(f"  - shift_scale: {self.config.shift_scale}")
            elif self.config.timestep_type == "lognorm_blend":
                print(f"  - lognorm_alpha: {self.config.lognorm_alpha}")
            if self.config.gradient_checkpointing:
                print(f"  - Gradient Checkpointing: Enabled")


def main():
    import argparse
    
    fix_windows_encoding()
    
    parser = argparse.ArgumentParser(description="Anima Full Trainer")
    
    parser.add_argument("--dit_path", type=str, required=True, help="Anima DiT model path")
    parser.add_argument("--vae_path", type=str, required=True, help="VAE model path (Qwen-Image VAE)")
    parser.add_argument("--text_encoder_path", type=str, required=True, help="Qwen3 0.6B text encoder path (safetensors)")
    parser.add_argument("--qwen_tokenizer_path", type=str, default=None, help="Qwen tokenizer path (HuggingFace model id or local path, default: Qwen/Qwen2.5-0.5B)")
    parser.add_argument("--image_folder", type=str, required=True, help="Image folder path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    
    parser.add_argument("--num_train_steps", type=int, default=5000, help="Training steps")
    parser.add_argument("--checkpoint_every_n_steps", type=int, default=500, help="Checkpoint interval")
    parser.add_argument("--checkpoints_total_limit", type=int, default=3, help="Max checkpoints")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--lr_warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--resolution", type=int, default=1024, help="Image resolution")
    
    parser.add_argument("--timestep_type", type=str, default="shift", choices=TIMESTEP_TYPES, help="Timestep sampling type")
    parser.add_argument("--sigmoid_scale", type=float, default=1.0, help="Sigmoid scale")
    parser.add_argument("--shift_scale", type=float, default=3.0, help="Shift scale (default 3.0 for Anima)")
    parser.add_argument("--lognorm_alpha", type=float, default=0.75, help="Lognorm alpha")
    
    parser.add_argument("--use_caption", action="store_true", default=True, help="Use captions")
    parser.add_argument("--caption_ext", type=str, default=".txt", help="Caption file extension")
    parser.add_argument("--default_caption", type=str, default="", help="Default caption")
    parser.add_argument("--noise_offset", type=float, default=0.0, help="Noise offset")
    
    parser.add_argument("--train_text_encoder", action="store_true", default=False, help="Train text encoder")
    parser.add_argument("--te_learning_rate", type=float, default=1e-6, help="Text encoder learning rate")
    
    parser.add_argument("--blocks_to_swap", type=int, default=0, help="Number of blocks to swap (0=disabled)")
    parser.add_argument("--use_adafactor", action="store_true", default=False, help="Use Adafactor optimizer (recommended with block swap)")
    parser.add_argument("--use_pinned_memory", action="store_true", default=False, help="Use pinned memory for block swap")
    
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume checkpoint")
    
    args = parser.parse_args()
    
    config = AnimaConfig(
        dit_path=args.dit_path,
        vae_path=args.vae_path,
        text_encoder_path=args.text_encoder_path,
        qwen_tokenizer_path=args.qwen_tokenizer_path,
        image_folder=args.image_folder,
        output_dir=args.output_dir,
        num_train_steps=args.num_train_steps,
        checkpoint_every_n_steps=args.checkpoint_every_n_steps,
        checkpoints_total_limit=args.checkpoints_total_limit,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_warmup_steps=args.lr_warmup_steps,
        resolution=args.resolution,
        timestep_type=args.timestep_type,
        sigmoid_scale=args.sigmoid_scale,
        shift_scale=args.shift_scale,
        lognorm_alpha=args.lognorm_alpha,
        use_caption=args.use_caption,
        caption_ext=args.caption_ext,
        default_caption=args.default_caption,
        noise_offset=args.noise_offset,
        train_text_encoder=args.train_text_encoder,
        te_learning_rate=args.te_learning_rate,
        blocks_to_swap=args.blocks_to_swap,
        use_adafactor=args.use_adafactor,
        use_pinned_memory=args.use_pinned_memory,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        seed=args.seed,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    
    trainer = AnimaTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
