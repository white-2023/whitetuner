"""
ZImage-V2 训练器

基于双流架构的 ZImage Transformer:
1. 双流 Block - 图像和文本有独立 FFN
2. 单流 Block - 后期融合处理
3. 独立的 img/txt modulation
"""

import os
import gc
import math
import json
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from PIL.ImageOps import exif_transpose
from tqdm import tqdm

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer, Qwen3ForCausalLM

from zimage_modules.zimage_model_v2 import ZImageV2Transformer2DModel, load_zimage_v2_transformer

from base_trainer import (
    BaseTrainer,
    BaseTrainerConfig,
    fix_windows_encoding,
    sample_timesteps,
    compute_loss_weighting,
    TIMESTEP_TYPES,
    TimestepType,
    LOSS_WEIGHTING_TYPES,
    LossWeightingType,
)


ZIMAGE_SCHEDULER_CONFIG = {
    "num_train_timesteps": 1000,
    "use_dynamic_shifting": False,
    "shift": 3.0,
}


class ZImageV2Config(BaseTrainerConfig):
    def __init__(
        self,
        image_folder: str,
        output_dir: str,
        model_id: str = "Tongyi-MAI/Z-Image-Turbo",
        num_train_steps: int = 5000,
        checkpoint_every_n_steps: int = 500,
        checkpoints_total_limit: int = 3,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        lr_warmup_steps: int = 100,
        resolution: int = 1024,
        timestep_type: TimestepType = "sigmoid",
        sigmoid_scale: float = 1.0,
        shift_scale: float = 3.0,
        lognorm_alpha: float = 0.75,
        min_timestep: int = None,
        max_timestep: int = None,
        loss_weighting_scheme: LossWeightingType = "none",
        use_caption: bool = True,
        caption_ext: str = ".txt",
        default_caption: str = "",
        prompt_dropout_prob: float = 0.1,
        noise_offset: float = 0.0,
        cache_dir: str = None,
        cache_latents: bool = True,
        use_tensorboard: bool = True,
        tensorboard_dir: str = None,
        gradient_accumulation_steps: int = 1,
        mixed_precision: str = "bf16",
        seed: int = 42,
        max_grad_norm: float = 1.0,
        resume_from_checkpoint: str = None,
        n_double_layers: int = 15,
        n_single_layers: int = 15,
        **kwargs,
    ):
        super().__init__(
            model_id=model_id,
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
        self.timestep_type = timestep_type
        self.sigmoid_scale = sigmoid_scale
        self.shift_scale = shift_scale
        self.lognorm_alpha = lognorm_alpha
        self.min_timestep = min_timestep
        self.max_timestep = max_timestep
        self.loss_weighting_scheme = loss_weighting_scheme
        self.use_caption = use_caption
        self.caption_ext = caption_ext
        self.default_caption = default_caption
        self.prompt_dropout_prob = prompt_dropout_prob
        self.noise_offset = noise_offset
        self.cache_latents = cache_latents
        self.n_double_layers = n_double_layers
        self.n_single_layers = n_single_layers
        
        if cache_dir is None:
            self.cache_dir = os.path.join(image_folder, ".zimage_v2_cache")
        else:
            self.cache_dir = cache_dir


def verify_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False


class ZImageV2Dataset(Dataset):
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
        
        supported_exts = ('.jpg', '.jpeg', '.png', '.webp')
        
        self.samples = []
        skipped_corrupted = 0
        corrupted_files = []
        
        for f in os.listdir(image_folder):
            if f.lower().endswith(supported_exts):
                image_path = os.path.join(image_folder, f)
                
                if not verify_image(image_path):
                    skipped_corrupted += 1
                    corrupted_files.append(image_path)
                    continue
                
                caption = default_caption
                if use_caption:
                    base_name = os.path.splitext(f)[0]
                    caption_path = os.path.join(image_folder, base_name + caption_ext)
                    if os.path.exists(caption_path):
                        with open(caption_path, 'r', encoding='utf-8') as cf:
                            caption = cf.read().strip()
                
                self.samples.append({
                    'image_path': image_path,
                    'caption': caption,
                })
        
        if verbose:
            print(f"[ZImage-V2] 找到 {len(self.samples)} 张有效图片")
            if use_caption:
                with_caption = sum(1 for s in self.samples if s['caption'] != default_caption)
                print(f"  - 有 caption: {with_caption}")
                print(f"  - 使用默认 caption: {len(self.samples) - with_caption}")
            if skipped_corrupted > 0:
                print(f"  - 跳过 {skipped_corrupted} 张损坏的图片")
        
        if len(self.samples) == 0:
            raise ValueError(f"未找到有效图片，请检查路径: {image_folder}")
        
        self.transform = transforms.Compose([transforms.ToTensor()])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        image = exif_transpose(Image.open(sample['image_path'])).convert('RGB')
        orig_w, orig_h = image.size
        
        scale = self.resolution / max(orig_w, orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        new_w = (new_w // 32) * 32
        new_h = (new_h // 32) * 32
        
        image = image.resize((new_w, new_h), Image.BICUBIC)
        image_tensor = self.transform(image)
        
        return {
            'image': image_tensor,
            'caption': sample['caption'],
            'image_path': sample['image_path'],
            'sample_idx': idx,
        }


class ZImageV2Trainer(BaseTrainer):
    def __init__(self, config: ZImageV2Config):
        super().__init__(config)
        self.config: ZImageV2Config = config
        
        self.latent_shift = 0.0
        self.latent_scale = 1.0
        
        self.text_embeds_cache = {}
        self.null_prompt_embeds = None
    
    def _check_stop(self) -> bool:
        return self.should_stop
    
    def create_dataset(self):
        if self.accelerator.is_main_process:
            print("\n创建 ZImage-V2 数据集")
            print("=" * 60)
        
        self.dataset = ZImageV2Dataset(
            image_folder=self.config.image_folder,
            resolution=self.config.resolution,
            use_caption=self.config.use_caption,
            caption_ext=self.config.caption_ext,
            default_caption=self.config.default_caption,
            dtype=self.config.dtype,
            verbose=self.accelerator.is_main_process,
        )
        
        def collate_fn(batch):
            images = torch.stack([item['image'] for item in batch])
            captions = [item['caption'] for item in batch]
            sample_indices = [item['sample_idx'] for item in batch]
            
            return {
                'image': images,
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
            print(f"[ZImage-V2] DataLoader 创建完成，batch_size={self.config.batch_size}")
    
    def load_models(self):
        if self._check_stop():
            return
        
        self._load_text_encoder_and_vae()
        if self._check_stop():
            return
        
        self._cache_embeddings_and_latents()
        if self._check_stop():
            return
        
        self._load_transformer()
        if self._check_stop():
            return
        
        self._prepare_for_ddp()
    
    def _load_text_encoder_and_vae(self):
        if self.accelerator.is_main_process:
            print("\n阶段 1: 加载 Text Encoder 和 VAE")
            print("=" * 60)
        
        if self._check_stop():
            if self.accelerator.is_main_process:
                print("检测到停止信号，跳过加载 Text Encoder 和 VAE")
            return
        
        model_path = self.config.model_id
        
        if self.accelerator.is_main_process:
            print(">>> 加载 Text Encoder (Qwen3)...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            subfolder="tokenizer",
        )
        
        self.text_encoder = Qwen3ForCausalLM.from_pretrained(
            model_path,
            subfolder="text_encoder",
            torch_dtype=torch.float32,
        )
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        self.text_encoder.to(self.accelerator.device)
        
        if self.accelerator.is_main_process:
            print("[ZImage-V2] Text Encoder 加载完成")
        
        if self._check_stop():
            return
        
        if self.accelerator.is_main_process:
            print("\n>>> 加载 VAE...")
        
        self.vae = AutoencoderKL.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae.to(self.accelerator.device)
        
        if self.accelerator.is_main_process:
            print("[ZImage-V2] VAE 加载完成")
        
        self.flush_memory()
    
    def _cache_embeddings_and_latents(self):
        if self.accelerator.is_main_process:
            print("\n阶段 2: 缓存 Text Embeddings 和 Latents")
            print("=" * 60)
        
        if self._check_stop():
            if self.accelerator.is_main_process:
                print("检测到停止信号，跳过缓存")
            return
        
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            os.makedirs(self.config.cache_dir, exist_ok=True)
        
        self.accelerator.wait_for_everyone()
        
        null_cache_file = os.path.join(self.config.cache_dir, "null_prompt.pt")
        if os.path.exists(null_cache_file):
            self.null_prompt_embeds = torch.load(null_cache_file, map_location='cpu')
            if self.accelerator.is_main_process:
                print(f"已加载空 prompt embedding 缓存")
        else:
            if self.accelerator.is_main_process:
                print("编码空 prompt embedding (用于 prompt dropout)...")
            with torch.no_grad():
                messages = [{"role": "user", "content": ""}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                inputs = self.tokenizer(
                    formatted_prompt,
                    padding="max_length",
                    max_length=512,
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = inputs.input_ids.to(self.accelerator.device)
                attention_mask = inputs.attention_mask.to(self.accelerator.device).bool()
                outputs = self.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                self.null_prompt_embeds = outputs.hidden_states[-2].float().cpu()
                valid_length = attention_mask.sum().item()
                self.null_prompt_embeds = self.null_prompt_embeds[:, :valid_length, :]
                if self.accelerator.is_main_process:
                    torch.save(self.null_prompt_embeds, null_cache_file)
                    print(f"空 prompt embedding 已缓存，shape: {self.null_prompt_embeds.shape}")
        
        samples_to_encode = []
        for idx in range(len(self.dataset)):
            cache_file = os.path.join(self.config.cache_dir, f"sample_{idx}.pt")
            if os.path.exists(cache_file):
                self.text_embeds_cache[idx] = torch.load(cache_file, map_location='cpu')
            else:
                samples_to_encode.append(idx)
        
        num_processes = self.accelerator.num_processes
        process_index = self.accelerator.process_index
        
        if self.accelerator.is_main_process:
            if len(self.text_embeds_cache) == 0:
                print(f"没有缓存，现在开始创建缓存 ({len(samples_to_encode)} 个样本)")
            else:
                print(f"已加载 {len(self.text_embeds_cache)} 个缓存，需要编码 {len(samples_to_encode)} 个样本")
            print(f"使用 {num_processes} 个 GPU 进行分布式缓存")
        
        if len(samples_to_encode) > 0:
            my_samples = []
            for i, idx in enumerate(samples_to_encode):
                if i % num_processes == process_index:
                    my_samples.append(idx)
            
            cache_batch_size = 4
            
            pbar = None
            if self.accelerator.is_main_process:
                pbar = tqdm(total=len(samples_to_encode), desc="缓存 embeddings 和 latents")
            
            with torch.no_grad():
                for batch_start in range(0, len(my_samples), cache_batch_size):
                    if self._check_stop():
                        if self.accelerator.is_main_process:
                            print(f"\n检测到停止信号，停止缓存")
                        break
                    
                    batch_indices = my_samples[batch_start:batch_start + cache_batch_size]
                    
                    for idx in batch_indices:
                        sample = self.dataset[idx]
                        
                        caption = sample['caption']
                        messages = [{"role": "user", "content": caption}]
                        formatted_prompt = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=True,
                        )
                        
                        inputs = self.tokenizer(
                            formatted_prompt,
                            padding="max_length",
                            max_length=512,
                            truncation=True,
                            return_tensors="pt",
                        )
                        
                        input_ids = inputs.input_ids.to(self.accelerator.device)
                        attention_mask = inputs.attention_mask.to(self.accelerator.device).bool()
                        
                        outputs = self.text_encoder(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                        )
                        prompt_embeds = outputs.hidden_states[-2].float().cpu()
                        valid_length = attention_mask.sum().item()
                        prompt_embeds = prompt_embeds[:, :valid_length, :]
                        
                        image = sample['image'].unsqueeze(0).to(self.accelerator.device, torch.float32)
                        image_normalized = image * 2 - 1
                        latents = self.vae.encode(image_normalized).latent_dist.sample()
                        latents = ((latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor).float().cpu()
                        
                        cache_data = {
                            'prompt_embeds': prompt_embeds,
                            'prompt_ori_len': valid_length,
                            'latents': latents if self.config.cache_latents else None,
                        }
                        
                        torch.save(cache_data, os.path.join(self.config.cache_dir, f"sample_{idx}.pt"))
                        self.text_embeds_cache[idx] = cache_data
                    
                    if pbar is not None:
                        total_done = (batch_start + len(batch_indices)) * num_processes
                        total_done = min(total_done, len(samples_to_encode))
                        pbar.n = total_done
                        pbar.refresh()
            
            if pbar is not None:
                pbar.close()
        
        self.accelerator.wait_for_everyone()
        
        if self._check_stop():
            return
        
        for idx in samples_to_encode:
            if idx not in self.text_embeds_cache:
                cache_file = os.path.join(self.config.cache_dir, f"sample_{idx}.pt")
                if os.path.exists(cache_file):
                    self.text_embeds_cache[idx] = torch.load(cache_file, map_location='cpu')
        
        if self.accelerator.is_main_process:
            print(f"[ZImage-V2] 缓存完成，共 {len(self.text_embeds_cache)} 个样本")
        
        mem_before = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        del self.text_encoder, self.tokenizer
        self.text_encoder = None
        self.tokenizer = None
        
        if self.config.cache_latents:
            del self.vae
            self.vae = None
            if self.accelerator.is_main_process:
                print("[ZImage-V2] 已卸载 Text Encoder 和 VAE")
        else:
            if self.accelerator.is_main_process:
                print("[ZImage-V2] 已卸载 Text Encoder（保留 VAE 用于实时编码）")
        
        torch.cuda.empty_cache()
        gc.collect()
        mem_after = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        freed_memory = mem_before - mem_after
        if self.accelerator.is_main_process:
            print(f"[ZImage-V2] 释放了 {freed_memory:.2f}GB 显存")
    
    def _load_transformer(self):
        if self.accelerator.is_main_process:
            print("\n阶段 3: 加载 ZImage-V2 双流 Transformer", flush=True)
            print("=" * 60, flush=True)
            print(">>> 使用双流架构加载 Transformer...", flush=True)
        
        if self._check_stop():
            if self.accelerator.is_main_process:
                print("检测到停止信号，跳过加载 Transformer")
            return
        
        dtype = self.config.dtype
        model_path = self.config.model_id
        
        transformer_path = os.path.join(model_path, "transformer")
        if not os.path.exists(transformer_path):
            transformer_path = model_path
        
        self.transformer = load_zimage_v2_transformer(
            transformer_path,
            device="cpu",
            dtype=dtype,
            n_double_layers=self.config.n_double_layers,
            n_single_layers=self.config.n_single_layers,
        )
        
        if self._check_stop():
            return
        
        self.transformer.requires_grad_(True)
        self.transformer.train()
        
        if hasattr(self.transformer, 'enable_gradient_checkpointing'):
            self.transformer.enable_gradient_checkpointing()
            if self.accelerator.is_main_process:
                print("[ZImage-V2] 启用 Gradient Checkpointing")
        
        self.transformer.to(self.accelerator.device)
        
        if self.accelerator.is_main_process:
            total_params = sum(p.numel() for p in self.transformer.parameters())
            trainable_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
            print(f"[ZImage-V2] Transformer 加载完成")
            print(f"  - 总参数: {total_params:,}")
            print(f"  - 可训练参数: {trainable_params:,}")
            print(f"  - 架构: 双流 ({self.config.n_double_layers} double + {self.config.n_single_layers} single)")
        
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(**ZIMAGE_SCHEDULER_CONFIG)
        
        if self.accelerator.is_main_process:
            print(f"\n[ZImage-V2] 调度器: FlowMatchEulerDiscreteScheduler")
            print(f"  - 时间步: {ZIMAGE_SCHEDULER_CONFIG['num_train_timesteps']}")
            print(f"  - Shift: {ZIMAGE_SCHEDULER_CONFIG['shift']}")
    
    def _prepare_for_ddp(self):
        if self.accelerator.is_main_process:
            print("\n阶段 4: 准备分布式训练")
            print("=" * 60)
        
        if self.accelerator.num_processes > 1:
            self.transformer = self.accelerator.prepare(self.transformer)
            if self.accelerator.is_main_process:
                print(f"[ZImage-V2] Transformer 已准备 (DDP: True)")
        else:
            if self.accelerator.is_main_process:
                print(f"[ZImage-V2] Transformer 已准备 (单卡模式，跳过 DDP 包装)")
    
    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        return list(self.transformer.parameters())
    
    def train_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        device = self.accelerator.device
        sample_indices = batch['sample_indices']
        
        batch_size = len(sample_indices)
        
        latents_list = []
        prompt_embeds_list = []
        cap_ori_lens = []
        
        for idx in sample_indices:
            cache_data = self.text_embeds_cache[idx]
            prompt_embeds_list.append(cache_data['prompt_embeds'].squeeze(0).float())
            cap_ori_lens.append(cache_data.get('prompt_ori_len', cache_data['prompt_embeds'].shape[1]))
            
            if self.config.cache_latents and cache_data['latents'] is not None:
                latents_list.append(cache_data['latents'].squeeze(0).float())
        
        if self.config.cache_latents and latents_list:
            latents = torch.stack(latents_list, dim=0).to(device)
        else:
            images = batch['image'].to(device, torch.float32)
            with torch.no_grad():
                images_normalized = images * 2 - 1
                latents = self.vae.encode(images_normalized).latent_dist.sample()
                latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
                latents = latents.float()
        
        if self.config.prompt_dropout_prob > 0 and self.null_prompt_embeds is not None:
            dropout_mask = torch.rand(batch_size) < self.config.prompt_dropout_prob
            null_embed = self.null_prompt_embeds.squeeze(0).float()
            null_len = null_embed.shape[0]
            for i in range(batch_size):
                if dropout_mask[i]:
                    prompt_embeds_list[i] = null_embed
                    cap_ori_lens[i] = null_len
        
        max_len = max(pe.shape[0] for pe in prompt_embeds_list)
        embed_dim = prompt_embeds_list[0].shape[-1]
        prompt_embeds = torch.zeros(
            batch_size, max_len, embed_dim,
            device=device, dtype=torch.float32
        )
        for i, pe in enumerate(prompt_embeds_list):
            prompt_embeds[i, :pe.shape[0]] = pe.to(device)
        
        timesteps, timestep_weights = sample_timesteps(
            batch_size,
            num_train_timesteps=self.noise_scheduler.config.num_train_timesteps,
            device=device,
            timestep_type=self.config.timestep_type,
            sigmoid_scale=self.config.sigmoid_scale,
            shift=self.config.shift_scale,
            lognorm_alpha=self.config.lognorm_alpha,
            min_timestep=self.config.min_timestep,
            max_timestep=self.config.max_timestep,
        )
        
        noise = torch.randn_like(latents)
        
        if self.config.noise_offset > 0:
            noise = noise + self.config.noise_offset * torch.randn(
                (noise.shape[0], noise.shape[1], 1, 1), 
                device=noise.device, 
                dtype=noise.dtype
            )
        
        sigmas = timesteps.float() / self.noise_scheduler.config.num_train_timesteps
        sigmas = sigmas.view(-1, 1, 1, 1)
        noisy_latents = (1 - sigmas) * latents + sigmas * noise
        
        latent_model_input = noisy_latents.to(self.config.dtype).unsqueeze(2)
        
        timestep_model_input = ((1000 - timesteps.float()) / 1000).to(self.config.dtype)
        
        prompt_embeds = prompt_embeds.to(self.config.dtype)
        
        with self.accelerator.autocast():
            model_output = self.transformer(
                latent_model_input,
                timestep_model_input,
                prompt_embeds,
                cap_ori_lens=cap_ori_lens,
            )
        
        noise_pred = model_output.float().squeeze(2)
        noise_pred = -noise_pred
        
        target = (noise - latents).detach()
        
        per_sample_loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
        per_sample_loss = per_sample_loss.mean(dim=[1, 2, 3])
        
        weighted_loss = per_sample_loss * timestep_weights
        
        if self.config.loss_weighting_scheme != "none":
            loss_weights = compute_loss_weighting(
                weighting_scheme=self.config.loss_weighting_scheme,
                timesteps=timesteps,
                num_train_timesteps=self.noise_scheduler.config.num_train_timesteps,
                device=device,
            )
            weighted_loss = weighted_loss * loss_weights
        
        loss = weighted_loss.mean()
        
        return loss
    
    def save_checkpoint(self, step: int):
        output_dir = self.config.output_dir or os.path.join(self.script_dir, "output")
        checkpoint_dir = os.path.join(output_dir, "checkpoints", f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        unwrapped_transformer = self.accelerator.unwrap_model(self.transformer)
        transformer_dir = os.path.join(checkpoint_dir, "transformer")
        unwrapped_transformer.save_pretrained(transformer_dir, safe_serialization=True)
        
        self.save_accelerate_state(checkpoint_dir, step)
        self._save_gui_config(checkpoint_dir)
    
    def _save_gui_config(self, checkpoint_dir: str):
        if not self.accelerator.is_main_process:
            return
        
        gui_config = {
            "model_type": "zimage_v2",
            "model_id": self.config.model_id,
            "image_folder": self.config.image_folder,
            "output_dir": self.config.output_dir,
            "num_train_steps": self.config.num_train_steps,
            "learning_rate": self.config.learning_rate,
            "resolution": self.config.resolution,
            "timestep_type": self.config.timestep_type,
            "sigmoid_scale": self.config.sigmoid_scale,
            "shift_scale": self.config.shift_scale,
            "lognorm_alpha": self.config.lognorm_alpha,
            "min_timestep": self.config.min_timestep,
            "max_timestep": self.config.max_timestep,
            "loss_weighting_scheme": self.config.loss_weighting_scheme,
            "use_caption": self.config.use_caption,
            "default_caption": self.config.default_caption,
            "prompt_dropout_prob": self.config.prompt_dropout_prob,
            "noise_offset": self.config.noise_offset,
            "checkpoint_every_n_steps": self.config.checkpoint_every_n_steps,
            "checkpoints_total_limit": self.config.checkpoints_total_limit,
            "n_double_layers": self.config.n_double_layers,
            "n_single_layers": self.config.n_single_layers,
        }
        gui_config_path = os.path.join(checkpoint_dir, "gui_config.json")
        with open(gui_config_path, "w", encoding="utf-8") as f:
            json.dump(gui_config, f, indent=2, ensure_ascii=False)
    
    def save_final_model(self):
        self.accelerator.wait_for_everyone()
        
        output_dir = self.config.output_dir or os.path.join(self.script_dir, "output")
        actual_steps = self.current_step if self.current_step > 0 else self.adjusted_num_train_steps
        equivalent_steps = actual_steps * self.accelerator.num_processes * self.config.gradient_accumulation_steps
        model_dir = os.path.join(output_dir, f"zimage_v2_transformer_{equivalent_steps}steps")
        
        self.save_accelerate_state(model_dir, actual_steps)
        
        self.accelerator.wait_for_everyone()
        
        if not self.accelerator.is_main_process:
            return
        
        print("\n保存最终 ZImage-V2 模型")
        print("=" * 60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        unwrapped_transformer = self.accelerator.unwrap_model(self.transformer)
        transformer_dir = os.path.join(model_dir, "transformer")
        os.makedirs(transformer_dir, exist_ok=True)
        
        unwrapped_transformer.save_pretrained(transformer_dir, safe_serialization=True)
        
        metadata = {
            "model_type": "zimage_v2",
            "base_model": self.config.model_id,
            "training_method": "full_finetune",
            "architecture": "double_stream",
            "n_double_layers": self.config.n_double_layers,
            "n_single_layers": self.config.n_single_layers,
            "trained_steps": actual_steps,
            "equivalent_single_gpu_steps": equivalent_steps,
            "resolution": self.config.resolution,
            "learning_rate": self.config.learning_rate,
            "timestep_type": self.config.timestep_type,
            "sigmoid_scale": self.config.sigmoid_scale,
            "shift_scale": self.config.shift_scale,
            "lognorm_alpha": self.config.lognorm_alpha,
            "min_timestep": self.config.min_timestep,
            "max_timestep": self.config.max_timestep,
            "loss_weighting_scheme": self.config.loss_weighting_scheme,
            "stopped_early": self.should_stop,
            "num_gpus": self.accelerator.num_processes,
            "batch_size_per_gpu": self.config.batch_size,
            "effective_batch_size": self.effective_batch_size,
        }
        with open(os.path.join(model_dir, "training_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        self._save_gui_config(model_dir)
        
        print(f"\n[ZImage-V2] 最终模型已保存到: {model_dir}/")
        print(f"  - transformer/ (双流架构 transformer 权重)")
        print(f"  - accelerate_state/ (训练状态，可用于继续训练)")
        print(f"  - training_state.json")
        print(f"  - gui_config.json (GUI 参数)")
        print(f"  - training_metadata.json")
        
        if self.should_stop:
            print(f"\n训练在第 {actual_steps} 步停止")
        else:
            print(f"\n训练完成!")
        
        print(f"  数据集: {len(self.dataset)} 张图片")
        print(f"  使用 GPU 数量: {self.accelerator.num_processes}")
        print(f"\n可使用此目录继续训练或替换原模型的 transformer 文件夹")
    
    def pre_training_hook(self):
        if self.accelerator.is_main_process:
            print(f"ZImage-V2 双流架构训练")
            print(f"架构: {self.config.n_double_layers} 双流层 + {self.config.n_single_layers} 单流层")
            print(f"时间步采样: {self.config.timestep_type}")
            if self.config.timestep_type == "sigmoid":
                print(f"  - sigmoid_scale: {self.config.sigmoid_scale}")
            elif self.config.timestep_type == "shift":
                print(f"  - shift_scale: {self.config.shift_scale}")
            elif self.config.timestep_type == "lognorm_blend":
                print(f"  - lognorm_alpha: {self.config.lognorm_alpha}")
            if self.config.min_timestep is not None or self.config.max_timestep is not None:
                t_min = self.config.min_timestep if self.config.min_timestep is not None else 0
                t_max = self.config.max_timestep if self.config.max_timestep is not None else 1000
                print(f"  - 时间步范围: [{t_min}, {t_max})")
            print(f"Loss weighting: {self.config.loss_weighting_scheme}")
            print(f"Prompt dropout 概率: {self.config.prompt_dropout_prob}")
            print(f"Noise offset: {self.config.noise_offset}")


def main():
    import argparse
    
    fix_windows_encoding()
    
    parser = argparse.ArgumentParser(description="ZImage-V2 双流架构训练器")
    
    parser.add_argument("--model_id", type=str, required=True, help="模型路径")
    parser.add_argument("--image_folder", type=str, required=True, help="训练图片文件夹")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    
    parser.add_argument("--num_train_steps", type=int, default=5000, help="训练步数")
    parser.add_argument("--checkpoint_every_n_steps", type=int, default=500, help="检查点保存间隔")
    parser.add_argument("--checkpoints_total_limit", type=int, default=3, help="检查点数量限制")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率")
    parser.add_argument("--lr_warmup_steps", type=int, default=100, help="学习率预热步数")
    parser.add_argument("--resolution", type=int, default=1024, help="图片分辨率")
    
    parser.add_argument("--timestep_type", type=str, default="sigmoid", 
                        choices=TIMESTEP_TYPES, help="时间步采样类型")
    parser.add_argument("--sigmoid_scale", type=float, default=1.0, help="Sigmoid 分布集中程度")
    parser.add_argument("--shift_scale", type=float, default=3.0, help="Shift 采样偏移程度")
    parser.add_argument("--lognorm_alpha", type=float, default=0.75, help="LogNorm 混合比例")
    parser.add_argument("--min_timestep", type=int, default=None, help="最小时间步 (0-999)")
    parser.add_argument("--max_timestep", type=int, default=None, help="最大时间步 (1-1000)")
    
    parser.add_argument("--loss_weighting_scheme", type=str, default="none",
                        choices=LOSS_WEIGHTING_TYPES, help="Loss 权重方案: none/sigma_sqrt/cosmap")
    
    parser.add_argument("--use_caption", action="store_true", default=True, help="使用 caption")
    parser.add_argument("--caption_ext", type=str, default=".txt", help="Caption 文件扩展名")
    parser.add_argument("--default_caption", type=str, default="", help="默认 caption")
    
    parser.add_argument("--prompt_dropout_prob", type=float, default=0.1, help="Prompt dropout 概率 (0-1)")
    
    parser.add_argument("--noise_offset", type=float, default=0.0, help="Noise offset (0-0.1)")
    
    parser.add_argument("--n_double_layers", type=int, default=15, help="双流层数量")
    parser.add_argument("--n_single_layers", type=int, default=15, help="单流层数量")
    
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="混合精度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="从检查点恢复训练")
    
    args = parser.parse_args()
    
    config = ZImageV2Config(
        model_id=args.model_id,
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
        min_timestep=args.min_timestep,
        max_timestep=args.max_timestep,
        loss_weighting_scheme=args.loss_weighting_scheme,
        use_caption=args.use_caption,
        caption_ext=args.caption_ext,
        default_caption=args.default_caption,
        prompt_dropout_prob=args.prompt_dropout_prob,
        noise_offset=args.noise_offset,
        n_double_layers=args.n_double_layers,
        n_single_layers=args.n_single_layers,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        seed=args.seed,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    
    trainer = ZImageV2Trainer(config)
    trainer.run()


if __name__ == "__main__":
    main()

