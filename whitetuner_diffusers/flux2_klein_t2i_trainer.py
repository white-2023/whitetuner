"""
FLUX.2 Klein Text-to-Image Trainer
基于 diffusers 的 FLUX.2 Klein T2I LoKr/全量微调训练器
"""

import os
import gc
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from functools import lru_cache
from collections import OrderedDict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torchvision import transforms
from accelerate import Accelerator
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models import AutoencoderKLFlux2
from flux2_modules import Flux2Transformer2DModel, load_flux2_transformer_from_diffusers
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM
from optimum.quanto import freeze, qfloat8, quantize
from bitsandbytes.optim import AdamW8bit
from safetensors.torch import save_file


class LRUCache:
    def __init__(self, maxsize=1000):
        self.cache = OrderedDict()
        self.maxsize = maxsize
    
    def __contains__(self, key):
        return key in self.cache
    
    def __getitem__(self, key):
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def __setitem__(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        while len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)
    
    def __len__(self):
        return len(self.cache)
    
    def get(self, key, default=None):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return default

from base_trainer import (
    BaseTrainer, 
    BaseTrainerConfig,
    fix_windows_encoding,
    sample_timesteps,
    TIMESTEP_TYPES,
    TimestepType,
)

from lokr import (
    factorization,
    make_kron,
    LokrModule,
    apply_lokr_to_transformer,
)


def verify_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    batch_size, num_channels, height, width = latents.shape
    latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(batch_size, num_channels * 4, height // 2, width // 2)
    return latents


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    batch_size, num_channels, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
    return latents


def unpack_latents_with_ids(x: torch.Tensor, x_ids: torch.Tensor) -> torch.Tensor:
    x_list = []
    for data, pos in zip(x, x_ids):
        _, ch = data.shape
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)

        h = torch.max(h_ids) + 1
        w = torch.max(w_ids) + 1

        flat_ids = h_ids * w + w_ids

        out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

        out = out.view(h, w, ch).permute(2, 0, 1)
        x_list.append(out)

    return torch.stack(x_list, dim=0)


def prepare_latent_ids(latents: torch.Tensor) -> torch.Tensor:
    batch_size, _, height, width = latents.shape

    t = torch.arange(1)
    h = torch.arange(height)
    w = torch.arange(width)
    l = torch.arange(1)
    
    grid = torch.meshgrid(t, h, w, l, indexing='ij')
    ids = torch.stack(grid, dim=-1)
    ids = ids.view(1, -1, 4).expand(batch_size, -1, -1)
    
    return ids


def prepare_text_ids(x: torch.Tensor) -> torch.Tensor:
    B, L, _ = x.shape
    out_ids = []
    for i in range(B):
        t = torch.arange(1)
        h = torch.arange(1)
        w = torch.arange(1)
        l = torch.arange(L)
        coords = torch.cartesian_prod(t, h, w, l)
        out_ids.append(coords)
    return torch.stack(out_ids)


class Flux2KleinT2IConfig(BaseTrainerConfig):
    def __init__(
        self,
        image_folder: str,
        output_dir: str,
        model_id: str = "black-forest-labs/FLUX.2-klein-base-9B",
        num_train_steps: int = 5000,
        checkpoint_every_n_steps: int = 500,
        checkpoints_total_limit: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-4,
        lr_warmup_steps: int = 500,
        resolution: int = 1024,
        timestep_type: TimestepType = "linear",
        sigmoid_scale: float = 1.0,
        shift_scale: float = 3.0,
        lognorm_alpha: float = 0.75,
        full_training: bool = False,
        full_matrix: bool = True,
        lora_dim: int = 10000,
        lora_alpha: int = 1,
        lokr_factor: int = 4,
        decompose_both: bool = False,
        quantize_transformer: bool = True,
        quantize_text_encoder: bool = True,
        blocks_to_swap: int = 0,
        use_pinned_memory: bool = True,
        noise_offset: float = 0.0,
        text_encoder_layers: Tuple[int, ...] = (9, 18, 27),
        max_sequence_length: int = 512,
        use_caption: bool = True,
        caption_ext: str = ".txt",
        default_caption: str = "",
        use_tensorboard: bool = True,
        tensorboard_dir: str = None,
        gradient_accumulation_steps: int = 1,
        mixed_precision: str = "bf16",
        seed: int = 42,
        max_grad_norm: float = 1.0,
        resume_from_checkpoint: str = None,
        enable_buckets: bool = True,
        bucket_resolutions: List[int] = None,
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
            quantize_transformer=quantize_transformer,
            quantize_text_encoder=quantize_text_encoder,
            quantize_level=qfloat8,
            use_tensorboard=use_tensorboard,
            tensorboard_dir=tensorboard_dir,
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            seed=seed,
            max_grad_norm=max_grad_norm,
            resume_from_checkpoint=resume_from_checkpoint,
            **kwargs,
        )
        
        self.image_folder = image_folder
        self.timestep_type = timestep_type
        self.sigmoid_scale = sigmoid_scale
        self.shift_scale = shift_scale
        self.lognorm_alpha = lognorm_alpha
        self.full_training = full_training
        self.full_matrix = full_matrix
        self.lora_dim = lora_dim
        self.lora_alpha = lora_alpha
        self.lokr_factor = lokr_factor
        self.decompose_both = decompose_both
        self.blocks_to_swap = blocks_to_swap
        self.use_pinned_memory = use_pinned_memory
        self.noise_offset = noise_offset
        self.text_encoder_layers = text_encoder_layers
        self.max_sequence_length = max_sequence_length
        self.use_caption = use_caption
        self.caption_ext = caption_ext
        self.default_caption = default_caption
        self.vae_scale_factor = 8
        self.cache_dir = os.path.join(image_folder, ".flux2_klein_t2i_cache")
        self.enable_buckets = enable_buckets
        if bucket_resolutions is None:
            min_bucket = max(512, int(self.resolution * 0.5))
            max_bucket = 2048
            default_buckets = list(range(min_bucket, max_bucket + 1, 128))
            if self.resolution not in default_buckets:
                default_buckets.append(self.resolution)
                default_buckets.sort()
            self.bucket_resolutions = default_buckets
        else:
            self.bucket_resolutions = [int(r) for r in bucket_resolutions if int(r) > 0]


class Flux2KleinT2IDataset(Dataset):
    def __init__(
        self,
        image_folder: str,
        resolution: int = 1024,
        use_caption: bool = True,
        caption_ext: str = ".txt",
        default_caption: str = "",
        verbose: bool = True,
        enable_buckets: bool = True,
        bucket_resolutions: List[int] = None,
    ):
        self.image_folder = image_folder
        self.resolution = resolution
        self.use_caption = use_caption
        self.caption_ext = caption_ext
        self.default_caption = default_caption
        self.enable_buckets = enable_buckets
        self.bucket_resolutions = sorted([int(r) for r in (bucket_resolutions or []) if int(r) > 0])
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        self.samples = []
        corrupted_files = []
        
        for filename in sorted(os.listdir(image_folder)):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in image_extensions:
                continue
            
            image_path = os.path.join(image_folder, filename)
            
            caption = default_caption if default_caption else os.path.splitext(filename)[0]
            if use_caption:
                caption_path = os.path.splitext(image_path)[0] + caption_ext
                if os.path.exists(caption_path):
                    with open(caption_path, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
            
            if not verify_image(image_path):
                corrupted_files.append(image_path)
                continue
            
            with Image.open(image_path) as img:
                img = exif_transpose(img)
                w, h = img.size
            
            self.samples.append({
                'image_path': image_path,
                'caption': caption,
                'width': w,
                'height': h,
            })
        
        if verbose:
            print(f"找到 {len(self.samples)} 张有效图片")
            if corrupted_files:
                print(f"  - 跳过 {len(corrupted_files)} 张损坏的图片")
        
        if len(self.samples) == 0:
            raise ValueError(f"未找到有效的图片: {image_folder}")
        
        self.transform = transforms.Compose([transforms.ToTensor()])
        self._use_cache = False
        
        if self.enable_buckets and len(self.bucket_resolutions) > 0:
            if verbose:
                print(f"使用 bucket 训练，分辨率列表: {self.bucket_resolutions}")
                print(f"目标训练分辨率: {self.resolution}x{self.resolution} (面积: {self.resolution**2})")
            
            target_area = self.resolution * self.resolution
            
            for sample in self.samples:
                orig_w, orig_h = sample['width'], sample['height']
                
                best_bucket = None
                best_area_diff = float('inf')
                best_final_w = None
                best_final_h = None
                
                for bucket_res in self.bucket_resolutions:
                    max_side = max(orig_w, orig_h)
                    scale = bucket_res / max_side
                    new_w = int(orig_w * scale)
                    new_h = int(orig_h * scale)
                    
                    new_w = max(16, (new_w // 16) * 16)
                    new_h = max(16, (new_h // 16) * 16)
                    
                    final_area = new_w * new_h
                    area_diff = abs(final_area - target_area)
                    
                    if area_diff < best_area_diff:
                        best_area_diff = area_diff
                        best_bucket = bucket_res
                        best_final_w = new_w
                        best_final_h = new_h
                
                sample['bucket_res'] = int(best_bucket)
                sample['bucket_w'] = int(best_final_w)
                sample['bucket_h'] = int(best_final_h)
            
            if verbose:
                areas = [s['bucket_w'] * s['bucket_h'] for s in self.samples]
                print(f"  实际训练面积范围: {min(areas):,} ~ {max(areas):,}")
                print(f"  目标面积: {target_area:,}")
                print(f"  平均面积: {sum(areas) // len(areas):,}")
        else:
            for sample in self.samples:
                max_side = max(sample['width'], sample['height'])
                scale = self.resolution / max_side
                new_w = int(sample['width'] * scale)
                new_h = int(sample['height'] * scale)
                
                new_w = max(16, (new_w // 16) * 16)
                new_h = max(16, (new_h // 16) * 16)
                
                sample['bucket_res'] = int(self.resolution)
                sample['bucket_w'] = int(new_w)
                sample['bucket_h'] = int(new_h)
    
    def set_use_cache(self, use_cache: bool):
        self._use_cache = use_cache
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._use_cache:
            return {'sample_idx': idx}
        
        sample = self.samples[idx]
        image_path = sample['image_path']
        
        img = exif_transpose(Image.open(image_path)).convert('RGB')
        
        new_w = sample['bucket_w']
        new_h = sample['bucket_h']
        
        img = img.resize((new_w, new_h), Image.BICUBIC)
        img_tensor = self.transform(img)
        
        return {
            'image': img_tensor,
            'caption': sample['caption'],
            'sample_idx': idx,
        }


class ResolutionBucketBatchSampler(BatchSampler):
    def __init__(self, dataset: Flux2KleinT2IDataset, batch_size: int, shuffle: bool = True, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        from collections import defaultdict
        import random

        buckets = defaultdict(list)
        for idx, s in enumerate(self.dataset.samples):
            buckets[(s['bucket_h'], s['bucket_w'])].append(idx)

        bucket_keys = list(buckets.keys())
        if self.shuffle:
            random.shuffle(bucket_keys)

        for key in bucket_keys:
            indices = buckets[key]
            if self.shuffle:
                random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self):
        from collections import defaultdict
        buckets = defaultdict(int)
        for s in self.dataset.samples:
            buckets[(s['bucket_h'], s['bucket_w'])] += 1
        total = 0
        for count in buckets.values():
            if self.drop_last:
                total += count // self.batch_size
            else:
                total += (count + self.batch_size - 1) // self.batch_size
        return total


class Flux2KleinT2ITrainer(BaseTrainer):
    def __init__(self, config: Flux2KleinT2IConfig):
        super().__init__(config)
        self.config: Flux2KleinT2IConfig = config
        
        self.lokr_modules = None
        self.lokr_module_names = None
        
        self.cache_maxsize = 2000
        self.prompt_embeds_cache = LRUCache(maxsize=self.cache_maxsize)
        self.latent_cache = LRUCache(maxsize=self.cache_maxsize)
        self.cache_ready = False
        
        self.latents_bn_mean = None
        self.latents_bn_std = None
        
        self.use_fused_backward = False
    
    def _check_stop(self, stage: str = None) -> bool:
        return self.check_stop(stage)
    
    def load_models(self):
        self._load_text_encoder()
        if self._check_stop():
            return
        
        self._load_vae_and_transformer()
        if self._check_stop():
            return
        
        self._cache_embeddings_and_latents()
        if self._check_stop():
            return
        
        if not self.config.full_training:
            self._apply_lokr()
        else:
            self._prepare_full_training()
    
    def _load_text_encoder(self):
        if self.accelerator.is_main_process:
            print("\n阶段 1: 加载 Text Encoder")
            print("=" * 60)
        
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(
            self.config.model_id,
            subfolder="tokenizer",
        )
        
        if self.accelerator.is_main_process:
            print("Tokenizer 加载完成")
        
        self.text_encoder = Qwen3ForCausalLM.from_pretrained(
            self.config.model_id,
            subfolder="text_encoder",
            torch_dtype=self.config.dtype,
            low_cpu_mem_usage=True,
        )
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        
        if self.config.quantize_text_encoder:
            if self.accelerator.is_main_process:
                print(">>> 量化 Text Encoder...")
            exclude_patterns = ["*embed*", "*lm_head*"]
            quantize(self.text_encoder, weights=self.config.quantize_level, exclude=exclude_patterns)
            freeze(self.text_encoder)
        
        self.text_encoder.to(self.accelerator.device)
        
        if self.accelerator.is_main_process:
            print("Text Encoder (Qwen3) 加载完成")
    
    def _encode_prompt(self, prompt: str) -> Dict[str, torch.Tensor]:
        device = self.text_encoder.device
        dtype = self.text_encoder.dtype
        
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.config.max_sequence_length,
        )
        
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            output = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        
        hidden_states = torch.stack(
            [output.hidden_states[k] for k in self.config.text_encoder_layers], 
            dim=1
        )
        hidden_states = hidden_states.to(dtype=dtype, device=device)
        
        batch_size, num_layers, seq_len, hidden_dim = hidden_states.shape
        prompt_embeds = hidden_states.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_layers * hidden_dim)
        
        return {
            'prompt_embeds': prompt_embeds.cpu(),
            'attention_mask': attention_mask.cpu(),
        }
    
    def _load_vae_and_transformer(self):
        if self.accelerator.is_main_process:
            print("\n阶段 2: 加载 VAE 和 Transformer")
            print("=" * 60)
        
        self.vae = AutoencoderKLFlux2.from_pretrained(
            self.config.model_id,
            subfolder="vae",
            torch_dtype=self.config.dtype,
        )
        self.vae.requires_grad_(False)
        self.vae.eval()
        
        self.latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1)
        self.latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps)
        
        self.vae.to(self.accelerator.device)
        
        if self.accelerator.is_main_process:
            print("VAE 加载完成")
        
        if self.accelerator.is_main_process:
            print(">>> 加载 Transformer...")
        
        self.transformer = load_flux2_transformer_from_diffusers(
            self.config.model_id,
            subfolder="transformer",
            torch_dtype=self.config.dtype,
            device="cpu",
        )
        
        blocks_to_swap = self.config.blocks_to_swap
        
        if self.config.full_training:
            self.transformer.requires_grad_(True)
            if blocks_to_swap == 0:
                self.transformer.to(self.accelerator.device)
            if self.accelerator.is_main_process:
                print(">>> Full Training 模式：transformer 参数可训练")
                trainable_count = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
                total_count = sum(p.numel() for p in self.transformer.parameters())
                param_tensor_count = len([p for p in self.transformer.parameters() if p.requires_grad])
                print(f"    可训练参数: {trainable_count:,} / {total_count:,}")
                print(f"    参数 tensor 数量: {param_tensor_count}")
        elif self.config.quantize_transformer and blocks_to_swap == 0:
            if self.accelerator.is_main_process:
                print("\n>>> LoKr 模式：使用 quanto qfloat8 量化 Transformer...")
            
            exclude_patterns = ["*norm*", "proj_out*", "*embedder*"]
            
            all_blocks = list(self.transformer.transformer_blocks) + list(self.transformer.single_transformer_blocks)
            total_blocks = len(all_blocks)
            if self.accelerator.is_main_process:
                print(f"    共 {total_blocks} 个 blocks (double: {len(self.transformer.transformer_blocks)}, single: {len(self.transformer.single_transformer_blocks)})")
            
            from tqdm import tqdm
            for block in tqdm(all_blocks, desc="量化 blocks", disable=not self.accelerator.is_main_process):
                block.to(self.accelerator.device, dtype=self.config.dtype, non_blocking=True)
                quantize(block, weights=self.config.quantize_level, exclude=exclude_patterns)
                freeze(block)
                block.to("cpu", non_blocking=True)
            
            if self.accelerator.is_main_process:
                print("    正在量化其他模块...")
            quantize(self.transformer, weights=self.config.quantize_level, exclude=exclude_patterns)
            freeze(self.transformer)
            
            self.transformer.requires_grad_(False)
            self.transformer.to(self.accelerator.device)
            
            if self.accelerator.is_main_process:
                print(">>> qfloat8 量化完成")
        else:
            self.transformer.requires_grad_(False)
            if blocks_to_swap == 0:
                self.transformer.to(self.accelerator.device)
                if self.accelerator.is_main_process:
                    print(">>> Transformer 未量化，已移至 GPU")
        
        if blocks_to_swap > 0:
            if self.accelerator.is_main_process:
                print(f">>> 启用 block swap: {blocks_to_swap} blocks")
            self.transformer.enable_block_swap(
                blocks_to_swap,
                self.accelerator.device,
                supports_backward=True,
                use_pinned_memory=self.config.use_pinned_memory,
            )
            if self.accelerator.is_main_process:
                total_blocks = self.transformer.num_blocks
                blocks_on_gpu = total_blocks - blocks_to_swap
                print(f">>> Block swap 已启用: {blocks_on_gpu} blocks 在 GPU, {blocks_to_swap} blocks 在 CPU")
        
        if hasattr(self.transformer, 'enable_gradient_checkpointing'):
            # Activation CPU Offloading 只在 block swap 时有明显效果
            # 对于非 block swap 模式，主要显存被参数+梯度占用，activation offload 效果有限
            use_cpu_offload = blocks_to_swap > 0
            self.transformer.enable_gradient_checkpointing(activation_cpu_offloading=use_cpu_offload)
            if self.accelerator.is_main_process:
                if use_cpu_offload:
                    print("✓ 启用 Gradient Checkpointing (with Activation CPU Offloading)")
                else:
                    print("✓ 启用 Gradient Checkpointing")
        
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.config.model_id,
            subfolder="scheduler",
        )
        
        if self.accelerator.is_main_process:
            print("Transformer 和 Scheduler 加载完成")
    
    def _load_cache_item(self, idx: int):
        if idx in self.latent_cache and idx in self.prompt_embeds_cache:
            return
        cache_file = os.path.join(self.config.cache_dir, f"sample_{idx}.pt")
        cached = torch.load(cache_file, map_location='cpu', weights_only=True)
        self.prompt_embeds_cache[idx] = {
            'prompt_embeds': cached['prompt_embeds'],
            'attention_mask': cached['attention_mask'],
        }
        self.latent_cache[idx] = cached['latent']
    
    def _cache_embeddings_and_latents(self):
        if self.accelerator.is_main_process:
            print("\n阶段 3: 缓存 Text Embeddings 和 Latents")
            print("=" * 60)
            os.makedirs(self.config.cache_dir, exist_ok=True)
        
        self.accelerator.wait_for_everyone()
        
        samples_to_encode = []
        cached_count = 0
        for idx in range(len(self.dataset)):
            cache_file = os.path.join(self.config.cache_dir, f"sample_{idx}.pt")
            if os.path.exists(cache_file):
                cached_count += 1
            else:
                samples_to_encode.append(idx)
        
        if self.accelerator.is_main_process:
            print(f"  已有磁盘缓存: {cached_count} 个")
            print(f"  需要编码: {len(samples_to_encode)} 个")
            print(f"  LRU 内存缓存大小: {self.cache_maxsize} 个样本")
        
        if len(samples_to_encode) == 0:
            if self.accelerator.is_main_process:
                print("全部磁盘缓存已就绪，训练时按需加载到内存")
            self.cache_ready = True
            self.dataset.set_use_cache(True)
            self._unload_text_encoder()
            self._unload_vae()
            return
        
        device = self.accelerator.device
        dtype = self.config.dtype
        
        bn_mean = self.latents_bn_mean.to(device, dtype)
        bn_std = self.latents_bn_std.to(device, dtype)
        
        from tqdm import tqdm
        from collections import defaultdict
        
        num_processes = self.accelerator.num_processes
        process_index = self.accelerator.process_index
        
        my_samples = [s for i, s in enumerate(samples_to_encode) if i % num_processes == process_index]
        
        resolution_buckets = defaultdict(list)
        for idx in my_samples:
            sample = self.dataset[idx]
            bucket_h = self.dataset.samples[idx]['bucket_h']
            bucket_w = self.dataset.samples[idx]['bucket_w']
            resolution_buckets[(bucket_h, bucket_w)].append((idx, sample))
        
        if self.accelerator.is_main_process:
            print(f"  分桶: {len(resolution_buckets)} 种分辨率")
        
        pbar = None
        if self.accelerator.is_main_process:
            pbar = tqdm(total=len(samples_to_encode), desc="缓存 embeddings 和 latents")
        
        batch_size = 4
        
        with torch.no_grad():
            for resolution, bucket_items in resolution_buckets.items():
                for batch_start in range(0, len(bucket_items), batch_size):
                    if self._check_stop():
                        return
                    
                    batch_items = bucket_items[batch_start:batch_start + batch_size]
                    batch_indices = [item[0] for item in batch_items]
                    batch_samples = [item[1] for item in batch_items]
                    
                    image_batch = torch.stack([s['image'] for s in batch_samples]).to(device, dtype)
                    image_batch = image_batch * 2 - 1
                    
                    latents = self.vae.encode(image_batch).latent_dist.mode()
                    latents = patchify_latents(latents)
                    latents = (latents - bn_mean) / bn_std
                    
                    for i, idx in enumerate(batch_indices):
                        caption = batch_samples[i]['caption']
                        text_data = self._encode_prompt(caption)
                        
                        latent_cpu = latents[i:i+1].cpu()
                        
                        cache_data = {
                            'prompt_embeds': text_data['prompt_embeds'],
                            'attention_mask': text_data['attention_mask'],
                            'latent': latent_cpu,
                        }
                        cache_file = os.path.join(self.config.cache_dir, f"sample_{idx}.pt")
                        torch.save(cache_data, cache_file)
                    
                    if pbar is not None:
                        pbar.update(len(batch_items) * num_processes)
        
        if pbar is not None:
            pbar.close()
        
        self.accelerator.wait_for_everyone()
        
        self.cache_ready = True
        
        if self.accelerator.is_main_process:
            print(f"磁盘缓存完成: {len(self.dataset)} 个样本")
            print(f"缓存保存在: {self.config.cache_dir}")
            print(f"训练时按需加载到 LRU 内存缓存 (最大 {self.cache_maxsize} 个)")
        
        self.dataset.set_use_cache(True)
        self._unload_text_encoder()
        self._unload_vae()
    
    def _unload_text_encoder(self):
        if self.accelerator.is_main_process:
            print(">>> 卸载 Text Encoder...")
        
        mem_before = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        del self.text_encoder, self.tokenizer
        self.text_encoder = None
        self.tokenizer = None
        torch.cuda.empty_cache()
        gc.collect()
        mem_after = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        freed_memory = mem_before - mem_after
        if self.accelerator.is_main_process:
            print(f"卸载 Text Encoder 后释放了 {freed_memory:.2f}GB 显存")
    
    def _unload_vae(self):
        if self.accelerator.is_main_process:
            print(">>> 卸载 VAE...")
        
        self.latents_bn_mean = self.latents_bn_mean.cpu()
        self.latents_bn_std = self.latents_bn_std.cpu()
        
        mem_before = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        del self.vae
        self.vae = None
        torch.cuda.empty_cache()
        gc.collect()
        mem_after = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        freed_memory = mem_before - mem_after
        if self.accelerator.is_main_process:
            print(f"卸载 VAE 后释放了 {freed_memory:.2f}GB 显存")
    
    def _apply_lokr(self):
        if self.accelerator.is_main_process:
            print("\n>>> 应用 LoKr...")
        
        self.transformer.requires_grad_(False)
        
        self.lokr_modules, self.lokr_module_names = apply_lokr_to_transformer(
            self.transformer,
            lora_dim=self.config.lora_dim,
            alpha=self.config.lora_alpha,
            factor=self.config.lokr_factor,
            full_matrix=self.config.full_matrix,
            decompose_both=self.config.decompose_both,
            verbose=(self.accelerator.is_main_process),
        )
        
        for module in self.lokr_modules:
            module.to(self.accelerator.device, dtype=self.config.dtype)
        
        trainable_params_count = sum(p.numel() for module in self.lokr_modules for p in module.parameters() if p.requires_grad)
        total_params_count = sum(p.numel() for p in self.transformer.parameters())
        
        if self.accelerator.is_main_process:
            print(f"\n✓ LoKr 应用完成")
            print(f"  - LoKr 模块数量: {len(self.lokr_modules)}")
            print(f"  - 可训练参数: {trainable_params_count:,}")
            print(f"  - 总参数: {total_params_count:,}")
            print(f"  - 可训练比例: {trainable_params_count / total_params_count * 100:.2f}%")
        
        self._prepare_for_distributed_training()
    
    def _prepare_full_training(self):
        if self.accelerator.is_main_process:
            print("\n阶段 5: 准备分布式训练")
            print("=" * 60)
        self._prepare_for_distributed_training()
    
    def _prepare_for_distributed_training(self):
        blocks_to_swap = self.config.blocks_to_swap
        
        if blocks_to_swap > 0:
            self.transformer.move_to_device_except_swap_blocks(self.accelerator.device)
            self.transformer.prepare_block_swap_before_forward()
            if self.accelerator.is_main_process:
                if self.accelerator.num_processes > 1:
                    print("Block Swap + 多卡: 跳过 DDP 包装，使用手动梯度同步")
                print(f"✓ Transformer 已准备 (Block Swap 模式，手动梯度同步)")
        elif self.accelerator.num_processes > 1:
            self.transformer = self.accelerator.prepare(self.transformer)
            if self.accelerator.is_main_process:
                print(f"✓ Transformer 已准备 (DDP: True)")
        else:
            if self.accelerator.is_main_process:
                print(f"✓ Transformer 已准备 (单卡模式)")
    
    def create_dataset(self):
        if self.accelerator.is_main_process:
            print("\n创建 FLUX.2 Klein T2I 数据集")
            print("=" * 60)
        
        self.dataset = Flux2KleinT2IDataset(
            image_folder=self.config.image_folder,
            resolution=self.config.resolution,
            use_caption=self.config.use_caption,
            caption_ext=self.config.caption_ext,
            default_caption=self.config.default_caption,
            verbose=(self.accelerator.is_main_process),
            enable_buckets=self.config.enable_buckets,
            bucket_resolutions=self.config.bucket_resolutions,
        )
        
        if self.config.enable_buckets and self.config.batch_size > 1:
            batch_sampler = ResolutionBucketBatchSampler(
                dataset=self.dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                drop_last=False,
            )
            self.dataloader = DataLoader(
                self.dataset,
                batch_sampler=batch_sampler,
                num_workers=0,
                pin_memory=False,
                collate_fn=self.collate_fn,
            )
        else:
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                collate_fn=self.collate_fn,
            )
        
        if self.accelerator.is_main_process:
            print(f"DataLoader 创建完成, batch_size={self.config.batch_size}")
            if self.config.enable_buckets:
                print(f"使用 Bucket 训练模式")
    
    def collate_fn(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        sample_indices = [ex['sample_idx'] for ex in examples]
        return {'sample_indices': sample_indices}
    
    def train_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        device = self.accelerator.device
        dtype = self.config.dtype
        
        sample_indices = batch['sample_indices']
        batch_size = len(sample_indices)
        
        latents_list = []
        prompt_embeds_list = []
        
        for idx in sample_indices:
            self._load_cache_item(idx)
            latents_list.append(self.latent_cache[idx])
            prompt_embeds_list.append(self.prompt_embeds_cache[idx]['prompt_embeds'].squeeze(0))
        
        target_latents = torch.cat(latents_list, dim=0).to(device, dtype)
        
        max_len = max(pe.shape[0] for pe in prompt_embeds_list)
        embed_dim = prompt_embeds_list[0].shape[-1]
        prompt_embeds = torch.zeros(batch_size, max_len, embed_dim, device=device, dtype=dtype)
        
        for i, pe in enumerate(prompt_embeds_list):
            prompt_embeds[i, :pe.shape[0]] = pe.to(device, dtype)
        
        noise = torch.randn_like(target_latents)
        
        if self.config.noise_offset > 0:
            noise = noise + self.config.noise_offset * torch.randn(
                (noise.shape[0], noise.shape[1], 1, 1),
                device=noise.device,
                dtype=noise.dtype
            )
        
        timesteps, _ = sample_timesteps(
            batch_size,
            num_train_timesteps=1000,
            device=device,
            timestep_type=self.config.timestep_type,
            sigmoid_scale=self.config.sigmoid_scale,
            shift=self.config.shift_scale,
            lognorm_alpha=self.config.lognorm_alpha,
        )
        
        latent_ids = prepare_latent_ids(target_latents).to(device)
        
        sigmas = (timesteps.float() / 1000.0).to(dtype)
        sigmas = sigmas.view(-1, 1, 1, 1)
        noisy_latents = (1.0 - sigmas) * target_latents + sigmas * noise
        
        hidden_states = pack_latents(noisy_latents)
        txt_ids = prepare_text_ids(prompt_embeds).to(device)
        
        timesteps_normalized = (timesteps.float() / 1000.0).to(dtype)
        guidance = None
        
        if not hasattr(self, '_debug_step_count'):
            self._debug_step_count = 0
        self._debug_step_count += 1
        
        with self.accelerator.autocast():
            model_output = self.transformer(
                hidden_states=hidden_states,
                timestep=timesteps_normalized,
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                txt_ids=txt_ids,
                img_ids=latent_ids,
                return_dict=False,
            )[0]
        
        model_output = unpack_latents_with_ids(model_output, latent_ids)
        
        target = noise - target_latents
        
        loss = F.mse_loss(model_output.float(), target.float(), reduction='mean')
        
        
        return loss
    
    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        if self.config.full_training:
            return [p for p in self.transformer.parameters() if p.requires_grad]
        else:
            params = []
            for module in self.lokr_modules:
                params.extend(module.parameters())
            return params
    
    def create_optimizer(self, trainable_params: List[torch.nn.Parameter]):
        self.use_fused_backward = False
        
        self._grad_hook_reduce_count = 0
        self._grad_hook_total_count = 0
        self._first_step_reduce_checked = False
        
        import accelerate
        from accelerate.utils import DistributedType
        
        mode_str = "全量训练" if self.config.full_training else "LoKr"
        
        if self.config.blocks_to_swap > 0:
            # Block Swap 模式：使用 Adafactor + Fused Backward
            import transformers.optimization
            
            if self.accelerator.is_main_process:
                print(f">>> {mode_str} + Block Swap: 使用 Adafactor 优化器")
            
            optimizer = transformers.optimization.Adafactor(
                trainable_params,
                lr=self.config.learning_rate,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
            )
            
            from adafactor_fused import patch_adafactor_fused
            patch_adafactor_fused(optimizer)
            self.use_fused_backward = True
            self._raw_optimizer = optimizer
            
            # 延迟 reduce 模式：不在 grad_hook 中调用 reduce
            # 在 backward 完成后，按固定顺序遍历所有参数，执行 reduce 和 step_param
            # 这确保所有 GPU 以相同顺序处理参数，避免 all_reduce 死锁
            
            if self.accelerator.is_main_process:
                print(f"✓ Adafactor Fused 已启用 (延迟 reduce 模式)")
                print(f"✓ Block Swap + 多卡: 在 backward 后统一处理梯度")
            
            return optimizer
        
        # 非 Block Swap 模式：使用 AdamW8bit
        # AdamW8bit 比 Adafactor 收敛更稳定
        if self.accelerator.is_main_process:
            print(f">>> {mode_str}: 使用 AdamW8bit 优化器")
        
        optimizer = super().create_optimizer(trainable_params)
        
        if self.accelerator.is_main_process:
            print(f"✓ AdamW8bit 已启用 (lr={self.config.learning_rate})")
        
        if accelerate.PartialState().distributed_type != DistributedType.NO:
            for param in trainable_params[:5]:
                if param.requires_grad:
                    def create_check_hook():
                        def grad_check_hook(tensor: torch.Tensor):
                            if not self._first_step_reduce_checked:
                                self._grad_hook_total_count += 1
                                if tensor.grad is not None:
                                    self._grad_hook_reduce_count += 1
                        return grad_check_hook
                    param.register_post_accumulate_grad_hook(create_check_hook())
        
        return optimizer
    
    def _process_pending_gradients(self):
        """延迟 reduce 模式：在 backward 完成后统一处理所有梯度
        
        按固定顺序遍历所有参数，执行 reduce 和 step_param。
        这确保所有 GPU 以相同顺序处理参数，避免 all_reduce 死锁。
        """
        import torch.distributed as dist
        from accelerate.utils import DistributedType
        
        is_distributed = dist.is_initialized() and self.accelerator.distributed_type != DistributedType.NO
        rank = dist.get_rank() if is_distributed else 0
        
        # 统计有梯度的参数数量
        grad_count = 0
        total_params = 0
        step_reduce_count = 0  # 本步的 reduce 计数
        
        # 按固定顺序遍历所有参数（确保所有 GPU 顺序一致）
        for pg_idx, param_group in enumerate(self._raw_optimizer.param_groups):
            for param in param_group["params"]:
                total_params += 1
                if param.grad is not None:
                    grad_count += 1
                    step_reduce_count += 1
                    
                    # 多卡模式：all_reduce 梯度
                    if is_distributed:
                        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
                    
                    # 梯度裁剪
                    if self.config.max_grad_norm != 0.0:
                        torch.nn.utils.clip_grad_norm_([param], max_norm=self.config.max_grad_norm)
                    
                    # 优化器更新
                    self._raw_optimizer.step_param(param, param_group)
                    param.grad = None
        
        self._grad_hook_reduce_count += step_reduce_count
        self._grad_hook_total_count += 1
        
        # 每步都打印统计信息（前10步详细，之后每50步打印一次）
        step = self._grad_hook_total_count
        should_print = step <= 10 or step % 50 == 0
        
        if should_print and rank == 0:
            first_param = None
            for pg in self._raw_optimizer.param_groups:
                for p in pg["params"]:
                    first_param = p
                    break
                if first_param is not None:
                    break
            
            param_sample = first_param.data.flatten()[:5].tolist() if first_param is not None else []
            
            raw_lr = self._raw_optimizer.param_groups[0]["lr"]
            scheduler_lr = self.lr_scheduler.get_last_lr()[0] if hasattr(self.lr_scheduler, 'get_last_lr') else "N/A"
            
            adafactor_state = self._raw_optimizer.state.get(first_param, {})
            adafactor_step = adafactor_state.get("step", 0)
            
            print(f"[Grad Sync] step={step}: 处理 {grad_count}/{total_params} 参数")
            print(f"  raw_optimizer_lr={raw_lr:.2e}, scheduler_lr={scheduler_lr}, adafactor_state_step={adafactor_step}")
            print(f"  首参数样本: {[f'{v:.6f}' for v in param_sample]}")
    
    def save_checkpoint(self, step: int, is_final: bool = False):
        if not self.accelerator.is_main_process:
            return
        
        checkpoint_name = "final" if is_final else f"checkpoint-{step}"
        checkpoint_dir = os.path.join(self.config.output_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        unwrapped_transformer = self.accelerator.unwrap_model(self.transformer)
        
        if self.config.full_training:
            unwrapped_transformer.save_pretrained(
                os.path.join(checkpoint_dir, "transformer"),
                safe_serialization=True,
            )
        else:
            lokr_state_dict = {}
            for name, module in zip(self.lokr_module_names, self.lokr_modules):
                prefix = "lycoris_" + name.replace('.', '_')
                
                if hasattr(module, 'alpha'):
                    lokr_state_dict[f"{prefix}.alpha"] = module.alpha.cpu()
                
                for param_name, param in module.named_parameters():
                    lokr_state_dict[f"{prefix}.{param_name}"] = param.cpu()
            
            save_file(lokr_state_dict, os.path.join(checkpoint_dir, "lokr_weights.safetensors"))
            
            config_dict = {
                "lora_dim": self.config.lora_dim,
                "lora_alpha": self.config.lora_alpha,
                "lokr_factor": self.config.lokr_factor,
                "full_matrix": self.config.full_matrix,
                "decompose_both": self.config.decompose_both,
                "module_names": self.lokr_module_names,
            }
            with open(os.path.join(checkpoint_dir, "lokr_config.json"), 'w') as f:
                json.dump(config_dict, f, indent=2)
        
        print(f"检查点已保存到: {checkpoint_dir}")
    
    def pre_training_hook(self):
        if self.accelerator.is_main_process:
            mode_str = "Full Training" if self.config.full_training else "LoKr Training"
            print(f"\nFLUX.2 Klein T2I {mode_str}")
            print(f"时间步采样: {self.config.timestep_type}")
            if not self.config.full_training:
                print(f"LoKr 配置: dim={self.config.lora_dim}, alpha={self.config.lora_alpha}, factor={self.config.lokr_factor}")
            if self.config.noise_offset > 0:
                print(f"Noise offset: {self.config.noise_offset}")
    
    def save_final_model(self):
        self.accelerator.wait_for_everyone()
        
        if not self.accelerator.is_main_process:
            return
        
        final_dir = os.path.join(self.config.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        
        unwrapped_transformer = self.accelerator.unwrap_model(self.transformer)
        
        if self.config.full_training:
            print(f"\n>>> 保存完整 Transformer 到: {final_dir}")
            unwrapped_transformer.save_pretrained(
                os.path.join(final_dir, "transformer"),
                safe_serialization=True,
            )
        else:
            print(f"\n>>> 保存 LoKr 权重到: {final_dir}")
            
            lokr_state_dict = {}
            for name, module in zip(self.lokr_module_names, self.lokr_modules):
                prefix = "lycoris_" + name.replace('.', '_')
                
                if hasattr(module, 'alpha'):
                    lokr_state_dict[f"{prefix}.alpha"] = module.alpha.cpu()
                
                for param_name, param in module.named_parameters():
                    lokr_state_dict[f"{prefix}.{param_name}"] = param.cpu()
            
            save_file(lokr_state_dict, os.path.join(final_dir, "lokr_weights.safetensors"))
            
            config_dict = {
                "lora_dim": self.config.lora_dim,
                "lora_alpha": self.config.lora_alpha,
                "lokr_factor": self.config.lokr_factor,
                "full_matrix": self.config.full_matrix,
                "decompose_both": self.config.decompose_both,
                "module_names": self.lokr_module_names,
                "base_model": self.config.model_id,
            }
            with open(os.path.join(final_dir, "lokr_config.json"), 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            print(f"[ok] LoKr 权重已保存")
        
        print(f">>> 最终模型保存完成: {final_dir}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="FLUX.2 Klein T2I Trainer")
    
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_train_steps", type=int, default=5000)
    parser.add_argument("--checkpoint_every_n_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--timestep_type", type=str, default="sigmoid")
    parser.add_argument("--sigmoid_scale", type=float, default=1.0)
    parser.add_argument("--shift_scale", type=float, default=3.0)
    parser.add_argument("--lognorm_alpha", type=float, default=0.75)
    parser.add_argument("--noise_offset", type=float, default=0.0)
    parser.add_argument("--full_training", action="store_true")
    parser.add_argument("--blocks_to_swap", type=int, default=0)
    parser.add_argument("--use_pinned_memory", action="store_true")
    parser.add_argument("--use_caption", action="store_true")
    parser.add_argument("--caption_ext", type=str, default=".txt")
    parser.add_argument("--default_caption", type=str, default="")
    parser.add_argument("--disable_buckets", action="store_true")
    parser.add_argument("--bucket_resolutions", type=int, nargs='+', default=None)
    
    args = parser.parse_args()
    return args


def main():
    fix_windows_encoding()
    
    args = parse_args()
    
    config = Flux2KleinT2IConfig(
        model_id=args.model_id,
        image_folder=args.image_folder,
        output_dir=args.output_dir,
        num_train_steps=args.num_train_steps,
        checkpoint_every_n_steps=args.checkpoint_every_n_steps,
        checkpoints_total_limit=args.checkpoints_total_limit,
        learning_rate=args.learning_rate,
        resolution=args.resolution,
        timestep_type=args.timestep_type,
        sigmoid_scale=args.sigmoid_scale,
        shift_scale=args.shift_scale,
        lognorm_alpha=args.lognorm_alpha,
        noise_offset=args.noise_offset,
        full_training=args.full_training,
        blocks_to_swap=args.blocks_to_swap,
        use_pinned_memory=args.use_pinned_memory,
        use_caption=args.use_caption,
        caption_ext=args.caption_ext,
        default_caption=args.default_caption,
        enable_buckets=(not args.disable_buckets),
        bucket_resolutions=args.bucket_resolutions,
    )
    
    trainer = Flux2KleinT2ITrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
