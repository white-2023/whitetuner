"""
Qwen Image T2I LoKr 训练器

文生图（Text-to-Image）训练
- 支持 Qwen-Image 和 Qwen-Image-2512
- LoKr 高效微调
- 支持多种时间步采样方法
"""

import os
import gc
import math
import json
from typing import List, Optional, Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from PIL.ImageOps import exif_transpose
from tqdm import tqdm

from diffusers import AutoencoderKLQwenImage
from diffusers import FlowMatchEulerDiscreteScheduler
from qwen_modules import QwenImageTransformer2DModel, load_qwen_transformer_from_diffusers
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer
from optimum.quanto import quantize, freeze, qint8, qfloat8, QTensor, QBytesTensor
from safetensors.torch import save_file, load_file

from base_trainer import (
    BaseTrainer,
    BaseTrainerConfig,
    pack_latents,
    unpack_latents,
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


class QwenImageConfig(BaseTrainerConfig):
    
    def __init__(
        self,
        image_folder: str,
        output_dir: str,
        model_id: str = "Qwen/Qwen-Image",
        num_train_steps: int = 5000,
        checkpoint_every_n_steps: int = 500,
        checkpoints_total_limit: int = 3,
        batch_size: int = 1,
        learning_rate: float = 1e-4,
        lr_warmup_steps: int = 100,
        resolution: int = 1024,
        timestep_type: TimestepType = "sigmoid",
        sigmoid_scale: float = 1.0,
        shift_scale: float = 1.0,
        lognorm_alpha: float = 0.75,
        full_training: bool = False,
        full_matrix: bool = True,
        lora_dim: int = 10000,
        lora_alpha: int = 1,
        lokr_factor: int = 4,
        decompose_both: bool = False,
        quantize_transformer: bool = True,
        quantize_text_encoder: bool = True,
        use_fp8: bool = False,
        blocks_to_swap: int = 0,
        use_pinned_memory: bool = True,
        optimizer_type: str = "adafactor",
        use_caption: bool = True,
        caption_ext: str = ".txt",
        default_caption: str = "",
        noise_offset: float = 0.0,
        cache_dir: str = None,
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
        
        self.use_caption = use_caption
        self.caption_ext = caption_ext
        self.default_caption = default_caption
        self.use_fp8 = use_fp8
        self.blocks_to_swap = blocks_to_swap
        self.use_pinned_memory = use_pinned_memory
        self.optimizer_type = optimizer_type
        self.noise_offset = noise_offset
        
        if cache_dir is None:
            self.cache_dir = os.path.join(image_folder, ".qwen_image_cache")
        else:
            self.cache_dir = cache_dir


def verify_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False


class QwenImageDataset(Dataset):
    
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
            print(f"✓ 找到 {len(self.samples)} 张有效图片")
            if use_caption:
                with_caption = sum(1 for s in self.samples if s['caption'] != default_caption)
                print(f"  - 有 caption: {with_caption}")
                print(f"  - 使用默认 caption: {len(self.samples) - with_caption}")
            if skipped_corrupted > 0:
                print(f"  - 跳过 {skipped_corrupted} 张损坏图片:")
                for f in corrupted_files[:5]:
                    print(f"    {f}")
                if len(corrupted_files) > 5:
                    print(f"    ... 以及 {len(corrupted_files) - 5} 张更多")
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid images found in: {image_folder}")
        
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


class QwenImageTrainer(BaseTrainer):
    
    def __init__(self, config: QwenImageConfig):
        super().__init__(config)
        self.config: QwenImageConfig = config
        
        self.lokr_modules = None
        self.lokr_module_names = None
        
        self.text_embeds_cache = {}
        self.latent_cache = {}
    
    def _check_stop(self) -> bool:
        return self.should_stop
    
    def create_dataset(self):
        if self.accelerator.is_main_process:
            print("\n创建 Qwen Image 数据集")
            print("=" * 60)
        
        self.dataset = QwenImageDataset(
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
            print(f"✓ DataLoader 创建完成，batch_size={self.config.batch_size}")
    
    def load_models(self):
        if self._check_stop():
            return
        
        self._load_text_encoder()
        if self._check_stop():
            return
        
        self._load_vae()
        if self._check_stop():
            return
        
        self._cache_embeddings_and_latents()
        if self._check_stop():
            return
        
        self._load_transformer()
        if self._check_stop():
            return
        
        self._apply_lokr()
        if self._check_stop():
            return
        
        self._prepare_for_ddp()
    
    def _load_text_encoder(self):
        if self.accelerator.is_main_process:
            print("\n阶段 1: 加载 Text Encoder")
            print("=" * 60)
        
        if self._check_stop():
            return
        
        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.model_id,
            subfolder="text_encoder",
            torch_dtype=self.config.dtype,
            low_cpu_mem_usage=True,
        )
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        
        self.text_encoder.model.visual = None
        if self.accelerator.is_main_process:
            print("✓ 已移除 visual 模块 (T2I 不需要)")
        
        if self.config.quantize_text_encoder:
            exclude_patterns = ["*embed*", "*lm_head*"]
            quantize(self.text_encoder, weights=self.config.quantize_level, exclude=exclude_patterns)
            freeze(self.text_encoder)
        
        self.text_encoder.to(self.accelerator.device)
        
        self.tokenizer = Qwen2Tokenizer.from_pretrained(
            self.config.model_id,
            subfolder="tokenizer",
        )
        
        if self.accelerator.is_main_process:
            print("✓ Text Encoder 和 Tokenizer 加载完成")
    
    def _load_vae(self):
        if self.accelerator.is_main_process:
            print("\n阶段 2: 加载 VAE")
            print("=" * 60)
        
        if self._check_stop():
            return
        
        self.vae = AutoencoderKLQwenImage.from_pretrained(
            self.config.model_id,
            subfolder="vae",
            torch_dtype=self.config.dtype,
        )
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae.to(self.accelerator.device)
        
        if self.accelerator.is_main_process:
            print("✓ VAE 加载完成")
    
    def _encode_prompt(self, prompt: str) -> Dict[str, torch.Tensor]:
        if not prompt or not prompt.strip():
            raise ValueError(
                "Qwen 模型需要有效的提示词(caption)才能训练。\n"
                "请为每张图片提供对应的 .txt 文件或在 metadata 中指定 caption。"
            )
        
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = inputs.input_ids.to(self.accelerator.device)
        attention_mask = inputs.attention_mask.to(self.accelerator.device)
        
        with torch.no_grad():
            outputs = self.text_encoder.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            prompt_embeds = outputs.hidden_states[-1]
        
        valid_length = attention_mask.sum().item()
        if valid_length == 0:
            raise ValueError(
                "Text encoder 输出的有效长度为 0，请检查提示词是否有效。\n"
                "当前提示词: " + repr(prompt[:100] if len(prompt) > 100 else prompt)
            )
        
        prompt_embeds = prompt_embeds[:, :valid_length, :]
        attention_mask = attention_mask[:, :valid_length]
        
        return {
            'prompt_embeds': prompt_embeds.cpu(),
            'attention_mask': attention_mask.cpu(),
        }
    
    def _cache_embeddings_and_latents(self):
        if self.accelerator.is_main_process:
            print("\n阶段 3: 缓存 Text Embeddings 和 Latents")
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
                self.text_embeds_cache[idx] = {
                    'prompt_embeds': cached['prompt_embeds'],
                    'attention_mask': cached['attention_mask'],
                }
                self.latent_cache[idx] = cached['latents']
            else:
                samples_to_encode.append(idx)
        
        num_processes = self.accelerator.num_processes
        process_index = self.accelerator.process_index
        
        if self.accelerator.is_main_process:
            if len(self.text_embeds_cache) == 0:
                print(f"没有缓存，现在开始创建缓存 ({len(samples_to_encode)} 个样本)")
            else:
                print(f"已加载 {len(self.text_embeds_cache)} 个缓存，需要编码 {len(samples_to_encode)} 个样本")
        
        if len(samples_to_encode) > 0:
            my_samples = []
            for i, idx in enumerate(samples_to_encode):
                if i % num_processes == process_index:
                    my_samples.append(idx)
            
            pbar = None
            if self.accelerator.is_main_process:
                pbar = tqdm(total=len(samples_to_encode), desc="缓存 embeddings 和 latents")
            
            with torch.no_grad():
                for idx in my_samples:
                    if self._check_stop():
                        break
                    
                    sample = self.dataset[idx]
                    
                    caption = sample['caption']
                    text_data = self._encode_prompt(caption)
                    
                    image = sample['image'].unsqueeze(0).to(self.accelerator.device, self.config.dtype)
                    image_rescaled = image * 2 - 1
                    image_rescaled = image_rescaled.unsqueeze(2)
                    
                    encoded = self.vae.encode(image_rescaled).latent_dist.sample()
                    if encoded.dim() == 5:
                        encoded = encoded.squeeze(2)
                    
                    latents_mean = (
                        torch.tensor(self.vae.config.latents_mean)
                        .view(1, self.vae.config.z_dim, 1, 1)
                        .to(device=encoded.device, dtype=encoded.dtype)
                    )
                    latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                        1, self.vae.config.z_dim, 1, 1
                    ).to(device=encoded.device, dtype=encoded.dtype)
                    
                    latents = (encoded - latents_mean) * latents_std
                    
                    cache_data = {
                        'prompt_embeds': text_data['prompt_embeds'],
                        'attention_mask': text_data['attention_mask'],
                        'latents': latents.cpu(),
                    }
                    
                    cache_file = os.path.join(self.config.cache_dir, f"sample_{idx}.pt")
                    torch.save(cache_data, cache_file)
                    
                    self.text_embeds_cache[idx] = {
                        'prompt_embeds': text_data['prompt_embeds'],
                        'attention_mask': text_data['attention_mask'],
                    }
                    self.latent_cache[idx] = latents.cpu()
                    
                    if pbar is not None:
                        pbar.update(num_processes)
            
            if pbar is not None:
                pbar.close()
        
        self.accelerator.wait_for_everyone()
        
        if self._check_stop():
            return
        
        for idx in samples_to_encode:
            if idx not in self.text_embeds_cache:
                cache_file = os.path.join(self.config.cache_dir, f"sample_{idx}.pt")
                if os.path.exists(cache_file):
                    cached = torch.load(cache_file, map_location='cpu')
                    self.text_embeds_cache[idx] = {
                        'prompt_embeds': cached['prompt_embeds'],
                        'attention_mask': cached['attention_mask'],
                    }
                    self.latent_cache[idx] = cached['latents']
        
        if self.accelerator.is_main_process:
            print(f"✓ 缓存完成，共 {len(self.text_embeds_cache)} 个样本")
        
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
            print(f"✓ 卸载编码器后释放了 {freed_memory:.2f}GB 显存")
    
    def _load_transformer(self):
        if self.accelerator.is_main_process:
            print("\n阶段 4: 加载 Transformer")
            print("=" * 60)
        
        if self._check_stop():
            return
        
        # Full training 模式检查
        if self.config.full_training:
            if self.config.use_fp8:
                raise ValueError("Full training 不支持 FP8 量化，因为 FP8 权重不支持梯度计算")
            if self.config.blocks_to_swap == 0:
                if self.accelerator.is_main_process:
                    print("警告: Full training 需要大量显存，建议启用 block_swap")
        
        # Block Swap 需要 FP8（qint8 与 block swap 不兼容）
        use_fp8 = self.config.use_fp8
        if self.config.blocks_to_swap > 0 and not self.config.full_training and not use_fp8:
            if self.accelerator.is_main_process:
                print(">>> Block Swap 模式自动启用 FP8（qint8 与 block swap 不兼容）")
            use_fp8 = True
        
        loading_device = self.accelerator.device if self.config.blocks_to_swap == 0 else None
        
        if use_fp8 or self.config.blocks_to_swap > 0:
            if self.accelerator.is_main_process:
                mode_str = "FP8" if use_fp8 else "Block Swap"
                print(f">>> 加载 Transformer（支持 Block Swap，{mode_str} 模式）...")
            self.transformer = load_qwen_transformer_from_diffusers(
                self.config.model_id,
                subfolder="transformer",
                dtype=self.config.dtype,
                device=loading_device,
                use_fp8=use_fp8,
            )
        else:
            if self.accelerator.is_main_process:
                print(">>> 加载 Transformer（CPU，准备量化）...")
            self.transformer = load_qwen_transformer_from_diffusers(
                self.config.model_id,
                subfolder="transformer",
                dtype=self.config.dtype,
                device=None,  # 加载到 CPU
                use_fp8=False,
            )
        
        self._use_fp8 = use_fp8
        
        if self._check_stop():
            return
        
        if self.config.full_training:
            self.transformer.requires_grad_(True)
            if self.accelerator.is_main_process:
                print(">>> Full Training 模式：transformer 参数可训练")
                trainable_count = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
                total_count = sum(p.numel() for p in self.transformer.parameters())
                print(f"    可训练参数: {trainable_count:,} / {total_count:,}")
        elif use_fp8:
            self.transformer.requires_grad_(False)
            if self.accelerator.is_main_process:
                print(">>> FP8 模式：transformer 已冻结")
        elif self.config.quantize_transformer:
            # 使用 ai-toolkit 的方式：逐个 block 量化以节省显存
            if self.accelerator.is_main_process:
                print("\n>>> 使用 quanto qfloat8 量化（逐块量化以节省显存）...")
            exclude_patterns = ["*norm*", "proj_out*", "*embedder*"]
            
            # 逐个 block 量化：移到 GPU -> 量化 -> 移回 CPU
            all_blocks = list(self.transformer.transformer_blocks)
            if self.accelerator.is_main_process:
                print(f"    正在量化 {len(all_blocks)} 个 transformer blocks...")
            from tqdm import tqdm
            for block in tqdm(all_blocks, disable=not self.accelerator.is_main_process):
                block.to(self.accelerator.device, dtype=self.config.dtype, non_blocking=True)
                quantize(block, weights=self.config.quantize_level, exclude=exclude_patterns)
                freeze(block)
                block.to("cpu", non_blocking=True)
            
            # 量化其他部分
            if self.accelerator.is_main_process:
                print("    正在量化其他模块...")
            quantize(self.transformer, weights=self.config.quantize_level, exclude=exclude_patterns)
            freeze(self.transformer)
            
            self.transformer.requires_grad_(False)
            if self.accelerator.is_main_process:
                print(">>> Transformer 所有参数已冻结 (requires_grad=False)")
            
            self.transformer.to(self.accelerator.device)
            if self.accelerator.is_main_process:
                print(">>> quanto qfloat8 量化完成，模型已移至 GPU")
        
        if self.config.blocks_to_swap > 0:
            if self.accelerator.is_main_process:
                print(f">>> 启用 block swap: {self.config.blocks_to_swap} blocks")
            self.transformer.enable_block_swap(
                self.config.blocks_to_swap,
                self.accelerator.device,
                supports_backward=True,
                use_pinned_memory=self.config.use_pinned_memory,
            )
            if self.accelerator.is_main_process:
                blocks_on_gpu = self.transformer.num_blocks - self.config.blocks_to_swap
                print(f">>> Block swap 已启用: {blocks_on_gpu} blocks 在 GPU, {self.config.blocks_to_swap} blocks 在 CPU")
        
        if hasattr(self.transformer, 'enable_gradient_checkpointing'):
            use_cpu_offload = self.config.blocks_to_swap > 0
            self.transformer.enable_gradient_checkpointing(activation_cpu_offloading=use_cpu_offload)
            if self.accelerator.is_main_process:
                if use_cpu_offload:
                    print("✓ 启用 Gradient Checkpointing (with Activation CPU Offloading)")
                else:
                    print("✓ 启用 Gradient Checkpointing")
        
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=1.0,
            use_dynamic_shifting=True,
        )
        
        if self.accelerator.is_main_process:
            print("✓ Transformer 和 Scheduler 加载完成")
    
    def _apply_lokr(self):
        if self._check_stop():
            return
        
        if self.config.full_training:
            if self.accelerator.is_main_process:
                print("\n>>> Full Training 模式....")
            self.lokr_modules = []
            self.lokr_module_names = []
            return
        
        if self.accelerator.is_main_process:
            print("\n>>> 应用 LoKr...")
        
        self.lokr_modules, self.lokr_module_names = apply_lokr_to_transformer(
            self.transformer,
            lora_dim=self.config.lora_dim,
            alpha=self.config.lora_alpha,
            factor=self.config.lokr_factor,
            full_matrix=self.config.full_matrix,
            decompose_both=self.config.decompose_both,
            verbose=self.accelerator.is_main_process
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
    
    def _prepare_for_ddp(self):
        if self.accelerator.is_main_process:
            print("\n阶段 5: 准备分布式训练")
            print("=" * 60)
        
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
                print(f"✓ Transformer 已准备 (单卡模式，跳过 DDP 包装)")
    
    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        if self.config.full_training:
            return list(self.transformer.parameters())
        return [p for module in self.lokr_modules for p in module.parameters() if p.requires_grad]
    
    def create_optimizer(self, trainable_params: List[torch.nn.Parameter]):
        import accelerate
        from accelerate.utils import DistributedType
        import transformers.optimization
        
        blocks_to_swap = self.config.blocks_to_swap
        mode_str = "Full Training" if self.config.full_training else "LoKr"
        
        if blocks_to_swap > 0:
            from adafactor_fused import patch_adafactor_fused
            
            if self.accelerator.is_main_process:
                print(f"{mode_str} + Block Swap: 使用 Adafactor 优化器 (fused backward mode)")
            
            optimizer = transformers.optimization.Adafactor(
                trainable_params,
                lr=self.config.learning_rate,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
            )
            
            patch_adafactor_fused(optimizer)
            
            self.use_fused_backward = True
            self._raw_optimizer = optimizer
            
            self._grad_hook_reduce_count = 0
            self._grad_hook_total_count = 0
            
            for param_group in optimizer.param_groups:
                for parameter in param_group["params"]:
                    if parameter.requires_grad:
                        def create_grad_hook(p_group_idx):
                            def grad_hook(tensor: torch.Tensor):
                                self._grad_hook_total_count += 1
                                state = accelerate.PartialState()
                                if state.distributed_type != DistributedType.NO:
                                    self._grad_hook_reduce_count += 1
                                    if self._grad_hook_reduce_count <= 3:
                                        grad_before = tensor.grad.norm().item() if tensor.grad is not None else 0
                                    tensor.grad = self.accelerator.reduce(tensor.grad, reduction="mean")
                                    if self._grad_hook_reduce_count <= 3:
                                        grad_after = tensor.grad.norm().item() if tensor.grad is not None else 0
                                        print(f"[Reduce Check #{self._grad_hook_reduce_count}] grad_norm: {grad_before:.6f} -> {grad_after:.6f}")
                                if self.accelerator.sync_gradients and self.config.max_grad_norm != 0.0:
                                    self.accelerator.clip_grad_norm_(tensor, max_norm=self.config.max_grad_norm)
                                current_group = self._raw_optimizer.param_groups[p_group_idx]
                                self._raw_optimizer.step_param(tensor, current_group)
                                tensor.grad = None
                            return grad_hook
                        parameter.register_post_accumulate_grad_hook(create_grad_hook(0))
            
            if self.accelerator.is_main_process:
                print(f"Fused backward pass: 已注册 grad hooks")
                print(f"  - Distributed type: {accelerate.PartialState().distributed_type}")
            
            return optimizer
        
        if self.accelerator.is_main_process:
            print(f"{mode_str}: 使用 Adafactor 优化器")
        self.use_fused_backward = False
        return transformers.optimization.Adafactor(
            trainable_params,
            lr=self.config.learning_rate,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    
    def train_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        device = self.accelerator.device
        sample_indices = batch['sample_indices']
        batch_size = len(sample_indices)
        
        latents_list = []
        prompt_embeds_list = []
        attention_mask_list = []
        
        for idx in sample_indices:
            latents_list.append(self.latent_cache[idx])
            prompt_embeds_list.append(self.text_embeds_cache[idx]['prompt_embeds'].squeeze(0))
            attention_mask_list.append(self.text_embeds_cache[idx]['attention_mask'].squeeze(0))
        
        latents = torch.cat(latents_list, dim=0).to(device, self.config.dtype)
        
        max_len = max(pe.shape[0] for pe in prompt_embeds_list)
        embed_dim = prompt_embeds_list[0].shape[-1]
        prompt_embeds = torch.zeros(batch_size, max_len, embed_dim, device=device, dtype=self.config.dtype)
        attention_mask = torch.zeros(batch_size, max_len, device=device, dtype=torch.int64)
        
        for i, (pe, am) in enumerate(zip(prompt_embeds_list, attention_mask_list)):
            prompt_embeds[i, :pe.shape[0]] = pe.to(device, self.config.dtype)
            attention_mask[i, :am.shape[0]] = am.to(device)
        
        txt_seq_lens = attention_mask.sum(dim=1).tolist()
        
        num_channels = latents.shape[1]
        latent_height = latents.shape[2]
        latent_width = latents.shape[3]
        
        noise = torch.randn_like(latents)
        
        if self.config.noise_offset > 0:
            noise = noise + self.config.noise_offset * torch.randn(
                (noise.shape[0], noise.shape[1], 1, 1),
                device=noise.device,
                dtype=noise.dtype
            )
        
        timesteps, timestep_weights = sample_timesteps(
            batch_size,
            num_train_timesteps=self.noise_scheduler.config.num_train_timesteps,
            device=device,
            timestep_type=self.config.timestep_type,
            sigmoid_scale=self.config.sigmoid_scale,
            shift=self.config.shift_scale,
            lognorm_alpha=self.config.lognorm_alpha,
        )
        
        sigmas = timesteps.float() / self.noise_scheduler.config.num_train_timesteps
        sigmas = sigmas.view(-1, 1, 1, 1)
        noisy_latents = (1 - sigmas) * latents + sigmas * noise
        
        packed_latents = pack_latents(
            noisy_latents,
            batch_size,
            num_channels,
            latent_height,
            latent_width,
        )
        
        img_shapes = [[(1, latent_height // 2, latent_width // 2)] for _ in range(batch_size)]
        
        timesteps_normalized = timesteps.float() / 1000.0
        
        with self.accelerator.autocast():
            model_output = self.transformer(
                hidden_states=packed_latents.to(device, self.config.dtype),
                timestep=timesteps_normalized,
                guidance=None,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_mask=attention_mask,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
            )
        
        pixel_height = latent_height * self.config.vae_scale_factor
        pixel_width = latent_width * self.config.vae_scale_factor
        
        model_output = unpack_latents(
            model_output,
            pixel_height,
            pixel_width,
            self.config.vae_scale_factor
        )
        if model_output.dim() == 5:
            model_output = model_output.squeeze(2)
        
        target = noise - latents
        
        per_sample_loss = F.mse_loss(model_output.float(), target.float(), reduction="none")
        per_sample_loss = per_sample_loss.mean(dim=[1, 2, 3])
        
        weighted_loss = per_sample_loss * timestep_weights
        loss = weighted_loss.mean()
        
        return loss
    
    def save_checkpoint(self, step: int):
        output_dir = self.config.output_dir or os.path.join(self.script_dir, "output")
        checkpoint_dir = os.path.join(output_dir, "checkpoints", f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        unwrapped_transformer = self.accelerator.unwrap_model(self.transformer)
        
        if self.config.full_training:
            if self.accelerator.is_main_process:
                transformer_save_dir = os.path.join(checkpoint_dir, "transformer")
                os.makedirs(transformer_save_dir, exist_ok=True)
                unwrapped_transformer.save_pretrained(transformer_save_dir)
                print(f"  - 全量模型已保存到 {transformer_save_dir},")
        else:
            lokr_state_dict = {}
            for idx, (module, layer_name) in enumerate(zip(self.lokr_modules, self.lokr_module_names)):
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
            "model_id": self.config.model_id,
            "image_folder": self.config.image_folder,
            "output_dir": self.config.output_dir,
            "num_train_steps": self.config.num_train_steps,
            "learning_rate": self.config.learning_rate,
            "resolution": self.config.resolution,
            "timestep_type": self.config.timestep_type,
            "training_method": "full" if self.config.full_training else "lokr",
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
            print(f"从检查点恢复: {checkpoint_dir}")
        
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
            print(f"  - 恢复步数: {self.resume_step}")
        
        lokr_weights_path = os.path.join(checkpoint_dir, "lokr_weights.safetensors")
        if os.path.exists(lokr_weights_path) and self.lokr_modules is not None:
            if self.accelerator.is_main_process:
                print(f"  - 加载 LoKr 权重...")
            
            lokr_state_dict = load_file(lokr_weights_path)
            
            for module, layer_name in zip(self.lokr_modules, self.lokr_module_names):
                key_prefix = f"diffusion_model.{layer_name}"
                
                for param_name, param in module.named_parameters():
                    key = f"{key_prefix}.{param_name}"
                    if key in lokr_state_dict:
                        param.data.copy_(lokr_state_dict[key].to(param.device, param.dtype))
            
            if self.accelerator.is_main_process:
                print(f"  - LoKr 权重已恢复")
        
        accelerate_state_dir = os.path.join(checkpoint_dir, "accelerate_state")
        if os.path.exists(accelerate_state_dir):
            self.accelerator.load_state(accelerate_state_dir)
            if self.accelerator.is_main_process:
                print(f"  - Accelerate 状态已恢复")
    
    def save_final_model(self):
        self.accelerator.wait_for_everyone()
        
        if not self.accelerator.is_main_process:
            return
        
        print("\n保存最终模型")
        print("=" * 60)
        
        output_dir = self.config.output_dir or os.path.join(self.script_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        model_name = "Qwen-Image"
        if "2512" in self.config.model_id:
            model_name = "Qwen-Image-2512"
        
        actual_steps = self.current_step if self.current_step > 0 else self.adjusted_num_train_steps
        equivalent_single_gpu_steps = actual_steps * self.accelerator.num_processes * self.config.gradient_accumulation_steps
        
        unwrapped_transformer = self.accelerator.unwrap_model(self.transformer)
        
        if self.config.full_training:
            transformer_save_dir = os.path.join(output_dir, f"qwen_image_full_{equivalent_single_gpu_steps}steps")
            os.makedirs(transformer_save_dir, exist_ok=True)
            unwrapped_transformer.save_pretrained(transformer_save_dir)
            
            training_config = {
                "training_type": "full",
                "base_model": model_name,
                "resolution": self.config.resolution,
                "trained_steps": actual_steps,
                "equivalent_single_gpu_steps": equivalent_single_gpu_steps,
                "num_gpus": self.accelerator.num_processes,
            }
            with open(os.path.join(transformer_save_dir, "training_config.json"), "w") as f:
                json.dump(training_config, f, indent=2)
            
            print(f"\n✓ 全量模型已保存到: {transformer_save_dir}/")
        else:
            lokr_state_dict = {}
            for idx, (module, layer_name) in enumerate(zip(self.lokr_modules, self.lokr_module_names)):
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
                "base_model": model_name,
                "ss_network_module": "lycoris.kohya",
                "ss_network_dim": str(self.config.lora_dim),
                "ss_network_alpha": str(self.config.lora_alpha),
                "num_gpus": str(self.accelerator.num_processes),
            }
            
            model_filename = f"qwen_image_lokr_{equivalent_single_gpu_steps}steps.safetensors"
            config_filename = f"lokr_config_{equivalent_single_gpu_steps}steps.json"
            
            save_file(lokr_state_dict, os.path.join(output_dir, model_filename), metadata=metadata)
            
            lokr_config = {
                "lora_dim": self.config.lora_dim,
                "lora_alpha": self.config.lora_alpha,
                "lokr_factor": self.config.lokr_factor,
                "full_matrix": self.config.full_matrix,
                "decompose_both": self.config.decompose_both,
                "num_modules": len(self.lokr_modules),
                "base_model": model_name,
                "resolution": self.config.resolution,
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
            
            print(f"\n✓ 最终 LoKr 模型已保存到: {output_dir}/")
            print(f"  - {model_filename}")
            print(f"  - {config_filename}")
        
        self.save_checkpoint(actual_steps)
        
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        print(f"✓ 最终检查点已保存到: {checkpoint_dir}/checkpoint-{actual_steps}/")
        
        if self.should_stop:
            print(f"\n训练在第 {actual_steps} 步停止")
        else:
            print(f"\n训练完成!")
        
        print(f"  数据集: {len(self.dataset)} 张图片")
        print(f"  使用 GPU: {self.accelerator.num_processes}")
        if self.config.full_training:
            print(f"\n模型格式: Full (完整模型)")
        else:
            print(f"\n模型格式: LyCORIS LoKr")
        print(f"可在 ComfyUI 中直接加载")
    
    def pre_training_hook(self):
        if self.accelerator.is_main_process:
            if self.config.full_training:
                print(f"\nQwen Image T2I 全量训练")
            else:
                print(f"\nQwen Image T2I LoKr 训练")
            print(f"时间步采样: {self.config.timestep_type}")
            if self.config.timestep_type == "sigmoid":
                print(f"  - sigmoid_scale: {self.config.sigmoid_scale}")
            elif self.config.timestep_type == "shift":
                print(f"  - shift_scale: {self.config.shift_scale}")
            elif self.config.timestep_type == "lognorm_blend":
                print(f"  - lognorm_alpha: {self.config.lognorm_alpha}")
            if not self.config.full_training:
                print(f"LoKr 配置: dim={self.config.lora_dim}, alpha={self.config.lora_alpha}, factor={self.config.lokr_factor}")
                print(f"  - full_matrix: {self.config.full_matrix}")
            print(f"Noise offset: {self.config.noise_offset}")
            print(f"优化器: Adafactor")
            if getattr(self, '_use_fp8', self.config.use_fp8):
                print(f"FP8: 已启用")
            if self.config.blocks_to_swap > 0:
                print(f"Block Swap: {self.config.blocks_to_swap} 个 blocks")


def main():
    import argparse
    
    fix_windows_encoding()
    
    parser = argparse.ArgumentParser(description="Qwen Image T2I LoKr Trainer")
    
    parser.add_argument("--model_id", type=str, required=True, help="Model path (Qwen/Qwen-Image or Qwen/Qwen-Image-2512)")
    parser.add_argument("--image_folder", type=str, required=True, help="Image folder path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    
    parser.add_argument("--num_train_steps", type=int, default=5000, help="Training steps")
    parser.add_argument("--checkpoint_every_n_steps", type=int, default=500, help="Checkpoint interval")
    parser.add_argument("--checkpoints_total_limit", type=int, default=3, help="Max checkpoints")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr_warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--resolution", type=int, default=1024, help="Image resolution")
    
    parser.add_argument("--timestep_type", type=str, default="sigmoid", choices=TIMESTEP_TYPES, help="Timestep sampling type")
    parser.add_argument("--sigmoid_scale", type=float, default=1.0, help="Sigmoid scale")
    parser.add_argument("--shift_scale", type=float, default=1.0, help="Shift scale")
    parser.add_argument("--lognorm_alpha", type=float, default=0.75, help="Lognorm alpha")
    
    parser.add_argument("--use_caption", action="store_true", default=True, help="Use captions")
    parser.add_argument("--caption_ext", type=str, default=".txt", help="Caption file extension")
    parser.add_argument("--default_caption", type=str, default="", help="Default caption")
    parser.add_argument("--noise_offset", type=float, default=0.0, help="Noise offset")
    
    parser.add_argument("--training_mode", type=str, default="lokr", choices=["lokr", "full"], help="Training mode: lokr or full")
    parser.add_argument("--use_fp8", action="store_true", help="Use FP8 quantization instead of qint8 (LoKr mode only)")
    parser.add_argument("--blocks_to_swap", type=int, default=0, help="Number of transformer blocks to swap to CPU (0 to disable)")
    parser.add_argument("--optimizer_type", type=str, default="adafactor", help="Optimizer type (only adafactor supported)")
    
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume checkpoint")
    
    args = parser.parse_args()
    
    full_training = args.training_mode == "full"
    
    config = QwenImageConfig(
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
        full_training=full_training,
        use_caption=args.use_caption,
        caption_ext=args.caption_ext,
        default_caption=args.default_caption,
        noise_offset=args.noise_offset,
        use_fp8=args.use_fp8,
        blocks_to_swap=args.blocks_to_swap,
        optimizer_type=args.optimizer_type,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        seed=args.seed,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    
    trainer = QwenImageTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()


