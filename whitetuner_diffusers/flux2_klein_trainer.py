"""
FLUX.2 Klein Edit LoKr 训练器

基于 Qwen Edit Trainer 的结构实现 FLUX.2 Klein 多图 edit 训练

包含：
- Flux2KleinConfig 配置类
- Flux2KleinDataset 数据集类
- Flux2KleinTrainer 训练器类
"""

import os
import gc
import math
import json
from typing import List, Optional, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from PIL.ImageOps import exif_transpose
from tqdm import tqdm

from diffusers import (
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler,
)
from flux2_modules import Flux2Transformer2DModel, load_flux2_transformer_from_diffusers
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM
from optimum.quanto import quantize, freeze, qint8, qfloat8, QTensor, QBytesTensor
from safetensors.torch import save_file

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


def get_base_id(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    if '_' in stem:
        parts = stem.rsplit('_', 1)
        if parts[1].isdigit():
            return parts[0]
    return stem


# ============================================================
# FLUX.2 Klein 辅助函数
# ============================================================

def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """将 latents 转换为 2x2 patch 格式
    
    Args:
        latents: [B, C, H, W]
    Returns:
        [B, C*4, H//2, W//2]
    """
    batch_size, num_channels, height, width = latents.shape
    latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(batch_size, num_channels * 4, height // 2, width // 2)
    return latents


def unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """将 patch 格式转回标准格式
    
    Args:
        latents: [B, C*4, H, W]
    Returns:
        [B, C, H*2, W*2]
    """
    batch_size, num_channels, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels // 4, 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    latents = latents.reshape(batch_size, num_channels // 4, height * 2, width * 2)
    return latents


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack latents: [B, C, H, W] -> [B, H*W, C]"""
    batch_size, num_channels, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
    return latents


def unpack_latents_with_ids(x: torch.Tensor, x_ids: torch.Tensor) -> torch.Tensor:
    """使用位置 ID 将 tokens 还原为空间格式"""
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
    """生成 latent 的 4D 位置坐标 (T, H, W, L)
    
    Args:
        latents: [B, C, H, W]
    Returns:
        [B, H*W, 4]
    """
    batch_size, _, height, width = latents.shape

    t = torch.arange(1)  # [0] - time dimension
    h = torch.arange(height)
    w = torch.arange(width)
    l = torch.arange(1)  # [0] - layer dimension

    # Create position IDs: (H*W, 4)
    latent_ids = torch.cartesian_prod(t, h, w, l)

    # Expand to batch: (B, H*W, 4)
    latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

    return latent_ids


def prepare_text_ids(x: torch.Tensor) -> torch.Tensor:
    """生成文本的 4D 位置坐标
    
    Args:
        x: [B, L, D]
    Returns:
        [B, L, 4]
    """
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


def prepare_image_ids(
    image_latents: List[torch.Tensor],
    scale: int = 10,
) -> torch.Tensor:
    """为多个 condition 图像生成 4D 位置坐标
    
    Args:
        image_latents: List of [1, C, H, W] latents
        scale: 用于分隔不同图像的 time 坐标偏移
    Returns:
        [1, N_total, 4]
    """
    t_coords = [scale + scale * t for t in torch.arange(0, len(image_latents))]
    t_coords = [t.view(-1) for t in t_coords]

    image_latent_ids = []
    for x, t in zip(image_latents, t_coords):
        x = x.squeeze(0)
        _, height, width = x.shape

        x_ids = torch.cartesian_prod(t, torch.arange(height), torch.arange(width), torch.arange(1))
        image_latent_ids.append(x_ids)

    image_latent_ids = torch.cat(image_latent_ids, dim=0)
    image_latent_ids = image_latent_ids.unsqueeze(0)

    return image_latent_ids


# ============================================================
# FLUX.2 Klein 配置类
# ============================================================
class Flux2KleinConfig(BaseTrainerConfig):
    """FLUX.2 Klein Edit 训练配置"""
    
    def __init__(
        self,
        # 必选参数
        target_folder: str,
        prompt: str,
        output_dir: str,
        # Condition 配置
        condition_folder: str = None,
        condition_folders: List[str] = None,
        # 模型路径
        model_id: str = "black-forest-labs/FLUX.2-klein-base-9B",
        # 训练参数
        num_train_steps: int = 5000,
        checkpoint_every_n_steps: int = 500,
        checkpoints_total_limit: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-4,
        lr_warmup_steps: int = 500,
        resolution: int = 1024,
        # 时间步采样
        timestep_type: TimestepType = "linear",
        sigmoid_scale: float = 1.0,
        shift_scale: float = 3.0,
        lognorm_alpha: float = 0.75,
        # 训练模式
        full_training: bool = False,  # True: 全量训练, False: LoKr 微调
        # LoKr 参数 (仅在 full_training=False 时使用)
        full_matrix: bool = True,
        lora_dim: int = 10000,
        lora_alpha: int = 1,
        lokr_factor: int = 4,
        decompose_both: bool = False,
        # 量化
        quantize_transformer: bool = True,
        quantize_text_encoder: bool = True,
        # Block Swap
        blocks_to_swap: int = 0,
        use_pinned_memory: bool = True,
        # 正则化
        noise_offset: float = 0.0,
        # Text Encoder 配置
        text_encoder_layers: Tuple[int, ...] = (9, 18, 27),
        max_sequence_length: int = 512,
        # TensorBoard
        use_tensorboard: bool = True,
        tensorboard_dir: str = None,
        # 多卡训练
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
            **kwargs,
        )
        
        # Flux2 特有配置
        self.target_folder = target_folder
        self.prompt = prompt
        
        # Condition 配置
        if condition_folder:
            self.condition_folders = [condition_folder]
        elif condition_folders:
            self.condition_folders = condition_folders
        else:
            raise ValueError("必须指定 condition_folder 或 condition_folders")
        
        # 时间步采样
        self.timestep_type = timestep_type
        self.sigmoid_scale = sigmoid_scale
        self.shift_scale = shift_scale
        self.lognorm_alpha = lognorm_alpha
        
        # 训练模式
        self.full_training = full_training
        
        # LoKr 参数
        self.full_matrix = full_matrix
        self.lora_dim = lora_dim
        self.lora_alpha = lora_alpha
        self.lokr_factor = lokr_factor
        self.decompose_both = decompose_both
        
        # 其他配置
        self.noise_offset = noise_offset
        
        # Block Swap
        self.blocks_to_swap = blocks_to_swap
        self.use_pinned_memory = use_pinned_memory
        
        # Text Encoder 配置
        self.text_encoder_layers = text_encoder_layers
        self.max_sequence_length = max_sequence_length
        
        # FLUX.2 VAE scale factor
        self.vae_scale_factor = 8
        
        # 缓存目录 (保存在训练集目录下)
        self.cache_dir = os.path.join(target_folder, ".flux2_klein_cache")


# ============================================================
# FLUX.2 Klein 数据集
# ============================================================
class Flux2KleinDataset(Dataset):
    """FLUX.2 Klein Edit 数据集"""
    
    def __init__(
        self,
        target_folder: str,
        condition_folders: List[str],
        prompt: str,
        resolution: int = 1024,
        cache_dir: str = None,
        verbose: bool = True,
    ):
        self.target_folder = target_folder
        self.prompt = prompt
        self.resolution = resolution
        self.embed_cache_dir = cache_dir
        
        if isinstance(condition_folders, str):
            condition_folders = [condition_folders]
        self.condition_folders = condition_folders
        self.num_conditions = len(condition_folders)
        
        supported_exts = ('.jpg', '.jpeg', '.png', '.webp', '.JPEG', '.JPG', '.PNG', '.WEBP')
        target_files = [f for f in os.listdir(target_folder) if f.lower().endswith(supported_exts)]
        
        matched_pairs = []
        skipped_no_match = 0
        skipped_corrupted = 0
        corrupted_files = []
        total_conditions = 0
        
        for target_file in target_files:
            target_name = os.path.splitext(target_file)[0]
            target_path = os.path.join(target_folder, target_file)
            
            condition_paths_list = []
            all_found = True
            for condition_folder in condition_folders:
                condition_files = [f for f in os.listdir(condition_folder) if f.lower().endswith(supported_exts)]
                matched_files = []
                for condition_file in condition_files:
                    condition_stem = os.path.splitext(condition_file)[0]
                    condition_base_id = get_base_id(condition_file)
                    if target_name == condition_stem or target_name == condition_base_id:
                        condition_path = os.path.join(condition_folder, condition_file)
                        matched_files.append(condition_path)
                if not matched_files:
                    all_found = False
                    break
                matched_files.sort()
                condition_paths_list.append(matched_files)
            
            if not all_found:
                skipped_no_match += 1
                continue
            
            all_valid = True
            all_paths = [target_path]
            for paths in condition_paths_list:
                all_paths.extend(paths)
            for img_path in all_paths:
                if not verify_image(img_path):
                    all_valid = False
                    corrupted_files.append(img_path)
                    break
            
            if not all_valid:
                skipped_corrupted += 1
                continue
            
            matched_pairs.append((target_path, condition_paths_list))
            total_conditions += sum(len(paths) for paths in condition_paths_list)
        
        self.pairs = matched_pairs
        
        if verbose:
            print(f"[OK] {len(self.pairs)} 对图片，共 {total_conditions} 张条件图")
            if skipped_no_match > 0:
                print(f"  - skip {skipped_no_match} (no match)")
            if skipped_corrupted > 0:
                print(f"  - skip {skipped_corrupted} (corrupted):")
                for f in corrupted_files[:5]:
                    print(f"    {f}")
                if len(corrupted_files) > 5:
                    print(f"    ... +{len(corrupted_files) - 5}")
        
        if len(self.pairs) == 0:
            raise ValueError(f"未找到有效的图片对，请检查路径:\n  Target: {target_folder}\n  Conditions: {condition_folders}")
        
        self.transform = transforms.Compose([transforms.ToTensor()])
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def set_use_cache(self, use_cache: bool):
        """设置是否使用缓存模式 (跳过图像加载)"""
        self._use_cache = use_cache
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if getattr(self, '_use_cache', False):
            return {'sample_idx': idx}
        
        target_path, condition_paths_list = self.pairs[idx]
        
        target_img = exif_transpose(Image.open(target_path)).convert('RGB')
        target_w, target_h = target_img.size
        
        scale = self.resolution / max(target_w, target_h)
        new_w = int(target_w * scale)
        new_h = int(target_h * scale)
        
        new_w = (new_w // 16) * 16
        new_h = (new_h // 16) * 16
        
        target_img = target_img.resize((new_w, new_h), Image.BICUBIC)
        target_tensor = self.transform(target_img)
        
        condition_tensors = []
        all_condition_paths = []
        for paths_in_folder in condition_paths_list:
            for condition_path in paths_in_folder:
                condition_img = exif_transpose(Image.open(condition_path)).convert('RGB')
                condition_img = condition_img.resize((new_w, new_h), Image.BICUBIC)
                condition_tensor = self.transform(condition_img)
                condition_tensors.append(condition_tensor)
                all_condition_paths.append(condition_path)
        
        return {
            'target': target_tensor,
            'conditions': condition_tensors,
            'condition_paths': all_condition_paths,
            'prompt': self.prompt,
            'sample_idx': idx,
        }


# ============================================================
# FLUX.2 Klein 训练器
# ============================================================
class Flux2KleinTrainer(BaseTrainer):
    """FLUX.2 Klein Edit LoKr 训练器"""
    
    def __init__(self, config: Flux2KleinConfig):
        super().__init__(config)
        self.config: Flux2KleinConfig = config
        
        # LoKr 相关
        self.lokr_modules = None
        self.lokr_module_names = None
        
        # 缓存
        self.prompt_embeds_cache = {}
        self.null_prompt_embeds = None
        
        self.cache_in_memory = False
        
        # VAE BatchNorm 参数
        self.latents_bn_mean = None
        self.latents_bn_std = None
        
        # VAE 位置标志
        self._vae_on_gpu = False
        
        # Fused backward 标志
        self.use_fused_backward = False
    
    def _check_stop(self) -> bool:
        return self.should_stop
    
    def load_models(self):
        """加载所有模型"""
        self._load_text_encoder()
        if self._check_stop():
            return
        
        self._cache_prompt_embeddings()
        if self._check_stop():
            return
        
        self._load_vae_and_transformer()
        if self._check_stop():
            return
        
        self._cache_latents()
        if self._check_stop():
            return
        
        if not self.config.full_training:
            self._apply_lokr()
        else:
            self._prepare_full_training()
    
    def _load_text_encoder(self):
        """加载 Qwen3 Text Encoder"""
        if self.accelerator.is_main_process:
            print("\n阶段 1: 加载 Text Encoder")
            print("=" * 60)
        
        if self._check_stop():
            return
        
        # 加载 Qwen3 tokenizer
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(
            self.config.model_id,
            subfolder="tokenizer",
        )
        
        if self.accelerator.is_main_process:
            print("✓ Tokenizer 加载完成")
        
        # 加载 Qwen3 模型
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
            print("✓ Text Encoder (Qwen3) 加载完成")
    
    def _encode_prompt(self, prompt: str) -> Dict[str, torch.Tensor]:
        """编码单个 prompt，提取多层特征"""
        device = self.text_encoder.device
        dtype = self.text_encoder.dtype
        
        # 使用 chat template
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
        
        # Forward pass
        with torch.no_grad():
            output = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        
        # 提取指定层的特征并拼接
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
    
    def _cache_prompt_embeddings(self):
        """缓存 prompt embeddings (保存到磁盘)"""
        if self.accelerator.is_main_process:
            print("\n阶段 2: 缓存 Prompt Embeddings")
            print("=" * 60)
            os.makedirs(self.config.cache_dir, exist_ok=True)
        
        self.accelerator.wait_for_everyone()
        
        if self._check_stop():
            return
        
        prompt_cache_file = os.path.join(self.config.cache_dir, "prompt_embeds.pt")
        null_cache_file = os.path.join(self.config.cache_dir, "null_prompt_embeds.pt")
        
        if os.path.exists(prompt_cache_file) and os.path.exists(null_cache_file):
            if self.accelerator.is_main_process:
                print("从缓存加载 prompt embeddings...")
            prompt_data = torch.load(prompt_cache_file, map_location='cpu')
            self.null_prompt_embeds = torch.load(null_cache_file, map_location='cpu')
        else:
            if self.accelerator.is_main_process:
                print("编码 null prompt embedding...")
            self.null_prompt_embeds = self._encode_prompt(" ")
            
            if self.accelerator.is_main_process:
                print("编码训练 prompt embedding...")
            prompt_data = self._encode_prompt(self.config.prompt)
            
            if self.accelerator.is_main_process:
                torch.save(prompt_data, prompt_cache_file)
                torch.save(self.null_prompt_embeds, null_cache_file)
                print(f"✓ Prompt embeddings 已保存到: {self.config.cache_dir}")
        
        for idx in range(len(self.dataset)):
            self.prompt_embeds_cache[idx] = prompt_data
        
        if self.accelerator.is_main_process:
            print(f"✓ 已缓存 {len(self.prompt_embeds_cache)} 个 prompt embeddings (共享同一 prompt)")
        
        self._unload_text_encoder()
    
    def _unload_text_encoder(self):
        """卸载 Text Encoder 释放显存"""
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
            print(f"✓ 卸载 Text Encoder 后释放了 {freed_memory:.2f}GB 显存")
    
    def _load_vae_and_transformer(self):
        """加载 VAE 和 Transformer"""
        if self.accelerator.is_main_process:
            print("\n阶段 3: 加载 VAE 和 Transformer")
            print("=" * 60)
        
        if self._check_stop():
            return
        
        # 加载 VAE
        self.vae = AutoencoderKLFlux2.from_pretrained(
            self.config.model_id,
            subfolder="vae",
            torch_dtype=self.config.dtype,
        )
        self.vae.requires_grad_(False)
        self.vae.eval()
        
        # 缓存 BatchNorm 参数
        self.latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1)
        self.latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps)
        
        # VAE 保持在 GPU
        self.vae.to(self.accelerator.device)
        self._vae_on_gpu = True
        
        if self.accelerator.is_main_process:
            print("✓ VAE 加载完成")
        
        if self._check_stop():
            return
        
        # 加载 Transformer
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
            if self.accelerator.is_main_process:
                print(">>> Transformer 已冻结 (requires_grad=False)")
            
            self.transformer.to(self.accelerator.device)
            if self.accelerator.is_main_process:
                print(">>> qfloat8 量化完成，模型已移至 GPU")
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
            use_cpu_offload = blocks_to_swap > 0
            self.transformer.enable_gradient_checkpointing(activation_cpu_offloading=use_cpu_offload)
            if self.accelerator.is_main_process:
                if use_cpu_offload:
                    print("✓ 启用 Gradient Checkpointing (with Activation CPU Offloading)")
                else:
                    print("✓ 启用 Gradient Checkpointing")
        
        # 创建 scheduler
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.config.model_id,
            subfolder="scheduler",
        )
        
        if self.accelerator.is_main_process:
            print("✓ Transformer 和 Scheduler 加载完成")
    
    def _cache_latents(self):
        if self.accelerator.is_main_process:
            print("\n阶段 4: 缓存 Latents (磁盘模式)")
            print("=" * 60)
            os.makedirs(self.config.cache_dir, exist_ok=True)
        
        self.accelerator.wait_for_everyone()
        
        if self._check_stop():
            return
        
        samples_to_encode = []
        cached_count = 0
        for idx in range(len(self.dataset)):
            cache_file = os.path.join(self.config.cache_dir, f"latent_{idx}.pt")
            if os.path.exists(cache_file):
                cached_count += 1
            else:
                samples_to_encode.append(idx)
        
        if self.accelerator.is_main_process:
            print(f"  已有缓存: {cached_count} 个")
            print(f"  需要编码: {len(samples_to_encode)} 个")
        
        if len(samples_to_encode) == 0:
            if self.accelerator.is_main_process:
                print(f"✓ 全部已缓存到磁盘")
            self.dataset.set_use_cache(True)
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
            h, w = sample['target'].shape[1], sample['target'].shape[2]
            resolution_buckets[(h, w)].append((idx, sample))
        
        if self.accelerator.is_main_process:
            print(f"  分桶: {len(resolution_buckets)} 种分辨率")
            for res, items in list(resolution_buckets.items())[:3]:
                print(f"    {res}: {len(items)} 个样本")
            if len(resolution_buckets) > 3:
                print(f"    ... 还有 {len(resolution_buckets) - 3} 种分辨率")
        
        pbar = None
        if self.accelerator.is_main_process:
            pbar = tqdm(total=len(samples_to_encode), desc="缓存到磁盘")
        
        with torch.no_grad():
            for resolution, bucket_items in resolution_buckets.items():
                for idx, sample in bucket_items:
                    if self._check_stop():
                        return
                    
                    target_tensor = sample['target'].unsqueeze(0).to(device, dtype)
                    target_tensor = target_tensor * 2 - 1
                    
                    target_latent = self.vae.encode(target_tensor).latent_dist.mode()
                    target_latent = patchify_latents(target_latent)
                    target_latent = (target_latent - bn_mean) / bn_std
                    target_latent_cpu = target_latent.cpu()
                    
                    cond_latents_list = []
                    for cond_tensor in sample['conditions']:
                        cond_tensor = cond_tensor.unsqueeze(0).to(device, dtype)
                        cond_tensor = cond_tensor * 2 - 1
                        
                        cond_latent = self.vae.encode(cond_tensor).latent_dist.mode()
                        cond_latent = patchify_latents(cond_latent)
                        cond_latent = (cond_latent - bn_mean) / bn_std
                        cond_latents_list.append(cond_latent.cpu())
                    
                    cache_data = {
                        'target_latent': target_latent_cpu,
                        'condition_latents': cond_latents_list,
                    }
                    cache_file = os.path.join(self.config.cache_dir, f"latent_{idx}.pt")
                    torch.save(cache_data, cache_file)
                    
                    if pbar is not None:
                        pbar.update(num_processes)
        
        if pbar is not None:
            pbar.close()
        
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            print(f"✓ 已缓存 {len(self.dataset)} 个样本到磁盘")
            print(f"✓ 缓存目录: {self.config.cache_dir}")
        
        self.dataset.set_use_cache(True)
        
        self._unload_vae()
    
    def _unload_vae(self):
        """卸载 VAE 释放显存"""
        if self.accelerator.is_main_process:
            print(">>> 卸载 VAE...")
        
        self.latents_bn_mean = self.latents_bn_mean.cpu()
        self.latents_bn_std = self.latents_bn_std.cpu()
        
        mem_before = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        del self.vae
        self.vae = None
        self._vae_on_gpu = False
        torch.cuda.empty_cache()
        gc.collect()
        mem_after = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        freed_memory = mem_before - mem_after
        if self.accelerator.is_main_process:
            print(f"✓ 卸载 VAE 后释放了 {freed_memory:.2f}GB 显存")
    
    def _apply_lokr(self):
        """应用 LoKr 到 Transformer"""
        if self.accelerator.is_main_process:
            print("\n>>> 应用 LoKr...")
        
        # 冻结 transformer
        self.transformer.requires_grad_(False)
        
        # 应用 LoKr
        self.lokr_modules, self.lokr_module_names = apply_lokr_to_transformer(
            self.transformer,
            lora_dim=self.config.lora_dim,
            alpha=self.config.lora_alpha,
            factor=self.config.lokr_factor,
            full_matrix=self.config.full_matrix,
            decompose_both=self.config.decompose_both,
            verbose=(self.accelerator.is_main_process),
        )
        
        # 将 LoKr 模块移到 GPU
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
        """创建数据集和 DataLoader"""
        if self.accelerator.is_main_process:
            print("\n创建 FLUX.2 Klein 数据集")
            print("=" * 60)
        
        self.dataset = Flux2KleinDataset(
            target_folder=self.config.target_folder,
            condition_folders=self.config.condition_folders,
            prompt=self.config.prompt,
            resolution=self.config.resolution,
            cache_dir=self.config.cache_dir,
            verbose=(self.accelerator.is_main_process),
        )
        
        condition_counts = set()
        for target_path, condition_paths_list in self.dataset.pairs:
            total_conds = sum(len(paths) for paths in condition_paths_list)
            condition_counts.add(total_conds)
        
        has_variable_conditions = len(condition_counts) > 1
        if has_variable_conditions and self.config.batch_size > 1:
            if self.accelerator.is_main_process:
                print(f"[WARN] condition counts vary ({condition_counts}), forcing batch_size=1")
            self.config.batch_size = 1
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            collate_fn=self.collate_fn,
        )
        
        if self.accelerator.is_main_process:
            print(f"[OK] DataLoader batch_size={self.config.batch_size}")
    
    def collate_fn(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """数据整理函数 - 仅返回索引，图像从缓存加载"""
        sample_indices = [ex['sample_idx'] for ex in examples]
        return {
            'sample_indices': sample_indices,
        }
    
    def train_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        device = self.accelerator.device
        dtype = self.config.dtype
        
        # 注意：不要在这里调用 prepare_block_swap_before_forward()
        # 它已经在 _prepare_for_distributed_training() 中调用过了
        # 每个 train_step 重复调用会导致与 all_reduce 冲突
        
        sample_indices = batch['sample_indices']
        batch_size = len(sample_indices)
        
        target_latents_list = []
        condition_latents_cache = {}
        for idx in sample_indices:
            cache_file = os.path.join(self.config.cache_dir, f"latent_{idx}.pt")
            cached = torch.load(cache_file, map_location='cpu', weights_only=True)
            target_latents_list.append(cached['target_latent'])
            condition_latents_cache[idx] = cached['condition_latents']
        target_latents = torch.cat(target_latents_list, dim=0).to(device, dtype)
        
        _, num_channels, latent_height, latent_width = target_latents.shape
        
        # 2. 添加噪声
        noise = torch.randn_like(target_latents)
        
        if self.config.noise_offset > 0:
            noise = noise + self.config.noise_offset * torch.randn(
                (noise.shape[0], noise.shape[1], 1, 1),
                device=noise.device,
                dtype=noise.dtype
            )
        
        # 采样时间步
        timesteps, _ = sample_timesteps(
            batch_size,
            num_train_timesteps=self.noise_scheduler.config.num_train_timesteps,
            device=device,
            timestep_type=self.config.timestep_type,
            sigmoid_scale=self.config.sigmoid_scale,
            shift=self.config.shift_scale,
            lognorm_alpha=self.config.lognorm_alpha,
        )
        
        # 3. 准备位置 ID (用原始 latents，形状相同)
        target_latent_ids = prepare_latent_ids(target_latents).to(device)
        
        # Flow matching: noisy = (1 - sigma) * x + sigma * noise
        # 保持与输入相同的 dtype 进行噪声添加
        sigmas = (timesteps.float() / self.noise_scheduler.config.num_train_timesteps).to(dtype)
        sigmas = sigmas.view(-1, 1, 1, 1)
        noisy_latents = (1.0 - sigmas) * target_latents + sigmas * noise
        
        sample_idx = sample_indices[0]
        condition_latents_list = [lat.to(device, dtype) for lat in condition_latents_cache[sample_idx]]
        
        # 5. Pack latents
        packed_noisy_latents = pack_latents(noisy_latents)  # [B, H*W, C]
        
        packed_condition_latents = []
        for cond_latents in condition_latents_list:
            packed = pack_latents(cond_latents)
            packed_condition_latents.append(packed)
        
        all_packed = [packed_noisy_latents]
        for packed in packed_condition_latents:
            all_packed.append(packed)
        
        hidden_states = torch.cat(all_packed, dim=1)  # [B, total_seq, C]
        
        condition_image_ids = prepare_image_ids(
            condition_latents_list,
            scale=10,
        ).to(device)
        condition_image_ids = condition_image_ids.repeat(batch_size, 1, 1)
        
        img_ids = torch.cat([target_latent_ids, condition_image_ids], dim=1)
        
        # 6. 从缓存加载 prompt embeddings
        prompt_embeds_list = []
        for sample_idx in sample_indices:
            cache_data = self.prompt_embeds_cache[sample_idx]
            prompt_embeds_list.append(cache_data['prompt_embeds'].squeeze(0))
        
        # Pad to same length
        max_len = max(pe.shape[0] for pe in prompt_embeds_list)
        embed_dim = prompt_embeds_list[0].shape[-1]
        prompt_embeds = torch.zeros(batch_size, max_len, embed_dim, device=device, dtype=dtype)
        
        for i, pe in enumerate(prompt_embeds_list):
            prompt_embeds[i, :pe.shape[0]] = pe.to(device, dtype)
        
        # 生成 text IDs
        txt_ids = prepare_text_ids(prompt_embeds).to(device)
        
        # 7. Transformer 前向传播
        # timesteps 需要转换为模型 dtype 以避免 rms_norm 警告
        timesteps_normalized = (timesteps / 1000.0).to(dtype)
        
        # FLUX.2 Klein 不使用 guidance embedding
        guidance = None
        
        with self.accelerator.autocast():
            model_output = self.transformer(
                hidden_states=hidden_states,
                timestep=timesteps_normalized,
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                txt_ids=txt_ids,
                img_ids=img_ids,
                return_dict=False,
            )[0]
        
        # 8. 只取 target 部分的输出
        target_seq_len = packed_noisy_latents.shape[1]
        model_output = model_output[:, :target_seq_len]
        
        # 9. Unpack
        model_output = unpack_latents_with_ids(model_output, target_latent_ids)
        
        # 10. 计算 loss (Flow Matching)
        target = noise - target_latents
        
        loss = F.mse_loss(model_output.float(), target.float(), reduction='mean')
        
        return loss
    
    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        """获取可训练参数"""
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
        
        if self.config.blocks_to_swap > 0:
            import transformers.optimization
            
            mode_str = "全量训练" if self.config.full_training else "LoKr"
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
        
        optimizer = super().create_optimizer(trainable_params)
        
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
            # 保存 LoKr weights (ComfyUI lycoris 格式)
            lokr_state_dict = {}
            for name, module in zip(self.lokr_module_names, self.lokr_modules):
                # ComfyUI 期望格式: lycoris_{key.replace(".", "_")}.lokr_w1/w2/alpha
                prefix = "lycoris_" + name.replace('.', '_')
                
                # 添加 alpha
                if hasattr(module, 'alpha'):
                    lokr_state_dict[f"{prefix}.alpha"] = module.alpha.cpu()
                
                for param_name, param in module.named_parameters():
                    lokr_state_dict[f"{prefix}.{param_name}"] = param.cpu()
            
            save_file(lokr_state_dict, os.path.join(checkpoint_dir, "lokr_weights.safetensors"))
            
            # 保存配置
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
        
        print(f"✓ 检查点已保存到: {checkpoint_dir}")
    
    def pre_training_hook(self):
        """训练开始前的钩子"""
        if self.accelerator.is_main_process:
            mode_str = "Full Training" if self.config.full_training else "LoKr Training"
            print(f"\nFLUX.2 Klein Edit {mode_str}")
            print(f"时间步采样: {self.config.timestep_type}")
            if not self.config.full_training:
                print(f"LoKr 配置: dim={self.config.lora_dim}, alpha={self.config.lora_alpha}, factor={self.config.lokr_factor}")
            if self.config.noise_offset > 0:
                print(f"Noise offset: {self.config.noise_offset}")
    
    def save_final_model(self):
        self.accelerator.wait_for_everyone()
        
        if not self.accelerator.is_main_process:
            return
        
        output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        actual_steps = self.current_step if self.current_step > 0 else self.config.num_train_steps
        unwrapped_transformer = self.accelerator.unwrap_model(self.transformer)
        
        if self.config.full_training:
            final_dir = os.path.join(output_dir, f"flux2_klein_full_{actual_steps}steps")
            os.makedirs(final_dir, exist_ok=True)
            
            print(f"\n>>> 保存最终模型到: {final_dir}")
            unwrapped_transformer.save_pretrained(
                os.path.join(final_dir, "transformer"),
                safe_serialization=True,
            )
            print(f"[ok] Full Training 模型已保存")
        else:
            final_dir = os.path.join(output_dir, f"flux2_klein_lokr_{actual_steps}steps")
            os.makedirs(final_dir, exist_ok=True)
            
            print(f"\n>>> 保存 LoKr 权重到: {final_dir}")
            
            # ComfyUI lycoris 格式: lycoris_{key.replace(".", "_")}.lokr_w1/w2/alpha
            lokr_state_dict = {}
            for name, module in zip(self.lokr_module_names, self.lokr_modules):
                prefix = "lycoris_" + name.replace('.', '_')
                
                # 添加 alpha
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
            
            print(f"✓ LoKr 权重已保存")
        
        print(f">>> 最终模型保存完成: {final_dir}")


def parse_args():
    """解析命令行参数"""
    import argparse
    parser = argparse.ArgumentParser(description="FLUX.2 Klein Edit Trainer")
    
    # 必选参数
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--target_folder", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    
    # Condition
    parser.add_argument("--condition_folder", type=str, default=None)
    parser.add_argument("--condition_folders", type=str, nargs='+', default=None)
    
    # 训练参数
    parser.add_argument("--num_train_steps", type=int, default=5000)
    parser.add_argument("--checkpoint_every_n_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--resolution", type=int, default=1024)
    
    # 时间步采样
    parser.add_argument("--timestep_type", type=str, default="linear", choices=TIMESTEP_TYPES)
    parser.add_argument("--sigmoid_scale", type=float, default=1.0)
    parser.add_argument("--shift_scale", type=float, default=3.0)
    parser.add_argument("--lognorm_alpha", type=float, default=0.75)
    
    # 训练模式
    parser.add_argument("--full_training", action="store_true")
    parser.add_argument("--blocks_to_swap", type=int, default=0)
    parser.add_argument("--use_pinned_memory", action="store_true")
    
    # LoKr 参数
    parser.add_argument("--lora_dim", type=int, default=10000)
    parser.add_argument("--lora_alpha", type=int, default=1)
    parser.add_argument("--lokr_factor", type=int, default=4)
    parser.add_argument("--full_matrix", action="store_true", default=True)
    
    # 量化
    parser.add_argument("--no_quantize_transformer", action="store_true")
    parser.add_argument("--no_quantize_text_encoder", action="store_true")
    
    # 正则化
    parser.add_argument("--noise_offset", type=float, default=0.0)
    
    # 其他
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    
    return parser.parse_args()


def main():
    fix_windows_encoding()
    args = parse_args()
    
    config = Flux2KleinConfig(
        model_id=args.model_id,
        target_folder=args.target_folder,
        prompt=args.prompt,
        output_dir=args.output_dir,
        condition_folder=args.condition_folder,
        condition_folders=args.condition_folders,
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
        full_training=args.full_training,
        blocks_to_swap=args.blocks_to_swap,
        use_pinned_memory=args.use_pinned_memory,
        lora_dim=args.lora_dim,
        lora_alpha=args.lora_alpha,
        lokr_factor=args.lokr_factor,
        full_matrix=args.full_matrix,
        quantize_transformer=not args.no_quantize_transformer,
        quantize_text_encoder=not args.no_quantize_text_encoder,
        noise_offset=args.noise_offset,
        seed=args.seed,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    
    trainer = Flux2KleinTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()