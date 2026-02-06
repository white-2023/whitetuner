"""
Qwen Image Edit LoKr 训练器

包含：
- LoKr 实现
- QwenEditConfig 配置类
- QwenEditDataset 数据集类
- QwenEditTrainer 训练器类
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
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor
from optimum.quanto import quantize, freeze, qint8, qfloat8, QTensor, QBytesTensor
from safetensors.torch import save_file

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


# ============================================================
# Qwen Edit 配置类
# ============================================================
class QwenEditConfig(BaseTrainerConfig):
    """Qwen Image Edit 训练配置"""
    
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
        model_id: str = r"D:\models\Qwen-Image-Edit-2509",
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
        use_fp8: bool = False,
        blocks_to_swap: int = 0,
        use_pinned_memory: bool = True,
        optimizer_type: str = "adafactor",
        noise_offset: float = 0.0,
        # 缓存
        condition_image_size: int = 1024 * 1024,
        vae_image_size: int = 256 * 256,
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
        )
        
        self.target_folder = target_folder
        if not prompt or not prompt.strip():
            raise ValueError(
                "Qwen Edit 模型需要有效的提示词(prompt)才能训练。\n"
                "请在配置中指定 prompt 参数。"
            )
        self.prompt = prompt
        self.condition_folder = condition_folder
        self.condition_folders = condition_folders
        
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
        
        self.embed_cache_dir = os.path.join(target_folder, "_embed_cache")
        self.condition_image_size = condition_image_size
        self.vae_image_size = vae_image_size
        self.noise_offset = noise_offset
        self.use_fp8 = use_fp8
        self.blocks_to_swap = blocks_to_swap
        self.use_pinned_memory = use_pinned_memory
        self.optimizer_type = optimizer_type


# ============================================================
# Qwen Edit 数据集
# ============================================================
def verify_image(path: str) -> bool:
    """验证图片是否可以正常读取"""
    try:
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False


class QwenEditDataset(Dataset):
    """加载配对的 target 和 condition 图片"""
    
    def __init__(
        self, 
        target_folder: str, 
        condition_folders: List[str], 
        prompt: str, 
        resolution: int = 512, 
        dtype: torch.dtype = torch.bfloat16, 
        verbose: bool = True,
        embed_cache_dir: str = None,
    ):
        super().__init__()
        self.target_folder = target_folder
        self.prompt = prompt
        self.resolution = resolution
        self.dtype = dtype
        self.embed_cache_dir = embed_cache_dir if embed_cache_dir else os.path.join(target_folder, "_embed_cache")
        
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
        
        for target_file in target_files:
            target_name = os.path.splitext(target_file)[0]
            target_path = os.path.join(target_folder, target_file)
            
            condition_paths = []
            all_found = True
            for condition_folder in condition_folders:
                condition_files = [f for f in os.listdir(condition_folder) if f.lower().endswith(supported_exts)]
                found = False
                for condition_file in condition_files:
                    condition_name = os.path.splitext(condition_file)[0]
                    if target_name == condition_name:
                        condition_path = os.path.join(condition_folder, condition_file)
                        condition_paths.append(condition_path)
                        found = True
                        break
                if not found:
                    all_found = False
                    break
            
            if not all_found:
                skipped_no_match += 1
                continue
            
            all_valid = True
            all_paths = [target_path] + condition_paths
            for img_path in all_paths:
                if not verify_image(img_path):
                    all_valid = False
                    corrupted_files.append(img_path)
                    break
            
            if not all_valid:
                skipped_corrupted += 1
                continue
            
            matched_pairs.append((target_path, condition_paths))
        
        self.pairs = matched_pairs
        
        if verbose:
            print(f"✓ 找到 {len(self.pairs)} 对有效的图片 (每对包含 {self.num_conditions} 个 condition)")
            if skipped_no_match > 0:
                print(f"  - 跳过 {skipped_no_match} 张无匹配 condition 的图片")
            if skipped_corrupted > 0:
                print(f"  - 跳过 {skipped_corrupted} 对损坏的图片:")
                for f in corrupted_files[:5]:
                    print(f"    {f}")
                if len(corrupted_files) > 5:
                    print(f"    ... 还有 {len(corrupted_files) - 5} 个")
        
        if len(self.pairs) == 0:
            raise ValueError(f"未找到有效的图片对，请检查路径:\n  Target: {target_folder}\n  Conditions: {condition_folders}")
        
        self.transform = transforms.Compose([transforms.ToTensor()])
    
    def get_cache_path(self, idx: int) -> str:
        """获取缓存文件路径，文件名包含 hash 以区分不同的 prompt + condition 组合"""
        import hashlib
        import base64
        target_path, condition_paths = self.pairs[idx]
        target_name = os.path.splitext(os.path.basename(target_path))[0]
        hash_data = {
            "prompt": self.prompt,
            "condition_paths": condition_paths,
        }
        hash_input = str(hash_data).encode('utf-8')
        hash_str = base64.urlsafe_b64encode(hashlib.md5(hash_input).digest()).decode('ascii')
        hash_str = hash_str.replace('=', '')
        return os.path.join(self.embed_cache_dir, f"{target_name}_{hash_str}.pt")
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        target_path, condition_paths = self.pairs[idx]
        
        target_img = exif_transpose(Image.open(target_path)).convert('RGB')
        target_w, target_h = target_img.size
        
        scale = self.resolution / max(target_w, target_h)
        new_w = int(target_w * scale)
        new_h = int(target_h * scale)
        
        new_w = (new_w // 32) * 32
        new_h = (new_h // 32) * 32
        
        target_img = target_img.resize((new_w, new_h), Image.BICUBIC)
        target_tensor = self.transform(target_img)
        
        condition_tensors = []
        for condition_path in condition_paths:
            condition_img = exif_transpose(Image.open(condition_path)).convert('RGB')
            condition_img = condition_img.resize((new_w, new_h), Image.BICUBIC)
            condition_tensor = self.transform(condition_img)
            condition_tensors.append(condition_tensor)
        
        return {
            'target': target_tensor,
            'conditions': condition_tensors,
            'condition_paths': condition_paths,
            'prompt': self.prompt,
            'sample_idx': idx,
        }


# ============================================================
# Qwen Edit 训练器
# ============================================================
class QwenEditTrainer(BaseTrainer):
    """Qwen Image Edit LoKr 训练器"""
    
    def __init__(self, config: QwenEditConfig):
        super().__init__(config)
        self.config: QwenEditConfig = config
        
        # LoKr 相关
        self.lokr_modules = None
        self.lokr_module_names = None
        
        # 缓存
        self.conditioning_embeds_cache = {}
        self.null_prompt_embeds = None
    
    def load_models(self):
        """加载所有模型"""
        self._load_text_encoder()
        if self._check_stop():
            return
        
        self._cache_conditioning_embeddings()
        if self._check_stop():
            return
        
        self._load_vae_and_transformer()
        if self._check_stop():
            return
        
        self._apply_lokr()
        if self._check_stop():
            return
        
        self._prepare_for_ddp()
    
    def _load_text_encoder(self):
        """加载文本编码器"""
        if self.accelerator.is_main_process:
            print("\n阶段 1: Text Encoder 处理")
            print("=" * 60)
        
        if self._check_stop():
            if self.accelerator.is_main_process:
                print("检测到停止信号，跳过加载 Text Encoder")
            return
        
        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.model_id,
            subfolder="text_encoder",
            torch_dtype=self.config.dtype,
            low_cpu_mem_usage=True,
        )
        self.text_encoder.requires_grad_(False)
        
        if self.accelerator.is_main_process:
            print("✓ 保留 text_encoder.model.visual (用于编码条件图)")
            print("  (下一步将加载 Transformer 模型，速度取决于硬盘 IO)")
        
        if self._check_stop():
            return
        
        if self.config.quantize_text_encoder:
            exclude_patterns = ["*embed*", "*lm_head*"]
            quantize(self.text_encoder, weights=self.config.quantize_level, exclude=exclude_patterns)
            freeze(self.text_encoder)
        
        self.text_encoder.to(self.accelerator.device)
        
        self.tokenizer = Qwen2Tokenizer.from_pretrained(
            self.config.model_id,
            subfolder="tokenizer",
        )
        
        self.processor = Qwen2VLProcessor.from_pretrained(
            self.config.model_id,
            subfolder="processor",
        )
        
        if self.accelerator.is_main_process:
            print(f"✓ Text Encoder、Tokenizer 和 Processor 加载完成")
    
    def _check_stop(self, stage: str = None) -> bool:
        return self.check_stop(stage)
    
    def _cache_conditioning_embeddings(self):
        """缓存条件图的 Visual Embeddings (多卡分布式)"""
        if self.accelerator.is_main_process:
            print("\n阶段 2: 缓存条件图的 Visual Embeddings")
            print("=" * 60)
        
        if self._check_stop():
            if self.accelerator.is_main_process:
                print("检测到停止信号，跳过缓存")
            return
        
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            os.makedirs(self.config.embed_cache_dir, exist_ok=True)
        
        self.accelerator.wait_for_everyone()
        
        self.conditioning_embeds_cache = {}
        
        samples_to_encode = []
        for idx in range(len(self.dataset)):
            cache_file = self.dataset.get_cache_path(idx)
            if os.path.exists(cache_file):
                self.conditioning_embeds_cache[idx] = torch.load(cache_file, map_location='cpu')
            else:
                samples_to_encode.append(idx)
        
        num_processes = self.accelerator.num_processes
        process_index = self.accelerator.process_index
        
        if self.accelerator.is_main_process:
            if len(self.conditioning_embeds_cache) == 0:
                print(f"没有缓存，现在开始创建缓存 ({len(samples_to_encode)} 个样本)")
            else:
                print(f"已加载 {len(self.conditioning_embeds_cache)} 个缓存，需要编码 {len(samples_to_encode)} 个样本")
            print(f"使用 {num_processes} 个 GPU 进行分布式缓存")
        
        if len(samples_to_encode) > 0:
            my_samples = []
            for i, idx in enumerate(samples_to_encode):
                if i % num_processes == process_index:
                    my_samples.append(idx)
            
            cache_batch_size = 4
            
            if self.accelerator.is_main_process:
                print(f"每个 GPU 分配任务: ", end="")
                for rank in range(num_processes):
                    count = len([i for i, idx in enumerate(samples_to_encode) if i % num_processes == rank])
                    print(f"GPU{rank}={count} ", end="")
                print()
                print(f"每个 GPU 每次处理 {cache_batch_size} 张，总共每次处理 {cache_batch_size * num_processes} 张")
            
            pbar = None
            if self.accelerator.is_main_process:
                pbar = tqdm(total=len(samples_to_encode), desc="缓存条件图embeddings (所有GPU)")
            
            with torch.no_grad():
                for batch_start in range(0, len(my_samples), cache_batch_size):
                    if self._check_stop():
                        if self.accelerator.is_main_process:
                            print(f"\n检测到停止信号，停止缓存")
                        break
                    
                    batch_indices = my_samples[batch_start:batch_start + cache_batch_size]
                    
                    for idx in batch_indices:
                        sample = self.dataset[idx]
                        condition_paths = sample['condition_paths']
                        
                        processed_images = []
                        for condition_path in condition_paths:
                            img = Image.open(condition_path).convert('RGB')
                            
                            ratio = img.width / img.height
                            width = math.sqrt(self.config.condition_image_size * ratio)
                            height = width / ratio
                            width = round(width / 32) * 32
                            height = round(height / 32) * 32
                            
                            img = img.resize((int(width), int(height)), Image.BICUBIC)
                            processed_images.append(img)
                        
                        num_images = len(processed_images)
                        image_placeholders = "<|vision_start|><|image_pad|><|vision_end|>" * num_images
                        formatted_prompt = f"{image_placeholders}{self.config.prompt}"
                        
                        inputs = self.processor(
                            text=[formatted_prompt],
                            images=processed_images,
                            return_tensors="pt",
                            padding=True,
                        )
                        device = self.accelerator.device
                        input_ids = inputs.input_ids.to(device)
                        attention_mask = inputs.attention_mask.to(device)
                        pixel_values = inputs.pixel_values.to(device)
                        image_grid_thw = inputs.image_grid_thw.to(device)
                        
                        outputs = self.text_encoder(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            image_grid_thw=image_grid_thw,
                            output_hidden_states=True,
                        )
                        
                        prompt_embed = outputs.hidden_states[-1].cpu()
                        prompt_mask = attention_mask.cpu()
                        
                        cache_data = {
                            'prompt_embeds': prompt_embed,
                            'attention_mask': prompt_mask,
                        }
                        cache_file = self.dataset.get_cache_path(idx)
                        torch.save(cache_data, cache_file)
                        self.conditioning_embeds_cache[idx] = cache_data
                    
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
            if idx not in self.conditioning_embeds_cache:
                cache_file = self.dataset.get_cache_path(idx)
                if os.path.exists(cache_file):
                    self.conditioning_embeds_cache[idx] = torch.load(cache_file, map_location='cpu')
        
        if self.accelerator.is_main_process:
            print(f"✓ 条件图embeddings缓存完成，共 {len(self.conditioning_embeds_cache)} 个样本")
            print(f"✓ 缓存保存在: {self.config.embed_cache_dir}")
        
        # 注意: Qwen Edit 不支持 prompt dropout
        # 因为它需要 VL 模型同时编码条件图和文本
        # "空 prompt" 需要每个样本单独编码（保留条件图），实现复杂且显存翻倍
        self.null_prompt_embeds = None
        
        mem_before = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        del self.text_encoder, self.processor, self.tokenizer
        self.text_encoder = None
        self.processor = None
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
            if self.accelerator.is_main_process:
                print("检测到停止信号，跳过加载 VAE 和 Transformer")
            return
        
        self.vae = AutoencoderKLQwenImage.from_pretrained(
            self.config.model_id,
            subfolder="vae",
            torch_dtype=self.config.dtype,
        )
        self.vae.requires_grad_(False)
        self.vae.eval()
        
        # 如果不使用 block swap，将 VAE 保持在 GPU 上；否则在 train_step 中动态移动
        if self.config.blocks_to_swap == 0:
            self.vae.to(self.accelerator.device)
            self._vae_on_gpu = True
        else:
            self._vae_on_gpu = False
        
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
        if self.config.blocks_to_swap > 0 and not self.config.full_training:
            if not use_fp8:
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
        
        if self._check_stop():
            return
        
        self._use_fp8 = use_fp8
        
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
                print(f">>> Enabling block swap: {self.config.blocks_to_swap} blocks")
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
        elif hasattr(self.transformer, 'gradient_checkpointing'):
            self.transformer.gradient_checkpointing = True
            if self.accelerator.is_main_process:
                print("✓ 启用 Gradient Checkpointing")
        
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=1.0,
            use_dynamic_shifting=True,
        )
    
    def _apply_lokr(self):
        """应用 LoKr 或设置 Full Training"""
        if self._check_stop():
            if self.accelerator.is_main_process:
                print("检测到停止信号，跳过应用 LoKr")
            return
        
        if self.config.full_training:
            # Full Training 模式：不使用 LoKr，直接训练所有参数
            self.lokr_modules = []
            self.lokr_module_names = []
            
            trainable_params_count = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
            total_params_count = sum(p.numel() for p in self.transformer.parameters())
            
            if self.accelerator.is_main_process:
                print(f"\n✓ Full Training 模式")
                print(f"  - 可训练参数: {trainable_params_count:,}")
                print(f"  - 总参数: {total_params_count:,}")
                print(f"  - 可训练比例: {trainable_params_count / total_params_count * 100:.2f}%")
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
            print(f"  - LoKr模块数量: {len(self.lokr_modules)}")
            print(f"  - 可训练参数: {trainable_params_count:,}")
            print(f"  - 总参数: {total_params_count:,}")
            print(f"  - 可训练比例: {trainable_params_count / total_params_count * 100:.2f}%")
            lokr_mem = trainable_params_count * 4 / 1024**3
            print(f"  - LoKr 参数显存 (float32): {lokr_mem:.2f} GB")
            if torch.cuda.is_available():
                print(f"  - 当前 GPU 已分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
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
    
    def create_dataset(self):
        """创建数据集和 DataLoader"""
        if self.accelerator.is_main_process:
            print("\n创建数据集和 DataLoader")
            print("=" * 60)
        
        # 确定 condition 文件夹
        condition_folders = None
        if self.config.condition_folders is not None:
            condition_folders = self.config.condition_folders
        elif self.config.condition_folder is not None:
            condition_folders = self.config.condition_folder
        else:
            raise ValueError("请至少设置 condition_folder 或 condition_folders 之一")
        
        self.dataset = QwenEditDataset(
            target_folder=self.config.target_folder,
            condition_folders=condition_folders,
            prompt=self.config.prompt,
            resolution=self.config.resolution,
            dtype=self.config.dtype,
            verbose=self.accelerator.is_main_process,
            embed_cache_dir=self.config.embed_cache_dir,
        )
        
        def collate_fn(batch):
            targets = torch.stack([item['target'] for item in batch])
            prompts = [item['prompt'] for item in batch]
            
            num_conditions = len(batch[0]['conditions'])
            conditions = []
            for i in range(num_conditions):
                condition_i = torch.stack([item['conditions'][i] for item in batch])
                conditions.append(condition_i)
            
            sample_indices = [item['sample_idx'] for item in batch]
            
            return {
                'target': targets,
                'conditions': conditions,
                'sample_indices': sample_indices,
                'prompt': prompts[0],
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
            if self.accelerator.num_processes > 1:
                effective_batch_size = self.config.batch_size * self.accelerator.num_processes * self.config.gradient_accumulation_steps
                print(f"✓ 有效批次大小: {effective_batch_size}")
    
    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        """获取可训练参数"""
        if self.config.full_training:
            return [p for p in self.transformer.parameters() if p.requires_grad]
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
                                    tensor.grad = self.accelerator.reduce(tensor.grad, reduction="mean")
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
        target_images = batch['target'].to(device, self.config.dtype)
        conditions_list = batch['conditions']
        sample_indices = batch['sample_indices']
        
        batch_size = target_images.shape[0]
        
        # 1. VAE 编码 target 图像（仅在 block swap 模式下按需移动 VAE）
        with torch.no_grad():
            if not self._vae_on_gpu and self.vae.device != device:
                self.vae.to(device)
            
            target_images_rescaled = target_images * 2 - 1
            target_images_rescaled = target_images_rescaled.unsqueeze(2)
            
            encoded = self.vae.encode(target_images_rescaled).latent_dist.sample()
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
        
        num_channels = latents.shape[1]
        latent_height = latents.shape[2]
        latent_width = latents.shape[3]
        
        # 2. 添加噪声
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
        
        # 3. Pack target latents
        pixel_height = latent_height * self.config.vae_scale_factor
        pixel_width = latent_width * self.config.vae_scale_factor
        
        packed_latents = pack_latents(
            noisy_latents,
            batch_size,
            num_channels,
            latent_height,
            latent_width,
        )
        base_packed_tokens = packed_latents
        
        # 4. 编码并 pack 所有 condition 图像
        packed_latents_list = torch.chunk(packed_latents, batch_size, dim=0)
        packed_latents_with_controls_list = []
        img_shapes = [[(1, latent_height // 2, latent_width // 2)] for _ in range(batch_size)]
        
        with torch.no_grad():
            for b in range(batch_size):
                controls = []
                for condition_img in conditions_list:
                    cond_img = condition_img[b:b+1].to(device, self.config.dtype)
                    
                    cond_img_rescaled = cond_img * 2 - 1
                    cond_img_rescaled = cond_img_rescaled.unsqueeze(2)
                    
                    cond_encoded = self.vae.encode(cond_img_rescaled).latent_dist.sample()
                    if cond_encoded.dim() == 5:
                        cond_encoded = cond_encoded.squeeze(2)
                    
                    control_latent = (cond_encoded - latents_mean) * latents_std
                    
                    cl_height, cl_width = control_latent.shape[2], control_latent.shape[3]
                    packed_control = pack_latents(
                        control_latent,
                        1,
                        control_latent.shape[1],
                        cl_height,
                        cl_width,
                    )
                    
                    controls.append(packed_control)
                    img_shapes[b].append((1, cl_height // 2, cl_width // 2))
                
                control = torch.cat(controls, dim=1)
                packed_with_control = torch.cat([packed_latents_list[b], control], dim=1)
                packed_latents_with_controls_list.append(packed_with_control)
        
            # 仅在 block swap 模式下卸载 VAE 以节省显存
            if not self._vae_on_gpu:
                self.vae.to('cpu')
                torch.cuda.empty_cache()
        
        transformer_inputs = torch.cat(packed_latents_with_controls_list, dim=0)
        
        # 5. 从缓存加载 prompt embeddings
        prompt_embeds_list = []
        prompt_masks_list = []
        
        for sample_idx in sample_indices:
            cache_data = self.conditioning_embeds_cache[sample_idx]
            prompt_embeds_list.append(cache_data['prompt_embeds'].squeeze(0))
            prompt_masks_list.append(cache_data['attention_mask'].squeeze(0))
        

        
        max_len = max(pe.shape[0] for pe in prompt_embeds_list)
        embed_dim = prompt_embeds_list[0].shape[-1]
        prompt_embeds = torch.zeros(batch_size, max_len, embed_dim, device=device, dtype=self.config.dtype)
        attention_mask = torch.zeros(batch_size, max_len, device=device, dtype=torch.int64)
        
        for i, (pe, am) in enumerate(zip(prompt_embeds_list, prompt_masks_list)):
            prompt_embeds[i, :pe.shape[0]] = pe.to(device, self.config.dtype)
            attention_mask[i, :am.shape[0]] = am.to(device)
        
        if attention_mask.dim() == 3 and attention_mask.size(1) == 1:
            attention_mask = attention_mask.squeeze(1)
        txt_seq_lens = attention_mask.sum(dim=1).tolist()
        
        # 6. Transformer 推理
        timesteps_normalized = timesteps.float() / 1000.0
        
        with self.accelerator.autocast():
            model_output = self.transformer(
                hidden_states=transformer_inputs.to(device, self.config.dtype),
                timestep=timesteps_normalized,
                guidance=None,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_mask=attention_mask,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
            )
        
        # 7. Unpack
        model_output = model_output[:, : base_packed_tokens.size(1)]
        
        model_output = unpack_latents(
            model_output,
            pixel_height,
            pixel_width,
            self.config.vae_scale_factor
        )
        if model_output.dim() == 5:
            model_output = model_output.squeeze(2)
        
        # 8. 计算 loss
        target = noise - latents
        
        per_sample_loss = F.mse_loss(model_output.float(), target.float(), reduction="none")
        per_sample_loss = per_sample_loss.mean(dim=[1, 2, 3])
        
        weighted_loss = per_sample_loss * timestep_weights
        loss = weighted_loss.mean()
        
        return loss
    
    def save_checkpoint(self, step: int):
        output_dir = self.config.output_dir or os.path.join(self.script_dir, "output")
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        unwrapped_transformer = self.accelerator.unwrap_model(self.transformer)
        
        if self.config.full_training:
            transformer_dir = os.path.join(checkpoint_dir, "transformer")
            unwrapped_transformer.save_pretrained(transformer_dir, safe_serialization=True)
            
            training_config = {
                "full_training": True,
                "step": step,
            }
            with open(os.path.join(checkpoint_dir, "training_config.json"), "w") as f:
                json.dump(training_config, f, indent=2)
        else:
            # LoKr 模式: 保存 LoKr 权重
            lokr_state_dict = {}
            for idx, (module, layer_name) in enumerate(zip(self.lokr_modules, self.lokr_module_names)):
                layer_name_formatted = layer_name.replace('.', '_')
                
                if hasattr(module, 'alpha'):
                    key = f"lycoris_{layer_name_formatted}.alpha"
                    lokr_state_dict[key] = module.alpha.cpu()
                
                for param_name, param in module.named_parameters():
                    key = f"lycoris_{layer_name_formatted}.{param_name}"
                    lokr_state_dict[key] = param.cpu()
            
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
    
    def load_checkpoint(self, checkpoint_dir: str):
        """从检查点恢复训练状态"""
        from safetensors.torch import load_file
        
        if not os.path.exists(checkpoint_dir):
            raise ValueError(f"检查点目录不存在: {checkpoint_dir}")
        
        if self.accelerator.is_main_process:
            print(f"✓ 从检查点恢复: {checkpoint_dir}")
        
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
        
        if self.config.full_training:
            # Full training: 从 transformer 目录加载权重
            transformer_dir = os.path.join(checkpoint_dir, "transformer")
            if os.path.exists(transformer_dir):
                if self.accelerator.is_main_process:
                    print(f"  - 加载 transformer 权重...")
                state_dict = load_file(os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors"))
                self.transformer.load_state_dict(state_dict, strict=True)
                if self.accelerator.is_main_process:
                    print(f"  - Transformer 权重已恢复")
        else:
            # LoKr 模式: 加载 LoKr 权重
            lokr_weights_path = os.path.join(checkpoint_dir, "lokr_weights.safetensors")
            if os.path.exists(lokr_weights_path) and self.lokr_modules is not None:
                if self.accelerator.is_main_process:
                    print(f"  - 加载 LoKr 权重...")
                
                lokr_state_dict = load_file(lokr_weights_path)
                
                for module, layer_name in zip(self.lokr_modules, self.lokr_module_names):
                    layer_name_formatted = layer_name.replace('.', '_')
                    
                    for param_name, param in module.named_parameters():
                        key = f"lycoris_{layer_name_formatted}.{param_name}"
                        if key in lokr_state_dict:
                            param.data.copy_(lokr_state_dict[key].to(param.device, param.dtype))
                
                if self.accelerator.is_main_process:
                    print(f"  - LoKr 权重已恢复")
        
        accelerate_state_dir = os.path.join(checkpoint_dir, "accelerate_state")
        if os.path.exists(accelerate_state_dir):
            self.accelerator.load_state(accelerate_state_dir)
            if self.accelerator.is_main_process:
                print(f"  - Accelerate 状态已恢复 (optimizer, scheduler, RNG)")
    
    def save_final_model(self):
        """保存最终模型"""
        self.accelerator.wait_for_everyone()
        
        if not self.accelerator.is_main_process:
            return
        
        print("\n保存最终模型")
        print("=" * 60)
        
        output_dir = self.config.output_dir or os.path.join(self.script_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        actual_steps = self.current_step if self.current_step > 0 else self.adjusted_num_train_steps
        equivalent_single_gpu_steps = actual_steps * self.accelerator.num_processes * self.config.gradient_accumulation_steps
        
        unwrapped_transformer = self.accelerator.unwrap_model(self.transformer)
        
        if self.config.full_training:
            transformer_dir = os.path.join(output_dir, f"transformer_{equivalent_single_gpu_steps}steps")
            unwrapped_transformer.save_pretrained(transformer_dir, safe_serialization=True)
            
            training_config = {
                "full_training": True,
                "model_id": self.config.model_id,
                "resolution": self.config.resolution,
                "prompt": self.config.prompt,
                "trained_steps": actual_steps,
                "equivalent_single_gpu_steps": equivalent_single_gpu_steps,
                "original_num_train_steps": self.config.num_train_steps,
                "stopped_early": self.should_stop,
                "num_gpus": self.accelerator.num_processes,
            }
            config_filename = f"training_config_{equivalent_single_gpu_steps}steps.json"
            with open(os.path.join(output_dir, config_filename), "w") as f:
                json.dump(training_config, f, indent=2)
            
            self.save_checkpoint(actual_steps)
            
            print(f"\n✓ 最终模型已保存到: {transformer_dir}/")
            print(f"  - {config_filename}")
            print(f"✓ 最终检查点已保存到: {output_dir}/checkpoint-{actual_steps}/")
            
            if self.should_stop:
                print(f"\n⏹️ 训练在第 {actual_steps} 步停止")
            else:
                print(f"\n✅ 训练完成!")
            
            print(f"  数据集: {len(self.dataset)} 对图片")
            print(f"  Prompt: '{self.config.prompt}'")
            print(f"  使用 GPU 数量: {self.accelerator.num_processes}")
            print(f"\n✓ 模型格式: Full Finetune")
            print(f"✓ 可直接替换原始 transformer 使用")
        else:
            # LoKr 模式: 保存 LoKr 权重
            lokr_state_dict = {}
            for idx, (module, layer_name) in enumerate(zip(self.lokr_modules, self.lokr_module_names)):
                layer_name_formatted = layer_name.replace('.', '_')
                
                if hasattr(module, 'alpha'):
                    key = f"lycoris_{layer_name_formatted}.alpha"
                    lokr_state_dict[key] = module.alpha.cpu()
                
                for param_name, param in module.named_parameters():
                    key = f"lycoris_{layer_name_formatted}.{param_name}"
                    lokr_state_dict[key] = param.cpu()
            
            metadata = {
                "lora_dim": str(self.config.lora_dim),
                "lora_alpha": str(self.config.lora_alpha),
                "lokr_factor": str(self.config.lokr_factor),
                "full_matrix": str(self.config.full_matrix),
                "model_type": "lokr",
                "base_model": "Qwen-Image-Edit-2509",
                "ss_network_module": "lycoris.kohya",
                "ss_network_dim": str(self.config.lora_dim),
                "ss_network_alpha": str(self.config.lora_alpha),
                "num_gpus": str(self.accelerator.num_processes),
            }
            
            model_filename = f"qwen_edit_lokr_{equivalent_single_gpu_steps}steps.safetensors"
            config_filename = f"lokr_config_{equivalent_single_gpu_steps}steps.json"
            
            save_file(lokr_state_dict, os.path.join(output_dir, model_filename), metadata=metadata)
            
            lokr_config = {
                "lora_dim": self.config.lora_dim,
                "lora_alpha": self.config.lora_alpha,
                "lokr_factor": self.config.lokr_factor,
                "full_matrix": self.config.full_matrix,
                "decompose_both": self.config.decompose_both,
                "num_modules": len(self.lokr_modules),
                "model_id": self.config.model_id,
                "resolution": self.config.resolution,
                "prompt": self.config.prompt,
                "format": "lycoris/lokr",
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
            
            print(f"\n✓ 最终模型已保存到: {output_dir}/")
            print(f"  - {model_filename}")
            print(f"  - {config_filename}")
            print(f"✓ 最终检查点已保存到: {checkpoint_dir}/checkpoint-{actual_steps}/")
            
            if self.should_stop:
                print(f"\n⏹️ 训练在第 {actual_steps} 步停止")
            else:
                print(f"\n✅ 训练完成!")
            
            print(f"  数据集: {len(self.dataset)} 对图片")
            print(f"  Prompt: '{self.config.prompt}'")
            print(f"  使用 GPU 数量: {self.accelerator.num_processes}")
            if self.config.full_training:
                print(f"\n模型格式: Full (完整模型)")
            else:
                print(f"\n模型格式: LyCORIS LoKr")
            print(f"可在 ComfyUI 中直接加载")
    
    def pre_training_hook(self):
        """训练前钩子"""
        if self.accelerator.is_main_process:
            print(f"✓ Edit-v2 模式：使用预缓存的条件图 embeddings")
            print(f"✓ 显存优化: 预缓存 + 动态权重合并 + Gradient Checkpointing")
            print(f"时间步采样: {self.config.timestep_type}")
            if self.config.timestep_type == "sigmoid":
                print(f"  - sigmoid_scale: {self.config.sigmoid_scale}")
            elif self.config.timestep_type == "shift":
                print(f"  - shift_scale: {self.config.shift_scale}")
            elif self.config.timestep_type == "lognorm_blend":
                print(f"  - lognorm_alpha: {self.config.lognorm_alpha}")
            print(f"Noise offset: {self.config.noise_offset}")
            print(f"优化器: Adafactor")
            if getattr(self, '_use_fp8', self.config.use_fp8):
                print(f"FP8: 已启用")
            if self.config.blocks_to_swap > 0:
                print(f"Block Swap: {self.config.blocks_to_swap} 个 blocks")


# ============================================================
# 主函数
# ============================================================
def main():
    import argparse
    
    fix_windows_encoding()
    
    parser = argparse.ArgumentParser(description="Qwen Image Edit LoKr 训练器")
    
    # 必选参数
    parser.add_argument("--model_id", type=str, required=True, help="模型路径")
    parser.add_argument("--target_folder", type=str, required=True, help="目标图片文件夹")
    parser.add_argument("--prompt", type=str, required=True, help="训练提示词")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    
    # Condition 文件夹
    parser.add_argument("--condition_folder", type=str, default=None, help="条件图片文件夹")
    parser.add_argument("--condition_folder_2", type=str, default=None, help="条件图片文件夹2")
    parser.add_argument("--condition_folder_3", type=str, default=None, help="条件图片文件夹3")
    
    # 训练参数
    parser.add_argument("--num_train_steps", type=int, default=5000, help="训练步数")
    parser.add_argument("--checkpoint_every_n_steps", type=int, default=500, help="检查点保存间隔")
    parser.add_argument("--checkpoints_total_limit", type=int, default=5, help="检查点数量限制")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="学习率预热步数")
    parser.add_argument("--resolution", type=int, default=1024, help="图片分辨率")
    
    # 时间步采样
    parser.add_argument("--timestep_type", type=str, default="linear",
                        choices=TIMESTEP_TYPES, help="时间步采样类型")
    parser.add_argument("--sigmoid_scale", type=float, default=1.0, help="Sigmoid 分布集中程度")
    parser.add_argument("--shift_scale", type=float, default=3.0, help="Shift 采样偏移程度")
    parser.add_argument("--lognorm_alpha", type=float, default=0.75, help="LogNorm 混合比例")
    
    # 正则化
    parser.add_argument("--noise_offset", type=float, default=0.0, help="Noise offset")
    
    # 训练模式
    parser.add_argument("--full_training", action="store_true", help="全量训练模式 (不使用 LoKr，需要 block_swap)")
    
    # FP8 和 Block Swap
    parser.add_argument("--use_fp8", action="store_true", help="使用 FP8 量化代替 qint8 (仅 LoKr 模式)")
    parser.add_argument("--blocks_to_swap", type=int, default=0, help="交换到 CPU 的 transformer blocks 数量 (0 禁用)")
    parser.add_argument("--optimizer_type", type=str, default="adafactor", help="优化器类型 (仅支持 adafactor)")
    
    # 多卡训练参数
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="混合精度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="从检查点恢复训练")
    
    args = parser.parse_args()
    
    # 构建 condition_folders 列表
    condition_folders = []
    if args.condition_folder:
        condition_folders.append(args.condition_folder)
    if args.condition_folder_2:
        condition_folders.append(args.condition_folder_2)
    if args.condition_folder_3:
        condition_folders.append(args.condition_folder_3)
    
    if not condition_folders:
        raise ValueError("至少需要提供一个 condition_folder")
    
    config = QwenEditConfig(
        model_id=args.model_id,
        target_folder=args.target_folder,
        condition_folders=condition_folders if len(condition_folders) > 1 else None,
        condition_folder=condition_folders[0] if len(condition_folders) == 1 else None,
        prompt=args.prompt,
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
        full_training=args.full_training,
        noise_offset=args.noise_offset,
        use_fp8=args.use_fp8,
        blocks_to_swap=args.blocks_to_swap,
        optimizer_type=args.optimizer_type,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        seed=args.seed,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    
    trainer = QwenEditTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()


