"""
ZImage 训练器

基于 ai-toolkit 的 ZImage 实现
- 支持多种时间步采样方法
- 完整 1000 步时间步范围
- 支持 LoKr 微调 和 全量训练两种模式
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
from safetensors.torch import save_file, load_file

try:
    from zimage_modules import load_zimage_transformer
except ImportError:
    from whitetuner_diffusers.zimage_modules import load_zimage_transformer

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

from lokr import (
    factorization,
    make_kron,
    LokrModule,
    apply_lokr_to_transformer,
)


# ============================================================
# ZImage 调度器配置
# ============================================================
ZIMAGE_SCHEDULER_CONFIG = {
    "num_train_timesteps": 1000,
    "use_dynamic_shifting": False,
    "shift": 3.0,
}


# ============================================================
# ZImage 配置类
# ============================================================
class ZImageConfig(BaseTrainerConfig):
    """ZImage 训练配置 (支持 LoKr 和 全量训练，使用 qfloat8 量化)"""
    
    def __init__(
        self,
        image_folder: str,
        output_dir: str,
        model_id: str = "Tongyi-MAI/Z-Image-Turbo",
        num_train_steps: int = 5000,
        checkpoint_every_n_steps: int = 500,
        checkpoints_total_limit: int = 3,
        batch_size: int = 1,
        learning_rate: float = 1e-4,
        lr_warmup_steps: int = 100,
        resolution: int = 1024,
        timestep_type: TimestepType = "sigmoid",
        sigmoid_scale: float = 1.0,
        shift_scale: float = 3.0,
        lognorm_alpha: float = 0.75,
        min_timestep: int = None,
        max_timestep: int = None,
        loss_weighting_scheme: LossWeightingType = "none",
        # 训练模式
        full_training: bool = False,  # True: 全量训练, False: LoKr 微调
        # Caption 设置
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
            quantize_transformer=True,  # 始终使用 qfloat8
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
        
        # 训练模式
        self.full_training = full_training
        
        # LoKr 参数（固定值）
        self.lora_dim = 10000
        self.lora_alpha = 1
        self.lokr_factor = 4
        self.full_matrix = True
        self.decompose_both = False
        
        # Caption 设置
        self.use_caption = use_caption
        self.caption_ext = caption_ext
        self.default_caption = default_caption
        self.prompt_dropout_prob = prompt_dropout_prob
        self.noise_offset = noise_offset
        self.cache_latents = cache_latents
        
        if cache_dir is None:
            self.cache_dir = os.path.join(image_folder, ".zimage_cache")
        else:
            self.cache_dir = cache_dir


# ============================================================
# ZImage 数据集
# ============================================================
def verify_image(path: str) -> bool:
    """验证图片是否可以正常读取"""
    try:
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False


class ZImageDataset(Dataset):
    """ZImage 训练数据集 - 单图 + caption"""
    
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
                print(f"  - 跳过 {skipped_corrupted} 张损坏的图片:")
                for f in corrupted_files[:5]:
                    print(f"    {f}")
                if len(corrupted_files) > 5:
                    print(f"    ... 还有 {len(corrupted_files) - 5} 个")
        
        if len(self.samples) == 0:
            raise ValueError(f"未找到有效图片，请检查路径: {image_folder}")
        
        self.transform = transforms.Compose([transforms.ToTensor()])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # 加载图片
        image = exif_transpose(Image.open(sample['image_path'])).convert('RGB')
        orig_w, orig_h = image.size
        
        # 计算缩放比例，保持宽高比
        scale = self.resolution / max(orig_w, orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        # 确保是 32 的倍数（ZImage 的 bucket divisibility 是 32）
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


# ============================================================
# ZImage 训练器
# ============================================================
class ZImageTrainer(BaseTrainer):
    """ZImage 训练器 (支持 LoKr 和 全量训练，使用 qfloat8 量化)"""
    
    def __init__(self, config: ZImageConfig):
        super().__init__(config)
        self.config: ZImageConfig = config
        
        self.latent_shift = 0.0
        self.latent_scale = 1.0
        
        self.text_embeds_cache = {}
        self.null_prompt_embeds = None
        
        # LoKr 相关
        self.lokr_modules = None
        self.lokr_module_names = None
    
    def _check_stop(self, stage: str = None) -> bool:
        return self.check_stop(stage)
    
    def create_dataset(self):
        """创建数据集"""
        if self.accelerator.is_main_process:
            print("\n创建 ZImage 数据集")
            print("=" * 60)
        
        self.dataset = ZImageDataset(
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
        """加载 ZImage 模型并缓存 embeddings (多卡分布式)"""
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
        
        # 根据训练模式应用 LoKr 或全量训练
        if not self.config.full_training:
            self._apply_lokr()
        else:
            self._setup_full_training()
        if self._check_stop():
            return
        
        self._prepare_for_ddp()
    
    def _load_text_encoder_and_vae(self):
        """加载 Text Encoder 和 VAE（所有卡都加载，用于分布式缓存）"""
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
            print("✓ Text Encoder 加载完成")
        
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
            print("✓ VAE 加载完成")
        
        self.flush_memory()
    
    def _cache_embeddings_and_latents(self):
        """缓存 text embeddings 和 latents (多卡分布式)"""
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
            
            if self.accelerator.is_main_process:
                print(f"每个 GPU 分配任务: ", end="")
                for rank in range(num_processes):
                    count = len([i for i, idx in enumerate(samples_to_encode) if i % num_processes == rank])
                    print(f"GPU{rank}={count} ", end="")
                print()
                print(f"每个 GPU 每次处理 {cache_batch_size} 张，总共每次处理 {cache_batch_size * num_processes} 张")
            
            pbar = None
            if self.accelerator.is_main_process:
                pbar = tqdm(total=len(samples_to_encode), desc="缓存 embeddings 和 latents (所有GPU)")
            
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
            print(f"✓ 缓存完成，共 {len(self.text_embeds_cache)} 个样本")
        
        mem_before = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        del self.text_encoder, self.tokenizer
        self.text_encoder = None
        self.tokenizer = None
        
        if self.config.cache_latents:
            del self.vae
            self.vae = None
            if self.accelerator.is_main_process:
                print("✓ 已卸载 Text Encoder 和 VAE")
        else:
            if self.accelerator.is_main_process:
                print("✓ 已卸载 Text Encoder（保留 VAE 用于实时编码）")
        
        torch.cuda.empty_cache()
        gc.collect()
        mem_after = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        freed_memory = mem_before - mem_after
        if self.accelerator.is_main_process:
            print(f"✓ 释放了 {freed_memory:.2f}GB 显存")
    
    def _load_transformer(self):
        """加载 Transformer（训练目标）"""
        if self.accelerator.is_main_process:
            print("\n阶段 3: 加载 Transformer（取决于硬盘IO）", flush=True)
            print("=" * 60, flush=True)
            print(">>> 加载 ZImage Transformer...", flush=True)
        
        if self._check_stop():
            if self.accelerator.is_main_process:
                print("检测到停止信号，跳过加载 Transformer")
            return
        
        dtype = self.config.dtype
        model_path = self.config.model_id
        
        transformer_path = os.path.join(model_path, "transformer")
        if not os.path.exists(transformer_path):
            transformer_path = model_path

        self.transformer = load_zimage_transformer(
            transformer_path,
            device=str(self.accelerator.device),
            dtype=dtype,
        )
        
        if self._check_stop():
            return
        
        # 根据训练模式设置 requires_grad
        if self.config.full_training:
            self.transformer.requires_grad_(True)
            if self.accelerator.is_main_process:
                print(">>> Full Training 模式：transformer 参数可训练")
        else:
            # LoKr 模式
            self.transformer.requires_grad_(False)
            if self.accelerator.is_main_process:
                print(">>> LoKr 模式：Transformer 已冻结 (requires_grad=False)")
        
        self.transformer.train()
        
        # 启用 Gradient Checkpointing
        if hasattr(self.transformer, 'enable_gradient_checkpointing'):
            self.transformer.enable_gradient_checkpointing()
            if self.accelerator.is_main_process:
                print("✓ 启用 Gradient Checkpointing")
        
        if self.accelerator.is_main_process:
            total_params = sum(p.numel() for p in self.transformer.parameters())
            trainable_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
            print(f"✓ Transformer 加载完成")
            print(f"  - 总参数: {total_params:,}")
            print(f"  - 可训练参数: {trainable_params:,}")
        
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(**ZIMAGE_SCHEDULER_CONFIG)
        
        if self.accelerator.is_main_process:
            print(f"\n✓ 调度器: FlowMatchEulerDiscreteScheduler")
            print(f"  - 时间步: {ZIMAGE_SCHEDULER_CONFIG['num_train_timesteps']}")
            print(f"  - Shift: {ZIMAGE_SCHEDULER_CONFIG['shift']}")
    
    def _apply_lokr(self):
        """应用 LoKr 到 Transformer"""
        if self._check_stop():
            return
        
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
            verbose=self.accelerator.is_main_process
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

    def _setup_full_training(self):
        """设置全量训练模式"""
        if self.accelerator.is_main_process:
            print("\n>>> Full Training 模式设置...")
        
        self.lokr_modules = []
        self.lokr_module_names = []
        
        if self.accelerator.is_main_process:
            trainable_count = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
            total_count = sum(p.numel() for p in self.transformer.parameters())
            print(f"    可训练参数: {trainable_count:,} / {total_count:,}")

    def _prepare_for_ddp(self):
        if self.accelerator.is_main_process:
            print("\n阶段 4: 准备分布式训练")
            print("=" * 60)
        
        if self.accelerator.num_processes > 1:
            self.transformer = self.accelerator.prepare(self.transformer)
            if self.accelerator.is_main_process:
                print(f"✓ Transformer 已准备 (DDP: True)")
        else:
            if self.accelerator.is_main_process:
                print(f"✓ Transformer 已准备 (单卡模式，跳过 DDP 包装)")
    
    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        """获取可训练参数"""
        if self.config.full_training:
            return [p for p in self.transformer.parameters() if p.requires_grad]
        return [p for module in self.lokr_modules for p in module.parameters() if p.requires_grad]
    
    def create_optimizer(self, trainable_params: List[torch.nn.Parameter]):
        """创建优化器"""
        mode_str = "Full Training" if self.config.full_training else "LoKr"
        
        if self.accelerator.is_main_process:
            print(f"{mode_str}: 使用 AdamW 优化器")
        
        self.use_fused_backward = False
        return torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )
    
    def train_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """执行一个训练步骤 - 使用缓存的 embeddings 和 latents"""
        device = self.accelerator.device
        sample_indices = batch['sample_indices']
        
        batch_size = len(sample_indices)
        
        # 1. 从缓存加载 latents（使用 float32 避免精度问题）
        latents_list = []
        prompt_embeds_list = []
        
        for idx in sample_indices:
            cache_data = self.text_embeds_cache[idx]
            prompt_embeds_list.append(cache_data['prompt_embeds'].squeeze(0).float())
            
            if self.config.cache_latents and cache_data['latents'] is not None:
                latents_list.append(cache_data['latents'].squeeze(0).float())
        
        # 如果缓存了 latents，直接使用（保持 float32）
        if self.config.cache_latents and latents_list:
            latents = torch.stack(latents_list, dim=0).to(device)
        else:
            # 否则实时编码（需要 VAE）
            images = batch['image'].to(device, torch.float32)
            with torch.no_grad():
                images_normalized = images * 2 - 1
                latents = self.vae.encode(images_normalized).latent_dist.sample()
                latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
                latents = latents.float()
        
        # 2. Prompt dropout: 随机将部分 prompt 替换为空 prompt
        if self.config.prompt_dropout_prob > 0 and self.null_prompt_embeds is not None:
            dropout_mask = torch.rand(batch_size) < self.config.prompt_dropout_prob
            null_embed = self.null_prompt_embeds.squeeze(0).float()
            for i in range(batch_size):
                if dropout_mask[i]:
                    prompt_embeds_list[i] = null_embed
        
        # 3. 批量处理 prompt embeds（padding，保持 float32）
        cap_ori_lens = [int(pe.shape[0]) for pe in prompt_embeds_list]
        max_len = max(pe.shape[0] for pe in prompt_embeds_list)
        embed_dim = prompt_embeds_list[0].shape[-1]
        prompt_embeds = torch.zeros(
            batch_size, max_len, embed_dim,
            device=device, dtype=torch.float32
        )
        for i, pe in enumerate(prompt_embeds_list):
            prompt_embeds[i, :pe.shape[0]] = pe.to(device)
        
        # 5. 采样时间步
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
        
        # 6. 添加噪声（全部在 float32 下计算）
        noise = torch.randn_like(latents)
        
        # 应用 noise offset（帮助模型学习更多样的噪声分布）
        if self.config.noise_offset > 0:
            noise = noise + self.config.noise_offset * torch.randn(
                (noise.shape[0], noise.shape[1], 1, 1), 
                device=noise.device, 
                dtype=noise.dtype
            )
        
        # Flow matching: noisy = (1 - sigma) * clean + sigma * noise
        sigmas = timesteps.float() / self.noise_scheduler.config.num_train_timesteps
        sigmas = sigmas.view(-1, 1, 1, 1)
        noisy_latents = (1 - sigmas) * latents + sigmas * noise
        
        # 7. 准备输入（转为模型需要的 dtype）
        latent_model_input = noisy_latents.to(self.config.dtype).unsqueeze(2)
        
        # ZImage 的时间步格式: (1000 - t) / 1000
        timestep_model_input = ((1000 - timesteps.float()) / 1000).to(self.config.dtype)
        
        # prompt_embeds 也转为模型需要的 dtype
        prompt_embeds = prompt_embeds.to(self.config.dtype)
        
        # 8. Transformer 前向传播
        with self.accelerator.autocast():
            model_output = self.transformer(
                latent_model_input,
                timestep_model_input,
                prompt_embeds,
                cap_ori_lens,
            )
        
        # 9. 处理输出（立即转为 float32 避免 bf16 精度问题导致 NaN）
        noise_pred = model_output.float().squeeze(2)
        noise_pred = -noise_pred  # ZImage 需要取反
        
        # 10. 计算损失 (flow matching: noise - latents)
        target = (noise - latents).detach()
        
        per_sample_loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
        per_sample_loss = per_sample_loss.mean(dim=[1, 2, 3])
        
        # 应用 timestep 采样权重
        weighted_loss = per_sample_loss * timestep_weights
        
        # 应用 loss weighting scheme (sigma_sqrt / cosmap)
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
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        unwrapped_transformer = self.accelerator.unwrap_model(self.transformer)
        
        if self.config.full_training:
            # 全量训练：保存完整 transformer
            transformer_dir = os.path.join(checkpoint_dir, "transformer")
            unwrapped_transformer.save_pretrained(transformer_dir, safe_serialization=True)
            if self.accelerator.is_main_process:
                print(f"  - 全量模型已保存到 {transformer_dir}")
        else:
            # LoKr 模式：保存 LoKr 权重
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
        self._save_gui_config(checkpoint_dir)
    
    def _save_gui_config(self, checkpoint_dir: str):
        """保存 GUI 配置参数"""
        if not self.accelerator.is_main_process:
            return
        
        gui_config = {
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
            "training_mode": "full" if self.config.full_training else "lokr",
            "use_caption": self.config.use_caption,
            "default_caption": self.config.default_caption,
            "prompt_dropout_prob": self.config.prompt_dropout_prob,
            "noise_offset": self.config.noise_offset,
            "checkpoint_every_n_steps": self.config.checkpoint_every_n_steps,
            "checkpoints_total_limit": self.config.checkpoints_total_limit,
        }
        gui_config_path = os.path.join(checkpoint_dir, "gui_config.json")
        with open(gui_config_path, "w", encoding="utf-8") as f:
            json.dump(gui_config, f, indent=2, ensure_ascii=False)
    
    def save_final_model(self):
        """保存最终模型"""
        self.accelerator.wait_for_everyone()
        
        if not self.accelerator.is_main_process:
            return
        
        print("\n保存最终 ZImage 模型")
        print("=" * 60)
        
        output_dir = self.config.output_dir or os.path.join(self.script_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        actual_steps = self.current_step if self.current_step > 0 else self.adjusted_num_train_steps
        equivalent_steps = actual_steps * self.accelerator.num_processes * self.config.gradient_accumulation_steps
        
        unwrapped_transformer = self.accelerator.unwrap_model(self.transformer)
        
        if self.config.full_training:
            # 全量训练：保存完整模型
            model_dir = os.path.join(output_dir, f"zimage_full_{equivalent_steps}steps")
            os.makedirs(model_dir, exist_ok=True)
            
            transformer_dir = os.path.join(model_dir, "transformer")
            unwrapped_transformer.save_pretrained(transformer_dir, safe_serialization=True)
            
            metadata = {
                "model_type": "zimage",
                "base_model": self.config.model_id,
                "training_method": "full_finetune",
                "trained_steps": actual_steps,
                "equivalent_single_gpu_steps": equivalent_steps,
                "resolution": self.config.resolution,
                "learning_rate": self.config.learning_rate,
                "timestep_type": self.config.timestep_type,
                "stopped_early": self.should_stop,
                "num_gpus": self.accelerator.num_processes,
            }
            with open(os.path.join(model_dir, "training_metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
            
            self._save_gui_config(model_dir)
            
            print(f"\n✓ 最终全量模型已保存到: {model_dir}/")
            print(f"  - transformer/ (完整 transformer 权重)")
        else:
            # LoKr 模式：保存 LoKr 权重
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
                "base_model": self.config.model_id,
                "ss_network_module": "lycoris.kohya",
                "ss_network_dim": str(self.config.lora_dim),
                "ss_network_alpha": str(self.config.lora_alpha),
                "num_gpus": str(self.accelerator.num_processes),
            }
            
            model_filename = f"zimage_lokr_{equivalent_steps}steps.safetensors"
            config_filename = f"lokr_config_{equivalent_steps}steps.json"
            
            save_file(lokr_state_dict, os.path.join(output_dir, model_filename), metadata=metadata)
            
            lokr_config = {
                "lora_dim": self.config.lora_dim,
                "lora_alpha": self.config.lora_alpha,
                "lokr_factor": self.config.lokr_factor,
                "full_matrix": self.config.full_matrix,
                "decompose_both": self.config.decompose_both,
                "num_modules": len(self.lokr_modules),
                "base_model": self.config.model_id,
                "resolution": self.config.resolution,
                "format": "comfyui_lokr",
                "trained_steps": actual_steps,
                "equivalent_single_gpu_steps": equivalent_steps,
                "stopped_early": self.should_stop,
                "num_gpus": self.accelerator.num_processes,
            }
            with open(os.path.join(output_dir, config_filename), "w") as f:
                json.dump(lokr_config, f, indent=2)
            
            print(f"\n✓ 最终 LoKr 模型已保存到: {output_dir}/")
            print(f"  - {model_filename}")
            print(f"  - {config_filename}")
        
        # 保存检查点
        self.save_checkpoint(actual_steps)
        
        if self.should_stop:
            print(f"\n⏹️ 训练在第 {actual_steps} 步停止")
        else:
            print(f"\n✅ 训练完成!")
        
        print(f"  数据集: {len(self.dataset)} 张图片")
        print(f"  使用 GPU 数量: {self.accelerator.num_processes}")
        if self.config.full_training:
            print(f"\n模型格式: Full (完整模型)")
        else:
            print(f"\n模型格式: LyCORIS LoKr")
        print(f"可在 ComfyUI 中直接加载")
    
    def pre_training_hook(self):
        """训练前钩子"""
        if self.accelerator.is_main_process:
            if self.config.full_training:
                print(f"\nZImage 全量训练 (qfloat8)")
            else:
                print(f"\nZImage LoKr 训练 (qfloat8)")
            print(f"时间步采样: {self.config.timestep_type}")
            if self.config.timestep_type == "sigmoid":
                print(f"  - sigmoid_scale: {self.config.sigmoid_scale}")
            elif self.config.timestep_type == "shift":
                print(f"  - shift_scale: {self.config.shift_scale}")
            elif self.config.timestep_type == "lognorm_blend":
                print(f"  - lognorm_alpha: {self.config.lognorm_alpha}")
            print(f"Loss weighting: {self.config.loss_weighting_scheme}")
            print(f"Prompt dropout: {self.config.prompt_dropout_prob}")
            print(f"Noise offset: {self.config.noise_offset}")


# ============================================================
# 主函数
# ============================================================
def main():
    import argparse
    
    fix_windows_encoding()
    
    parser = argparse.ArgumentParser(description="ZImage 训练器 (支持 LoKr 和全量训练，使用 qfloat8)")
    
    # 必选参数
    parser.add_argument("--model_id", type=str, required=True, help="模型路径")
    parser.add_argument("--image_folder", type=str, required=True, help="训练图片文件夹")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    
    # 训练模式
    parser.add_argument("--training_mode", type=str, default="lokr", choices=["lokr", "full"], 
                        help="训练模式: lokr 或 full")
    
    # 训练参数
    parser.add_argument("--num_train_steps", type=int, default=5000, help="训练步数")
    parser.add_argument("--checkpoint_every_n_steps", type=int, default=500, help="检查点保存间隔")
    parser.add_argument("--checkpoints_total_limit", type=int, default=3, help="检查点数量限制")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--resolution", type=int, default=1024, help="图片分辨率")
    
    # 时间步采样
    parser.add_argument("--timestep_type", type=str, default="sigmoid", 
                        choices=TIMESTEP_TYPES, help="时间步采样类型")
    parser.add_argument("--sigmoid_scale", type=float, default=1.0, help="Sigmoid 分布集中程度")
    parser.add_argument("--shift_scale", type=float, default=3.0, help="Shift 采样偏移程度")
    parser.add_argument("--lognorm_alpha", type=float, default=0.75, help="LogNorm 混合比例")
    
    # Loss Weighting
    parser.add_argument("--loss_weighting_scheme", type=str, default="none",
                        choices=LOSS_WEIGHTING_TYPES, help="Loss 权重方案: none/sigma_sqrt/cosmap")
    
    # Caption
    parser.add_argument("--use_caption", action="store_true", default=True, help="使用 caption")
    parser.add_argument("--default_caption", type=str, default="", help="默认 caption")
    
    # 正则化
    parser.add_argument("--prompt_dropout_prob", type=float, default=0.1, help="Prompt dropout 概率 (0-1)")
    parser.add_argument("--noise_offset", type=float, default=0.0, help="Noise offset (0-0.1)")
    
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="从检查点恢复训练")
    
    args = parser.parse_args()
    
    full_training = args.training_mode == "full"
    
    config = ZImageConfig(
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
        loss_weighting_scheme=args.loss_weighting_scheme,
        full_training=full_training,
        use_caption=args.use_caption,
        default_caption=args.default_caption,
        prompt_dropout_prob=args.prompt_dropout_prob,
        noise_offset=args.noise_offset,
        seed=args.seed,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    
    trainer = ZImageTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
