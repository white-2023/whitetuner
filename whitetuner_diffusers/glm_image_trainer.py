"""
GLM-Image 训练器

支持两种训练模式:
- T2I (Text-to-Image): 文生图训练
- Edit (Image-to-Image): 图像编辑训练，支持多图条件输入

特性:
- 全量训练 + Block Swap
- 支持多种时间步采样方法
- Edit 模式支持 KV Cache 多图条件
"""

import os
import gc
import math
import json
import re
from typing import List, Optional, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from PIL.ImageOps import exif_transpose
from tqdm import tqdm

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, GlmImageTransformer2DModel
from transformers import T5EncoderModel, ByT5Tokenizer
from safetensors.torch import save_file, load_file

from base_trainer import (
    BaseTrainer,
    BaseTrainerConfig,
    fix_windows_encoding,
    sample_timesteps,
    TIMESTEP_TYPES,
    TimestepType,
)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'hhytuner_singlefile'))
from wan_modules.offloading_utils import (
    ModelOffloader,
    weighs_to_device,
    _move_non_linear_weight_to_device,
    synchronize_device,
    clean_memory_on_device,
)

# GlmImageKVCache 用于 Edit 模式的多图条件
try:
    from diffusers.models.transformers.transformer_glm_image import GlmImageKVCache
    HAS_KV_CACHE = True
except ImportError:
    GlmImageKVCache = None
    HAS_KV_CACHE = False

# Vision Language Encoder 用于 Edit 模式生成 prior tokens
try:
    from transformers import GlmImageForConditionalGeneration, GlmImageProcessor
    HAS_GLM_IMAGE_VLM = True
except ImportError:
    HAS_GLM_IMAGE_VLM = False
    GlmImageForConditionalGeneration = None
    GlmImageProcessor = None

# 任务类型
TASK_TYPES = ["t2i", "edit"]
TaskType = str


class GlmImageConfig(BaseTrainerConfig):
    
    def __init__(
        self,
        output_dir: str,
        model_id: str = "zai-org/GLM-Image",
        task_type: TaskType = "t2i",  # "t2i" 或 "edit"
        # T2I 模式参数
        image_folder: str = None,
        use_caption: bool = True,
        caption_ext: str = ".txt",
        default_caption: str = "",
        # Edit 模式参数
        condition_folder: str = None,
        target_folder: str = None,
        prompt: str = "",
        prior_dropout_prob: float = 0.1,
        # 通用参数
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
        blocks_to_swap: int = 0,
        use_pinned_memory: bool = True,
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
        
        self.task_type = task_type
        
        # T2I 模式参数
        self.image_folder = image_folder
        self.use_caption = use_caption
        self.caption_ext = caption_ext
        self.default_caption = default_caption
        
        # Edit 模式参数
        self.condition_folder = condition_folder
        self.target_folder = target_folder
        self.prompt = prompt
        self.prior_dropout_prob = prior_dropout_prob
        
        # 通用参数
        self.timestep_type = timestep_type
        self.sigmoid_scale = sigmoid_scale
        self.shift_scale = shift_scale
        self.lognorm_alpha = lognorm_alpha
        self.blocks_to_swap = blocks_to_swap
        self.use_pinned_memory = use_pinned_memory
        self.noise_offset = noise_offset
        
        if cache_dir is None:
            if task_type == "t2i" and image_folder:
                self.cache_dir = os.path.join(image_folder, ".glm_t2i_cache")
            elif task_type == "edit" and target_folder:
                self.cache_dir = os.path.join(target_folder, ".glm_edit_cache")
            else:
                self.cache_dir = os.path.join("cache", "glm_image", "default")
        else:
            self.cache_dir = cache_dir


def verify_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False


class GlmImageDataset(Dataset):
    
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
            print(f"[OK] 找到 {len(self.samples)} 张有效图片")
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


class GlmEditDataset(Dataset):
    """GLM-Image 编辑数据集 - 支持多图条件输入
    
    数据集结构支持三种模式：
    1. 单图条件：condition/image1.png -> target/image1.png
    2. 多图条件：condition/image1_1.png, condition/image1_2.png -> target/image1.png
    3. 子文件夹：condition/image1/ 下所有图像 -> target/image1.png
    """
    
    def __init__(
        self,
        condition_folder: str,
        target_folder: str,
        resolution: int = 1024,
        dtype: torch.dtype = torch.bfloat16,
        verbose: bool = True,
    ):
        self.condition_folder = condition_folder
        self.target_folder = target_folder
        self.resolution = resolution
        self.dtype = dtype
        
        # 查找匹配的图像对（支持多图条件）
        self.samples = self._find_samples()
        
        if verbose:
            total_cond = sum(len(s['condition_paths']) for s in self.samples)
            print(f"找到 {len(self.samples)} 个训练样本，共 {total_cond} 张条件图像")
        
        self.transform = transforms.Compose([transforms.ToTensor()])
    
    def _find_samples(self) -> List[Dict]:
        """查找目标图像和对应的条件图像（支持多图）"""
        samples = []
        image_exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
        
        for f in os.listdir(self.target_folder):
            if not f.lower().endswith(image_exts):
                continue
            
            target_path = os.path.join(self.target_folder, f)
            base_name = os.path.splitext(f)[0]
            
            condition_paths = []
            
            # 方式1: 同名条件图像
            for ext in image_exts:
                cond_path = os.path.join(self.condition_folder, base_name + ext)
                if os.path.exists(cond_path):
                    condition_paths.append(cond_path)
                    break
            
            # 方式2: 带数字后缀的多图条件
            for ext in image_exts:
                idx = 1
                while True:
                    cond_path = os.path.join(self.condition_folder, f"{base_name}_{idx}{ext}")
                    if os.path.exists(cond_path):
                        condition_paths.append(cond_path)
                        idx += 1
                    else:
                        break
            
            # 方式3: 同名文件夹
            cond_dir = os.path.join(self.condition_folder, base_name)
            if os.path.isdir(cond_dir):
                for cond_f in sorted(os.listdir(cond_dir)):
                    if cond_f.lower().endswith(image_exts):
                        condition_paths.append(os.path.join(cond_dir, cond_f))
            
            # 去重并排序
            condition_paths = sorted(list(set(condition_paths)))
            
            if len(condition_paths) > 0:
                samples.append({
                    'target_path': target_path,
                    'condition_paths': condition_paths,
                })
        
        return samples
    
    def _load_and_resize_image(self, path: str, target_size: Tuple[int, int] = None) -> Image.Image:
        """加载并调整图像大小"""
        image = Image.open(path)
        image = exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        if target_size is not None:
            image = image.resize(target_size, Image.LANCZOS)
        else:
            w, h = image.size
            scale = self.resolution / min(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # 确保是 32 的倍数
            new_w = (new_w // 32) * 32
            new_h = (new_h // 32) * 32
            
            image = image.resize((new_w, new_h), Image.LANCZOS)
        
        return image
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载目标图像
        target_image = self._load_and_resize_image(sample['target_path'])
        target_size = target_image.size
        
        # 加载所有条件图像（调整到相同大小）
        condition_images = []
        for cond_path in sample['condition_paths']:
            cond_image = self._load_and_resize_image(cond_path, target_size)
            condition_images.append(self.transform(cond_image))
        
        return {
            'conditions': condition_images,  # List of tensors
            'target': self.transform(target_image),
            'sample_idx': idx,
            'num_conditions': len(condition_images),
        }


class GlmImageTrainer(BaseTrainer):
    
    def __init__(self, config: GlmImageConfig):
        super().__init__(config)
        self.config: GlmImageConfig = config
        
        # Edit 模式需要检查依赖
        if config.task_type == "edit":
            if not HAS_KV_CACHE:
                raise RuntimeError(
                    "GLM-Image Edit 模式需要 diffusers 支持 GlmImageKVCache\n"
                    "请更新: pip install git+https://github.com/huggingface/diffusers.git"
                )
            if not HAS_GLM_IMAGE_VLM:
                raise RuntimeError(
                    "GLM-Image Edit 模式需要安装支持 GlmImageForConditionalGeneration 的 transformers\n"
                    "请安装: pip install git+https://github.com/huggingface/transformers.git"
                )
        
        self.text_embeds_cache = {}
        self.latent_cache = {}
        self.prior_token_cache = {}  # Edit 模式用
        
        self.vae_scale_factor = 8
        self.patch_size = 2
        self.latents_mean = None
        self.latents_std = None
        
        # Edit 模式的 VLM
        self.vlm = None
        self.vlm_processor = None
    
    def _check_stop(self) -> bool:
        return self.should_stop
    
    def cleanup_offloader(self):
        if hasattr(self, 'offloader') and self.offloader is not None:
            self.offloader.shutdown()
            self.offloader = None
    
    def create_dataset(self):
        task_name = "T2I" if self.config.task_type == "t2i" else "Edit"
        if self.accelerator.is_main_process:
            print(f"\n创建 GLM-Image {task_name} 数据集")
            print("=" * 60)
        
        if self.config.task_type == "t2i":
            self.dataset = GlmImageDataset(
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
        else:
            # Edit 模式
            self.dataset = GlmEditDataset(
                condition_folder=self.config.condition_folder,
                target_folder=self.config.target_folder,
                resolution=self.config.resolution,
                dtype=self.config.dtype,
                verbose=self.accelerator.is_main_process,
            )
            
            def collate_fn(batch):
                # Edit 模式 batch_size 必须为 1
                conditions_list = [item['conditions'] for item in batch]
                targets = torch.stack([item['target'] for item in batch])
                sample_indices = [item['sample_idx'] for item in batch]
                num_conditions = [item['num_conditions'] for item in batch]
                
                return {
                    'conditions_list': conditions_list,
                    'target': targets,
                    'sample_indices': sample_indices,
                    'num_conditions': num_conditions,
                }
        
        # Edit 模式强制 batch_size=1
        batch_size = 1 if self.config.task_type == "edit" else self.config.batch_size
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            collate_fn=collate_fn,
        )
        
        if self.accelerator.is_main_process:
            print(f"[OK] DataLoader 创建完成，batch_size={batch_size}")
    
    def load_models(self):
        if self._check_stop():
            return
        
        self._load_text_encoder()
        if self._check_stop():
            return
        
        # Edit 模式需要加载 VLM
        if self.config.task_type == "edit":
            self._load_vlm()
            if self._check_stop():
                return
        
        self._load_vae()
        if self._check_stop():
            return
        
        # 根据模式选择缓存方法
        if self.config.task_type == "t2i":
            self._cache_embeddings_and_latents()
        else:
            self._cache_edit_data()
        
        if self._check_stop():
            return
        
        self._load_transformer()
        if self._check_stop():
            return
        
        # Edit 模式: 缓存 KV cache (需要 transformer)
        if self.config.task_type == "edit":
            self._cache_kv_caches()
            if self._check_stop():
                return
        
        self._prepare_for_ddp()
    
    def _load_vlm(self):
        """加载 Vision Language Model (Edit 模式用)"""
        if self.accelerator.is_main_process:
            print("\n阶段 1.5: 加载 Vision Language Encoder")
            print("=" * 60)
        
        if self._check_stop():
            return
        
        self.vlm = GlmImageForConditionalGeneration.from_pretrained(
            self.config.model_id,
            subfolder="vision_language_encoder",
            torch_dtype=self.config.dtype,
            low_cpu_mem_usage=True,
        )
        self.vlm.requires_grad_(False)
        self.vlm.eval()
        self.vlm.to(self.accelerator.device)
        
        self.vlm_processor = GlmImageProcessor.from_pretrained(
            self.config.model_id,
            subfolder="processor",
        )
        
        if self.accelerator.is_main_process:
            print("[OK] Vision Language Encoder 加载完成")
    
    def _load_text_encoder(self):
        if self.accelerator.is_main_process:
            print("\n阶段 1: 加载 Text Encoder (T5)")
            print("=" * 60)
        
        if self._check_stop():
            return
        
        self.text_encoder = T5EncoderModel.from_pretrained(
            self.config.model_id,
            subfolder="text_encoder",
            torch_dtype=self.config.dtype,
            low_cpu_mem_usage=True,
        )
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        self.text_encoder.to(self.accelerator.device)
        
        self.tokenizer = ByT5Tokenizer.from_pretrained(
            self.config.model_id,
            subfolder="tokenizer",
        )
        
        if self.accelerator.is_main_process:
            print("[OK] T5 Text Encoder 和 ByT5 Tokenizer 加载完成")
    
    def _load_vae(self):
        if self.accelerator.is_main_process:
            print("\n阶段 2: 加载 VAE")
            print("=" * 60)
        
        if self._check_stop():
            return
        
        self.vae = AutoencoderKL.from_pretrained(
            self.config.model_id,
            subfolder="vae",
            torch_dtype=self.config.dtype,
        )
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae.to(self.accelerator.device)
        
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        
        if hasattr(self.vae.config, 'latents_mean') and self.vae.config.latents_mean is not None:
            self.latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1)
            self.latents_std = torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1)
            if self.accelerator.is_main_process:
                print("[OK] VAE 使用 latents_mean/std 归一化")
        else:
            self.latents_mean = None
            self.latents_std = None
        
        if self.accelerator.is_main_process:
            print(f"[OK] VAE 加载完成 (scale_factor={self.vae_scale_factor})")
    
    def _get_glyph_texts(self, prompt: str) -> List[str]:
        import re
        ocr_texts = (
            re.findall(r"'([^']*)'", prompt)
            + re.findall(r'"([^""]*)[""]', prompt) 
            + re.findall(r'"([^"]*)"', prompt)
            + re.findall(r"「([^「」]*)」", prompt)
        )
        return ocr_texts
    
    def _encode_prompt(self, prompt: str) -> Dict[str, torch.Tensor]:
        glyph_texts = self._get_glyph_texts(prompt)
        
        input_ids_list = self.tokenizer(
            glyph_texts if len(glyph_texts) > 0 else [""],
            max_length=2048,
            truncation=True,
        ).input_ids
        
        input_ids_list = [
            [self.tokenizer.pad_token_id] * ((len(ids) + 1) % 2) + ids 
            for ids in input_ids_list
        ]
        max_length = max(len(ids) for ids in input_ids_list)
        
        attention_mask = torch.tensor([
            [1] * len(ids) + [0] * (max_length - len(ids)) 
            for ids in input_ids_list
        ], device=self.accelerator.device)
        
        input_ids = torch.tensor([
            ids + [self.tokenizer.pad_token_id] * (max_length - len(ids)) 
            for ids in input_ids_list
        ], device=self.accelerator.device)
        
        with torch.no_grad():
            outputs = self.text_encoder(input_ids, attention_mask=attention_mask)
            glyph_embeds = outputs.last_hidden_state[attention_mask.bool()].unsqueeze(0)
        
        return {
            'prompt_embeds': glyph_embeds.cpu(),
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
                }
                self.latent_cache[idx] = cached['latents']
            else:
                samples_to_encode.append(idx)
        
        num_processes = self.accelerator.num_processes
        process_index = self.accelerator.process_index
        
        if self.accelerator.is_main_process:
            if len(self.text_embeds_cache) == 0:
                print(f"没有缓存，现在开始创建缓存 ({len(samples_to_encode)} 个样本)")
            elif len(samples_to_encode) > 0:
                print(f"发现部分缓存，需要处理 {len(samples_to_encode)} 个新样本")
            else:
                print(f"[OK] 所有 {len(self.dataset)} 个样本已有缓存")
        
        if len(samples_to_encode) > 0:
            my_samples = samples_to_encode[process_index::num_processes]
            
            desc = f"缓存 embeddings 和 latents (GPU {process_index})"
            pbar = tqdm(my_samples, desc=desc, disable=not self.accelerator.is_local_main_process)
            
            latents_mean_device = self.latents_mean.to(self.accelerator.device, self.config.dtype) if self.latents_mean is not None else None
            latents_std_device = self.latents_std.to(self.accelerator.device, self.config.dtype) if self.latents_std is not None else None
            
            for idx in pbar:
                if self._check_stop():
                    break
                
                sample = self.dataset[idx]
                caption = sample['caption']
                
                embed_result = self._encode_prompt(caption)
                
                image = sample['image'].unsqueeze(0).to(self.accelerator.device, self.config.dtype)
                image = image * 2.0 - 1.0
                
                with torch.no_grad():
                    latent_dist = self.vae.encode(image).latent_dist
                    latents = latent_dist.sample()
                    
                    if latents_mean_device is not None and latents_std_device is not None:
                        latents = (latents - latents_mean_device) / latents_std_device
                    else:
                        latents = latents * self.vae.config.scaling_factor
                
                self.text_embeds_cache[idx] = {
                    'prompt_embeds': embed_result['prompt_embeds'],
                }
                self.latent_cache[idx] = latents.cpu()
                
                cache_data = {
                    'prompt_embeds': embed_result['prompt_embeds'],
                    'latents': latents.cpu(),
                }
                cache_file = os.path.join(self.config.cache_dir, f"sample_{idx}.pt")
                torch.save(cache_data, cache_file)
            
            pbar.close()
        
        self.accelerator.wait_for_everyone()
        
        for idx in range(len(self.dataset)):
            if idx not in self.text_embeds_cache:
                cache_file = os.path.join(self.config.cache_dir, f"sample_{idx}.pt")
                if os.path.exists(cache_file):
                    cached = torch.load(cache_file, map_location='cpu')
                    self.text_embeds_cache[idx] = {
                        'prompt_embeds': cached['prompt_embeds'],
                    }
                    self.latent_cache[idx] = cached['latents']
        
        del self.text_encoder
        del self.vae
        gc.collect()
        torch.cuda.empty_cache()
        
        if self.accelerator.is_main_process:
            print("[OK] Text Encoder 和 VAE 已卸载，缓存完成")
    
    @staticmethod
    def _compute_generation_params(image_grid_thw, is_text_to_image: bool):
        """计算 prior token 生成参数"""
        grid_sizes = []
        grid_hw = []
        
        for i in range(image_grid_thw.shape[0]):
            t, h, w = image_grid_thw[i].tolist()
            grid_sizes.append(int(h * w))
            grid_hw.append((int(h), int(w)))
        
        if not is_text_to_image:
            max_new_tokens = grid_sizes[-1] + 1
            large_image_start_offset = 0
            target_grid_h, target_grid_w = grid_hw[-1]
        else:
            total_tokens = sum(grid_sizes)
            max_new_tokens = total_tokens + 1
            large_image_start_offset = sum(grid_sizes[1:])
            target_grid_h, target_grid_w = grid_hw[0]
        return max_new_tokens, large_image_start_offset, target_grid_h, target_grid_w
    
    @staticmethod
    def _extract_large_image_tokens(outputs: torch.Tensor, input_length: int, large_image_start_offset: int, large_image_tokens: int) -> torch.Tensor:
        """提取生成的 large image tokens"""
        generated_tokens = outputs[0][input_length:]
        large_image_start = large_image_start_offset
        large_image_end = large_image_start + large_image_tokens
        return generated_tokens[large_image_start:large_image_end]
    
    @staticmethod
    def _upsample_token_ids(token_ids: torch.Tensor, token_h: int, token_w: int) -> torch.Tensor:
        """上采样 prior token ids"""
        token_ids = token_ids.view(1, 1, token_h, token_w)
        token_ids = F.interpolate(token_ids.float(), scale_factor=2, mode="nearest").to(dtype=torch.long)
        token_ids = token_ids.view(1, -1)
        return token_ids
    
    def _generate_prior_tokens(self, image: Image.Image, prompt: str, height: int, width: int):
        """使用 VLM 生成图像的 prior tokens (Edit 模式用)"""
        device = self.accelerator.device
        is_text_to_image = False
        
        content = [{"type": "image", "image": image}]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        
        inputs = self.vlm_processor.apply_chat_template(
            messages,
            tokenize=True,
            target_h=height,
            target_w=width,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
        
        image_grid_thw = inputs.get("image_grid_thw")
        max_new_tokens, large_image_offset, token_h, token_w = self._compute_generation_params(
            image_grid_thw=image_grid_thw, is_text_to_image=is_text_to_image
        )
        
        prior_token_image_ids = None
        prior_token_image_embed = self.vlm.get_image_features(
            inputs["pixel_values"], image_grid_thw[:-1]
        )
        prior_token_image_embed = torch.cat(prior_token_image_embed, dim=0)
        prior_token_image_ids = self.vlm.get_image_tokens(
            prior_token_image_embed, image_grid_thw[:-1]
        )
        
        with torch.no_grad():
            outputs = self.vlm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                use_cache=True,
            )
        
        prior_token_ids_d32 = self._extract_large_image_tokens(
            outputs, inputs["input_ids"].shape[-1], large_image_offset, token_h * token_w
        )
        prior_token_ids = self._upsample_token_ids(prior_token_ids_d32, token_h, token_w)
        
        return prior_token_ids, prior_token_image_ids
    
    def _get_edit_cache_dir(self) -> str:
        """获取 Edit 模式缓存目录"""
        base_folder = self.config.target_folder
        cache_dir = os.path.join(base_folder, ".glm_edit_cache")
        return cache_dir
    
    def _cache_edit_data(self):
        """缓存 Edit 模式的 prior tokens, text embeddings 和 latents (支持磁盘缓存)"""
        if self.accelerator.is_main_process:
            print("\n阶段 3: 缓存 Prior Tokens, Embeddings 和 Latents (Edit 模式)")
            print("=" * 60)
            vlm_device = next(self.vlm.parameters()).device
            vlm_dtype = next(self.vlm.parameters()).dtype
            print(f"[INFO] VLM 设备: {vlm_device}, 精度: {vlm_dtype}")
            resolution = self.config.resolution
            tokens_per_sample = (resolution // 32) ** 2
            print(f"[INFO] 分辨率 {resolution}x{resolution} 需要生成 {tokens_per_sample} 个 prior tokens/样本 (自回归生成，较慢)")
        
        if self._check_stop():
            return
        
        device = self.accelerator.device
        prompt = self.config.prompt
        
        # 缓存目录
        cache_dir = self._get_edit_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        
        # 检查哪些样本需要重新计算
        samples_to_compute = []
        for idx in range(len(self.dataset)):
            cache_file = os.path.join(cache_dir, f"edit_cache_{idx}.pt")
            if os.path.exists(cache_file):
                try:
                    cached = torch.load(cache_file, map_location='cpu')
                    self.prior_token_cache[idx] = cached['prior_token_cache']
                    self.latent_cache[idx] = cached['latent']
                    self.text_embeds_cache[idx] = cached['text_embeds']
                    continue
                except Exception:
                    pass
            samples_to_compute.append(idx)
        
        if self.accelerator.is_main_process:
            loaded_count = len(self.dataset) - len(samples_to_compute)
            if loaded_count > 0:
                print(f"[OK] 从磁盘加载 {loaded_count} 个缓存")
            if len(samples_to_compute) > 0:
                print(f"[INFO] 需要计算 {len(samples_to_compute)} 个新缓存")
        
        if len(samples_to_compute) > 0:
            # 编码 prompt (所有样本共用)
            embed_result = self._encode_prompt(prompt)
            prompt_embeds = embed_result['prompt_embeds']
            
            latents_mean_device = self.latents_mean.to(device, self.config.dtype) if self.latents_mean is not None else None
            latents_std_device = self.latents_std.to(device, self.config.dtype) if self.latents_std is not None else None
            
            for idx in tqdm(samples_to_compute, desc="缓存数据", disable=not self.accelerator.is_main_process):
                if self._check_stop():
                    break
                
                sample = self.dataset.samples[idx]
                target_path = sample['target_path']
                condition_paths = sample['condition_paths']
                
                # 加载目标图像
                target_image = self.dataset._load_and_resize_image(target_path)
                target_tensor = self.dataset.transform(target_image)
                h, w = target_tensor.shape[1], target_tensor.shape[2]
                pixel_h, pixel_w = h, w
                target_size = target_image.size
                
                # 为每张条件图像生成 prior tokens 和编码 latent
                condition_data_list = []
                for cond_path in condition_paths:
                    cond_image = Image.open(cond_path).convert("RGB")
                    cond_image = exif_transpose(cond_image)
                    cond_image_resized = cond_image.resize(target_size, Image.LANCZOS)
                    
                    with torch.no_grad():
                        prior_token_ids, prior_token_image_ids = self._generate_prior_tokens(
                            cond_image_resized, prompt, pixel_h, pixel_w
                        )
                    
                    cond_tensor = self.dataset.transform(cond_image_resized).unsqueeze(0).to(device, self.config.dtype)
                    cond_normalized = cond_tensor * 2 - 1
                    with torch.no_grad():
                        cond_latent = self.vae.encode(cond_normalized).latent_dist.sample()
                        
                        if latents_mean_device is not None and latents_std_device is not None:
                            cond_latent = (cond_latent - latents_mean_device) / latents_std_device
                        else:
                            cond_latent = cond_latent * self.vae.config.scaling_factor
                    
                    condition_data_list.append({
                        'prior_token_ids': prior_token_ids.cpu(),
                        'prior_token_image_ids': prior_token_image_ids.cpu() if prior_token_image_ids is not None else None,
                        'latent': cond_latent.cpu(),
                    })
                
                self.prior_token_cache[idx] = {
                    'conditions': condition_data_list,
                    'num_conditions': len(condition_data_list),
                }
                
                # 编码目标图像到 latent
                target_tensor = target_tensor.unsqueeze(0).to(device, self.config.dtype)
                target_normalized = target_tensor * 2 - 1
                with torch.no_grad():
                    target_latent = self.vae.encode(target_normalized).latent_dist.sample()
                    
                    if latents_mean_device is not None and latents_std_device is not None:
                        target_latent = (target_latent - latents_mean_device) / latents_std_device
                    else:
                        target_latent = target_latent * self.vae.config.scaling_factor
                
                self.latent_cache[idx] = target_latent.cpu()
                self.text_embeds_cache[idx] = {'prompt_embeds': prompt_embeds}
                
                # 保存到磁盘
                cache_file = os.path.join(cache_dir, f"edit_cache_{idx}.pt")
                torch.save({
                    'prior_token_cache': self.prior_token_cache[idx],
                    'latent': self.latent_cache[idx],
                    'text_embeds': self.text_embeds_cache[idx],
                }, cache_file)
        
        # 卸载 VLM (不再需要)
        del self.vlm
        del self.vlm_processor
        self.vlm = None
        self.vlm_processor = None
        gc.collect()
        torch.cuda.empty_cache()
        
        if self.accelerator.is_main_process:
            print(f"[OK] 缓存完成 (缓存目录: {cache_dir})")
            print("[OK] VLM 已卸载")
            print("[INFO] KV Cache 将在 Transformer 加载后缓存")
    
    def _get_kv_cache_dir(self) -> str:
        """获取 KV Cache 缓存目录"""
        # 使用目标图像文件夹作为基础
        base_folder = self.config.target_folder
        cache_dir = os.path.join(base_folder, ".glm_kv_cache")
        return cache_dir
    
    def _cache_kv_caches(self):
        """缓存 Edit 模式的 KV Cache，使 Gradient Checkpointing 可用"""
        if self.accelerator.is_main_process:
            print("\n阶段 4.5: 缓存 KV Cache (Edit 模式)")
            print("=" * 60)
        
        if self._check_stop():
            return
        
        device = self.accelerator.device
        num_layers = self.transformer.config.num_layers
        
        # 检查磁盘缓存
        cache_dir = self._get_kv_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        
        self.kv_cache_store = {}
        samples_to_compute = []
        
        # 检查哪些样本需要重新计算
        for idx in range(len(self.dataset)):
            cache_file = os.path.join(cache_dir, f"kv_cache_{idx}.pt")
            condition_data = self.prior_token_cache[idx]
            
            if condition_data['num_conditions'] == 0:
                self.kv_cache_store[idx] = None
                continue
            
            if os.path.exists(cache_file):
                # 从磁盘加载
                try:
                    kv_data = torch.load(cache_file, map_location='cpu')
                    # 验证层数是否匹配
                    if len(kv_data) == num_layers:
                        self.kv_cache_store[idx] = kv_data
                        continue
                except Exception:
                    pass
            
            samples_to_compute.append(idx)
        
        if self.accelerator.is_main_process:
            loaded_count = len(self.dataset) - len(samples_to_compute)
            if loaded_count > 0:
                print(f"[OK] 从磁盘加载 {loaded_count} 个 KV Cache")
            if len(samples_to_compute) > 0:
                print(f"[INFO] 需要计算 {len(samples_to_compute)} 个新的 KV Cache")
        
        if len(samples_to_compute) > 0:
            # 临时移动 transformer 到 GPU
            if self.config.blocks_to_swap > 0:
                self.transformer.to(device)
            
            self.transformer.eval()
            
            for idx in tqdm(samples_to_compute, desc="计算 KV Cache", disable=not self.accelerator.is_main_process):
                if self._check_stop():
                    break
                
                condition_data = self.prior_token_cache[idx]
                prompt_embeds = self.text_embeds_cache[idx]['prompt_embeds'].to(device, self.config.dtype)
                target_latent = self.latent_cache[idx].to(device, self.config.dtype)
                
                latent_height = target_latent.shape[2]
                latent_width = target_latent.shape[3]
                pixel_height = latent_height * self.vae_scale_factor
                pixel_width = latent_width * self.vae_scale_factor
                target_size = torch.tensor([[pixel_height, pixel_width]], device=device, dtype=self.config.dtype)
                crops_coords = torch.zeros(1, 2, device=device, dtype=self.config.dtype)
                
                # 创建 KV cache 并运行所有条件图像的 forward
                kv_caches = GlmImageKVCache(num_layers=num_layers)
                kv_caches.set_mode("write")
                
                with torch.no_grad():
                    for cond_data in condition_data['conditions']:
                        cond_latent = cond_data['latent'].to(device, self.config.dtype)
                        cond_prior_token_ids = cond_data['prior_token_ids'].to(device)
                        cond_prior_drop = torch.zeros_like(cond_prior_token_ids, dtype=torch.bool)
                        
                        _ = self.transformer(
                            hidden_states=cond_latent,
                            encoder_hidden_states=torch.zeros_like(prompt_embeds)[:, :0, :],
                            prior_token_id=cond_prior_token_ids,
                            prior_token_drop=cond_prior_drop,
                            timestep=torch.zeros(1, device=device),
                            target_size=target_size,
                            crop_coords=crops_coords,
                            kv_caches=kv_caches,
                        )
                
                # 提取 KV cache
                kv_data = []
                for layer_idx in range(num_layers):
                    layer_cache = kv_caches[layer_idx]
                    kv_data.append({
                        'k': layer_cache.k_cache.cpu() if layer_cache.k_cache is not None else None,
                        'v': layer_cache.v_cache.cpu() if layer_cache.v_cache is not None else None,
                    })
                
                self.kv_cache_store[idx] = kv_data
                
                # 保存到磁盘
                cache_file = os.path.join(cache_dir, f"kv_cache_{idx}.pt")
                torch.save(kv_data, cache_file)
                
                kv_caches.clear()
                del kv_caches
            
            # 恢复 transformer 设备布局
            if self.config.blocks_to_swap > 0:
                self.transformer.to('cpu')
                torch.cuda.empty_cache()
        
        # 卸载不再需要的 text_encoder 和 vae
        if hasattr(self, 'text_encoder') and self.text_encoder is not None:
            del self.text_encoder
            self.text_encoder = None
        if hasattr(self, 'vae') and self.vae is not None:
            del self.vae
            self.vae = None
        gc.collect()
        torch.cuda.empty_cache()
        
        self.transformer.train()
        
        if self.accelerator.is_main_process:
            print(f"[OK] KV Cache 缓存完成 (缓存目录: {cache_dir})")
            print("[OK] TE/VAE 已卸载")
    
    def _load_kv_cache_for_sample(self, idx: int, device: torch.device) -> GlmImageKVCache:
        """从缓存加载 KV Cache 并设置为 read 模式"""
        kv_data = self.kv_cache_store.get(idx)
        if kv_data is None:
            return None
        
        num_layers = len(kv_data)
        kv_caches = GlmImageKVCache(num_layers=num_layers)
        
        for layer_idx, layer_data in enumerate(kv_data):
            if layer_data['k'] is not None:
                kv_caches[layer_idx].k_cache = layer_data['k'].to(device)
                kv_caches[layer_idx].v_cache = layer_data['v'].to(device)
        
        kv_caches.set_mode("read")
        return kv_caches
    
    def _load_transformer(self):
        if self.accelerator.is_main_process:
            print("\n阶段 4: 加载 Transformer")
            print("=" * 60)
        
        if self._check_stop():
            return
        
        loading_device = self.accelerator.device if self.config.blocks_to_swap == 0 else None
        
        if self.accelerator.is_main_process:
            print(f">>> 加载 GlmImageTransformer2DModel...")
        
        self.transformer = GlmImageTransformer2DModel.from_pretrained(
            self.config.model_id,
            subfolder="transformer",
            torch_dtype=self.config.dtype,
        )
        
        if loading_device is not None:
            self.transformer.to(loading_device)
        
        self.patch_size = self.transformer.config.patch_size
        
        self.transformer.requires_grad_(True)
        if self.accelerator.is_main_process:
            print(">>> Full Training 模式: transformer 参数可训练")
            trainable_count = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
            total_count = sum(p.numel() for p in self.transformer.parameters())
            print(f"    可训练参数: {trainable_count:,} / {total_count:,}")
        
        if self.config.blocks_to_swap > 0:
            if self.accelerator.is_main_process:
                print(f">>> 启用 block swap: {self.config.blocks_to_swap} blocks")
            
            num_blocks = len(self.transformer.transformer_blocks)
            if self.config.blocks_to_swap > num_blocks:
                raise ValueError(f"blocks_to_swap ({self.config.blocks_to_swap}) 超过总 block 数量 ({num_blocks})")
            
            self._enable_block_swap()
            
            if self.accelerator.is_main_process:
                blocks_on_gpu = num_blocks - self.config.blocks_to_swap
                print(f">>> Block swap 已启用: {blocks_on_gpu} blocks 在 GPU, {self.config.blocks_to_swap} blocks 在 CPU")
        
        # 启用 Gradient Checkpointing
        # Edit 模式使用预缓存的 KV Cache，每次训练步骤加载相同的缓存，重计算时状态一致
        if hasattr(self.transformer, 'enable_gradient_checkpointing'):
            self.transformer.enable_gradient_checkpointing()
            self.activation_cpu_offloading = self.config.blocks_to_swap > 0
            if self.accelerator.is_main_process:
                if self.activation_cpu_offloading:
                    print("[OK] 启用 Gradient Checkpointing (with Activation CPU Offloading)")
                else:
                    print("[OK] 启用 Gradient Checkpointing")
        
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.config.model_id,
            subfolder="scheduler",
        )
        
        if self.accelerator.is_main_process:
            print("[OK] Transformer 和 Scheduler 加载完成")
    
    def _enable_block_swap(self):
        device = self.accelerator.device
        blocks_to_swap = self.config.blocks_to_swap
        num_blocks = len(self.transformer.transformer_blocks)
        blocks = list(self.transformer.transformer_blocks)
        
        # 创建 ModelOffloader (与 Qwen Image 相同的实现)
        self.offloader = ModelOffloader(
            "glm-image-block",
            blocks,
            num_blocks,
            blocks_to_swap,
            supports_backward=True,  # 训练需要反向传播
            device=device,
            use_pinned_memory=self.config.use_pinned_memory,
        )
        
        # 准备 block 设备位置
        self.offloader.prepare_block_devices_before_forward(blocks)
        
        # 非 transformer_blocks 的模块移到 GPU
        for name, module in self.transformer.named_children():
            if name != "transformer_blocks":
                module.to(device)
        
        # 保存属性供 forward 使用
        self.transformer._blocks_to_swap = blocks_to_swap
        self.transformer._num_blocks = num_blocks
        self.transformer._device = device
        self.transformer._offloader = self.offloader
        self.transformer._original_forward = self.transformer.forward
        self.transformer._trainer = self
        
        def forward_with_swap(
            hidden_states,
            encoder_hidden_states,
            prior_token_id,
            prior_token_drop,
            timestep,
            target_size,
            crop_coords,
            **kwargs
        ):
            return self._forward_with_block_swap(
                hidden_states,
                encoder_hidden_states,
                prior_token_id,
                prior_token_drop,
                timestep,
                target_size,
                crop_coords,
                **kwargs
            )
        
        self.transformer.forward = forward_with_swap
    
    def _forward_with_block_swap(
        self,
        hidden_states,
        encoder_hidden_states,
        prior_token_id,
        prior_token_drop,
        timestep,
        target_size,
        crop_coords,
        **kwargs
    ):
        transformer = self.transformer
        device = transformer._device
        offloader = transformer._offloader
        blocks = list(transformer.transformer_blocks)
        
        batch_size, num_channels, height, width = hidden_states.shape
        
        image_rotary_emb = transformer.rope(hidden_states)
        image_rotary_emb = (
            image_rotary_emb[0].to(device, dtype=hidden_states.dtype),
            image_rotary_emb[1].to(device, dtype=hidden_states.dtype),
        )
        
        p = transformer.config.patch_size
        post_patch_height = height // p
        post_patch_width = width // p
        
        hidden_states = transformer.image_projector(hidden_states)
        encoder_hidden_states = transformer.glyph_projector(encoder_hidden_states)
        prior_embedding = transformer.prior_token_embedding(prior_token_id)
        prior_embedding[prior_token_drop] *= 0.0
        prior_hidden_states = transformer.prior_projector(prior_embedding)
        
        hidden_states = hidden_states + prior_hidden_states
        
        temb = transformer.time_condition_embed(timestep, target_size, crop_coords, hidden_states.dtype)
        
        for idx, block in enumerate(blocks):
            # 等待 block 准备好（如果需要从 CPU 移到 GPU）
            offloader.wait_for_block(idx)
            
            if torch.is_grad_enabled() and transformer.gradient_checkpointing:
                # 使用 gradient checkpointing，backward hook 会自动处理 block swap
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                )
            
            # 提交下一个 block 的移动（异步）
            offloader.submit_move_blocks_forward(blocks, idx)
        
        hidden_states = transformer.norm_out(hidden_states, temb)
        hidden_states = transformer.proj_out(hidden_states)
        
        hidden_states = hidden_states.reshape(batch_size, post_patch_height, post_patch_width, -1, p, p)
        output = hidden_states.permute(0, 3, 1, 4, 2, 5).flatten(4, 5).flatten(2, 3)
        
        return output
    
    def _gradient_checkpointing_func(self, block, hidden_states, encoder_hidden_states, temb, image_rotary_emb):
        """Gradient checkpointing wrapper with optional CPU offloading support"""
        from wan_modules.offloading_utils import to_device, to_cpu
        
        device = self.accelerator.device
        activation_cpu_offloading = getattr(self, 'activation_cpu_offloading', True)
        
        if activation_cpu_offloading:
            # CPU offloading: 将激活值存储在 CPU 以节省显存
            def custom_forward(hs, enc_hs, t, rot_emb_0, rot_emb_1):
                # 将输入移到 GPU
                cuda_hs = hs.to(device)
                cuda_enc_hs = enc_hs.to(device)
                cuda_t = t.to(device)
                cuda_rot_emb = (rot_emb_0.to(device), rot_emb_1.to(device))
                
                out_hs, out_enc = block(cuda_hs, cuda_enc_hs, cuda_t, cuda_rot_emb)
                
                # 将输出移回 CPU
                return out_hs.cpu(), out_enc.cpu()
            
            out_hs, out_enc = torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states.cpu(),
                encoder_hidden_states.cpu(),
                temb.cpu(),
                image_rotary_emb[0].cpu(),
                image_rotary_emb[1].cpu(),
                use_reentrant=False,
            )
            
            return out_hs.to(device), out_enc.to(device)
        else:
            # 标准 gradient checkpointing
            return torch.utils.checkpoint.checkpoint(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                None,
                None,
                None,
                use_reentrant=False,
            )
    
    def _prepare_for_ddp(self):
        if self.accelerator.is_main_process:
            print("\n阶段 5: 准备分布式训练")
            print("=" * 60)
        
        blocks_to_swap = self.config.blocks_to_swap
        
        if blocks_to_swap > 0:
            if self.accelerator.is_main_process:
                if self.accelerator.num_processes > 1:
                    print("Block Swap + 多卡: 跳过 DDP 包装，使用手动梯度同步")
                print(f"[OK] Transformer 已准备 (Block Swap 模式，手动梯度同步)")
        elif self.accelerator.num_processes > 1:
            self.transformer = self.accelerator.prepare(self.transformer)
            if self.accelerator.is_main_process:
                print(f"[OK] Transformer 已准备 (DDP: True)")
        else:
            if self.accelerator.is_main_process:
                print(f"[OK] Transformer 已准备 (单卡模式，跳过 DDP 包装)")
    
    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        return list(self.transformer.parameters())
    
    def create_optimizer(self, trainable_params: List[torch.nn.Parameter]):
        if self.config.blocks_to_swap > 0:
            import transformers.optimization
            from adafactor_fused import patch_adafactor_fused
            
            if self.accelerator.is_main_process:
                print("Full Training + Block Swap: 使用 Adafactor 优化器 (fused backward mode)")
            
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
            
            import accelerate
            from accelerate.utils import DistributedType
            
            for param_group in optimizer.param_groups:
                for parameter in param_group["params"]:
                    if parameter.requires_grad:
                        def create_grad_hook(p_group_idx):
                            def grad_hook(tensor: torch.Tensor):
                                state = accelerate.PartialState()
                                if state.distributed_type != DistributedType.NO:
                                    tensor.grad = self.accelerator.reduce(tensor.grad, reduction="mean")
                                if self.accelerator.sync_gradients and self.config.max_grad_norm != 0.0:
                                    self.accelerator.clip_grad_norm_(tensor, max_norm=self.config.max_grad_norm)
                                current_group = self._raw_optimizer.param_groups[p_group_idx]
                                self._raw_optimizer.step_param(tensor, current_group)
                                tensor.grad = None
                            return grad_hook
                        parameter.register_post_accumulate_grad_hook(create_grad_hook(0))
            
            if self.accelerator.is_main_process:
                print("[OK] Fused backward pass: 已注册 grad hooks")
            
            return optimizer
        
        return super().create_optimizer(trainable_params)
    
    def train_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        if self.config.task_type == "t2i":
            return self._train_step_t2i(batch)
        else:
            return self._train_step_edit(batch)
    
    def _train_step_t2i(self, batch: Dict[str, Any]) -> torch.Tensor:
        """T2I 模式训练步骤"""
        device = self.accelerator.device
        sample_indices = batch['sample_indices']
        batch_size = len(sample_indices)
        
        # Block swap: 准备 block 设备位置
        if self.config.blocks_to_swap > 0:
            blocks = list(self.transformer.transformer_blocks)
            self.offloader.prepare_block_devices_before_forward(blocks)
        
        latents_list = []
        prompt_embeds_list = []
        
        for idx in sample_indices:
            latents_list.append(self.latent_cache[idx])
            prompt_embeds_list.append(self.text_embeds_cache[idx]['prompt_embeds'].squeeze(0))
        
        latents = torch.cat(latents_list, dim=0).to(device, self.config.dtype)
        
        max_len = max(pe.shape[0] for pe in prompt_embeds_list)
        embed_dim = prompt_embeds_list[0].shape[-1]
        prompt_embeds = torch.zeros(batch_size, max_len, embed_dim, device=device, dtype=self.config.dtype)
        
        for i, pe in enumerate(prompt_embeds_list):
            prompt_embeds[i, :pe.shape[0]] = pe.to(device, self.config.dtype)
        
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
        
        image_seq_len = (latent_height * latent_width) // (self.patch_size ** 2)
        prior_token_ids = torch.randint(
            0, 
            self.transformer.config.prior_vq_quantizer_codebook_size,
            (batch_size, image_seq_len),
            device=device,
        )
        prior_token_drop = torch.ones(batch_size, image_seq_len, device=device, dtype=torch.bool)
        
        pixel_height = latent_height * self.vae_scale_factor
        pixel_width = latent_width * self.vae_scale_factor
        target_size = torch.tensor(
            [[pixel_height, pixel_width]] * batch_size,
            dtype=self.config.dtype,
            device=device
        )
        crop_coords = torch.zeros(batch_size, 2, dtype=self.config.dtype, device=device)
        
        with self.accelerator.autocast():
            model_output = self.transformer(
                hidden_states=noisy_latents,
                encoder_hidden_states=prompt_embeds,
                prior_token_id=prior_token_ids,
                prior_token_drop=prior_token_drop,
                timestep=timesteps,
                target_size=target_size,
                crop_coords=crop_coords,
            )
            
            if hasattr(model_output, 'sample'):
                model_output = model_output.sample
        
        target = noise - latents
        
        per_sample_loss = F.mse_loss(model_output.float(), target.float(), reduction="none")
        per_sample_loss = per_sample_loss.mean(dim=[1, 2, 3])
        
        weighted_loss = per_sample_loss * timestep_weights
        loss = weighted_loss.mean()
        
        return loss
    
    def _train_step_edit(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Edit 模式训练步骤 - 使用预缓存的 KV Cache，支持 Gradient Checkpointing"""
        device = self.accelerator.device
        sample_indices = batch['sample_indices']
        batch_size = len(sample_indices)
        
        # Edit 模式要求 batch_size=1
        if batch_size > 1:
            raise ValueError(
                f"GLM-Image Edit 模式使用 KV Cache 时需要 batch_size=1，当前 batch_size={batch_size}"
            )
        
        # Block swap: 准备 block 设备位置
        if self.config.blocks_to_swap > 0:
            blocks = list(self.transformer.transformer_blocks)
            self.offloader.prepare_block_devices_before_forward(blocks)
        
        idx = sample_indices[0]
        
        # 从缓存加载数据
        target_latent = self.latent_cache[idx].to(device, self.config.dtype)
        condition_data = self.prior_token_cache[idx]
        prompt_embeds = self.text_embeds_cache[idx]['prompt_embeds'].to(device, self.config.dtype)
        
        latent_height = target_latent.shape[2]
        latent_width = target_latent.shape[3]
        
        # 准备输入参数
        pixel_height = latent_height * self.vae_scale_factor
        pixel_width = latent_width * self.vae_scale_factor
        target_size = torch.tensor([[pixel_height, pixel_width]], device=device, dtype=self.config.dtype)
        crops_coords_top_left = torch.zeros(1, 2, device=device, dtype=self.config.dtype)
        
        # Prior dropout: 决定是否使用条件图像
        use_conditions = self.config.prior_dropout_prob == 0 or torch.rand(1).item() >= self.config.prior_dropout_prob
        
        if use_conditions and condition_data['num_conditions'] > 0:
            # 从预缓存加载 KV Cache (read 模式)
            # 预缓存保证每次加载相同内容，Gradient Checkpointing 重计算时状态一致
            kv_caches = self._load_kv_cache_for_sample(idx, device)
            
            # 使用第一张条件图像的 prior_token_ids
            target_prior_token_ids = condition_data['conditions'][0]['prior_token_ids'].to(device)
            target_prior_drop = torch.zeros_like(target_prior_token_ids, dtype=torch.bool)
        else:
            # 无条件训练：不使用 KV cache
            kv_caches = None
            
            # 生成随机 prior tokens
            image_seq_len = (latent_height * latent_width) // (self.patch_size ** 2)
            target_prior_token_ids = torch.randint(
                0, self.transformer.config.prior_vq_quantizer_codebook_size,
                (1, image_seq_len), device=device
            )
            target_prior_drop = torch.ones_like(target_prior_token_ids, dtype=torch.bool)  # drop all
        
        # 添加噪声
        noise = torch.randn_like(target_latent)
        
        if self.config.noise_offset > 0:
            noise = noise + self.config.noise_offset * torch.randn(
                (1, target_latent.shape[1], 1, 1),
                device=device,
                dtype=self.config.dtype
            )
        
        # 采样时间步
        timesteps, timestep_weights = sample_timesteps(
            1,
            num_train_timesteps=self.noise_scheduler.config.num_train_timesteps,
            device=device,
            timestep_type=self.config.timestep_type,
            sigmoid_scale=self.config.sigmoid_scale,
            shift=self.config.shift_scale,
            lognorm_alpha=self.config.lognorm_alpha,
        )
        
        sigmas = timesteps.float() / self.noise_scheduler.config.num_train_timesteps
        sigmas = sigmas.view(-1, 1, 1, 1)
        noisy_latents = (1 - sigmas) * target_latent + sigmas * noise
        
        # Forward pass - 训练目标图像的去噪
        with self.accelerator.autocast():
            model_output = self.transformer(
                hidden_states=noisy_latents,
                encoder_hidden_states=prompt_embeds,
                prior_token_id=target_prior_token_ids,
                prior_token_drop=target_prior_drop,
                timestep=timesteps,
                target_size=target_size,
                crop_coords=crops_coords_top_left,
                kv_caches=kv_caches,
            )
        
        if hasattr(model_output, 'sample'):
            model_output = model_output.sample
        
        # 注意：不要清空 kv_caches，因为 Gradient Checkpointing 重计算时需要它
        # kv_caches 每次通过 _load_kv_cache_for_sample 创建新对象，所以不需要清空
        
        # 计算 loss
        target = noise - target_latent
        
        loss = F.mse_loss(model_output.float(), target.float(), reduction="none")
        loss = loss.mean()
        
        if timestep_weights is not None:
            loss = loss * timestep_weights.item()
        
        return loss
    
    def save_checkpoint(self, step: int):
        output_dir = self.config.output_dir or os.path.join(self.script_dir, "output")
        checkpoint_dir = os.path.join(output_dir, "checkpoints", f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if self.accelerator.is_main_process:
            transformer_save_dir = os.path.join(checkpoint_dir, "transformer")
            os.makedirs(transformer_save_dir, exist_ok=True)
            
            unwrapped_transformer = self.accelerator.unwrap_model(self.transformer)
            if hasattr(unwrapped_transformer, '_original_forward'):
                original_forward = unwrapped_transformer.forward
                unwrapped_transformer.forward = unwrapped_transformer._original_forward
                unwrapped_transformer.save_pretrained(transformer_save_dir)
                unwrapped_transformer.forward = original_forward
            else:
                unwrapped_transformer.save_pretrained(transformer_save_dir)
            
            print(f"  - 全量模型已保存到 {transformer_save_dir}")
        
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
            "training_method": "full",
            "blocks_to_swap": self.config.blocks_to_swap,
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
        
        actual_steps = self.current_step if self.current_step > 0 else self.adjusted_num_train_steps
        equivalent_single_gpu_steps = actual_steps * self.accelerator.num_processes * self.config.gradient_accumulation_steps
        
        task_name = "t2i" if self.config.task_type == "t2i" else "edit"
        transformer_save_dir = os.path.join(output_dir, f"glm_image_{task_name}_full_{equivalent_single_gpu_steps}steps")
        os.makedirs(transformer_save_dir, exist_ok=True)
        
        unwrapped_transformer = self.accelerator.unwrap_model(self.transformer)
        if hasattr(unwrapped_transformer, '_original_forward'):
            original_forward = unwrapped_transformer.forward
            unwrapped_transformer.forward = unwrapped_transformer._original_forward
            unwrapped_transformer.save_pretrained(transformer_save_dir)
            unwrapped_transformer.forward = original_forward
        else:
            unwrapped_transformer.save_pretrained(transformer_save_dir)
        
        training_config = {
            "training_type": "full",
            "task_type": self.config.task_type,
            "base_model": "GLM-Image",
            "resolution": self.config.resolution,
            "trained_steps": actual_steps,
            "equivalent_single_gpu_steps": equivalent_single_gpu_steps,
            "num_gpus": self.accelerator.num_processes,
        }
        with open(os.path.join(transformer_save_dir, "training_config.json"), "w") as f:
            json.dump(training_config, f, indent=2)
        
        print(f"\n[OK] 全量模型已保存到: {transformer_save_dir}/")
        
        self.save_checkpoint(actual_steps)
        
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        print(f"[OK] 最终检查点已保存到: {checkpoint_dir}/checkpoint-{actual_steps}/")
        
        if self.should_stop:
            print(f"\n训练在第 {actual_steps} 步停止")
        else:
            print(f"\n训练完成!")
        
        task_str = "T2I 文生图" if self.config.task_type == "t2i" else "Edit 图像编辑"
        print(f"  任务类型: {task_str}")
        print(f"  数据集: {len(self.dataset)} 个样本")
        print(f"  使用 GPU: {self.accelerator.num_processes}")
        print(f"\n模型格式: Full (完整模型)")
    
    def pre_training_hook(self):
        if self.accelerator.is_main_process:
            task_str = "T2I 文生图" if self.config.task_type == "t2i" else "Edit 图像编辑"
            print(f"\nGLM-Image 全量训练 ({task_str})")
            print(f"时间步采样: {self.config.timestep_type}")
            if self.config.timestep_type == "sigmoid":
                print(f"  - sigmoid_scale: {self.config.sigmoid_scale}")
            elif self.config.timestep_type == "shift":
                print(f"  - shift_scale: {self.config.shift_scale}")
            elif self.config.timestep_type == "lognorm_blend":
                print(f"  - lognorm_alpha: {self.config.lognorm_alpha}")
            print(f"Noise offset: {self.config.noise_offset}")
            if self.config.blocks_to_swap > 0:
                print(f"Block Swap: {self.config.blocks_to_swap} 个 blocks")
            if self.config.task_type == "edit":
                print(f"Prior Dropout: {self.config.prior_dropout_prob * 100:.0f}%")


def main():
    import argparse
    
    fix_windows_encoding()
    
    parser = argparse.ArgumentParser(description="GLM-Image Trainer (T2I / Edit)")
    
    parser.add_argument("--model_id", type=str, required=True, help="Model path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--task_type", type=str, default="t2i", choices=TASK_TYPES, help="Task type: t2i or edit")
    
    # T2I 模式参数
    parser.add_argument("--image_folder", type=str, default=None, help="Image folder path (T2I mode)")
    parser.add_argument("--use_caption", action="store_true", default=True, help="Use captions (T2I mode)")
    parser.add_argument("--caption_ext", type=str, default=".txt", help="Caption file extension")
    parser.add_argument("--default_caption", type=str, default="", help="Default caption")
    
    # Edit 模式参数
    parser.add_argument("--condition_folder", type=str, default=None, help="Condition folder path (Edit mode)")
    parser.add_argument("--target_folder", type=str, default=None, help="Target folder path (Edit mode)")
    parser.add_argument("--prompt", type=str, default="", help="Prompt for edit (Edit mode)")
    parser.add_argument("--prior_dropout_prob", type=float, default=0.1, help="Prior dropout probability for CFG (Edit mode)")
    
    # 通用参数
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
    
    parser.add_argument("--noise_offset", type=float, default=0.0, help="Noise offset")
    
    parser.add_argument("--blocks_to_swap", type=int, default=0, help="Number of transformer blocks to swap to CPU")
    
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume checkpoint")
    
    args = parser.parse_args()
    
    # 验证参数
    if args.task_type == "t2i":
        if not args.image_folder:
            raise ValueError("T2I 模式需要指定 --image_folder")
    else:
        if not args.condition_folder or not args.target_folder:
            raise ValueError("Edit 模式需要指定 --condition_folder 和 --target_folder")
    
    config = GlmImageConfig(
        model_id=args.model_id,
        output_dir=args.output_dir,
        task_type=args.task_type,
        # T2I 模式
        image_folder=args.image_folder,
        use_caption=args.use_caption,
        caption_ext=args.caption_ext,
        default_caption=args.default_caption,
        # Edit 模式
        condition_folder=args.condition_folder,
        target_folder=args.target_folder,
        prompt=args.prompt,
        prior_dropout_prob=args.prior_dropout_prob,
        # 通用
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
        noise_offset=args.noise_offset,
        blocks_to_swap=args.blocks_to_swap,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        seed=args.seed,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    
    trainer = GlmImageTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()


