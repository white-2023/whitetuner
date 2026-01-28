"""
基础训练器模块 - 可复用的训练基础设施

包含：
- 时间步采样方法 (linear/sigmoid/weighted/shift/lognorm_blend)
- 分布式训练辅助函数
- Quanto 修复
- Latent 打包/解包工具
- 基础配置类
- 基础训练器抽象类
"""

import sys
import io
import os
import gc
import time
import json
import shutil
import logging
import signal
import atexit
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Literal

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import LogNormal

from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_constant_schedule_with_warmup
import optimum.quanto.tensor.function
import transformers
import diffusers


# ============================================================
# 时间步采样
# ============================================================
TimestepType = Literal["linear", "sigmoid", "weighted", "shift", "lognorm_blend"]

TIMESTEP_TYPES = ["linear", "sigmoid", "weighted", "shift", "lognorm_blend"]

TIMESTEP_DESCRIPTIONS = {
    "linear": "均匀分布 - 通用场景，各时间步平衡学习",
    "sigmoid": "集中中间 - 适合学习细节、纹理和风格",
    "weighted": "中间权重高 - 适合蒸馏模型训练和风格迁移",
    "shift": "偏向高噪声 - 快速学习构图和整体结构",
    "lognorm_blend": "对数正态+线性混合 - 平衡构图和细节",
}

# ============================================================
# Loss Weighting Scheme
# ============================================================
LossWeightingType = Literal["none", "sigma_sqrt", "cosmap"]

LOSS_WEIGHTING_TYPES = ["none", "sigma_sqrt", "cosmap"]

LOSS_WEIGHTING_DESCRIPTIONS = {
    "none": "无权重 - 所有时间步等权重",
    "sigma_sqrt": "Sigma平方根倒数 - 低噪声时权重更高，改善细节学习",
    "cosmap": "余弦映射 - 平滑的权重分布，中间时间步权重较高",
}


def compute_loss_weighting(
    weighting_scheme: LossWeightingType,
    timesteps: torch.Tensor,
    num_train_timesteps: int = 1000,
    device: torch.device = None,
) -> torch.Tensor:
    if weighting_scheme == "none" or weighting_scheme is None:
        return torch.ones(timesteps.shape[0], device=device)
    
    sigmas = (timesteps.float() + 1) / num_train_timesteps
    
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas ** -2.0).float()
    elif weighting_scheme == "cosmap":
        import math
        bot = 1 - 2 * sigmas + 2 * sigmas ** 2 + 1e-8
        weighting = (2 / (math.pi * bot)).float()
    else:
        weighting = torch.ones(timesteps.shape[0], device=device)
    
    return weighting

_DEFAULT_WEIGHING_SCHEME = None


def _get_default_weighing_scheme(num_timesteps: int = 1000) -> torch.Tensor:
    global _DEFAULT_WEIGHING_SCHEME
    if _DEFAULT_WEIGHING_SCHEME is None or len(_DEFAULT_WEIGHING_SCHEME) != num_timesteps:
        x = torch.arange(num_timesteps, dtype=torch.float32)
        y = torch.exp(-2 * ((x - num_timesteps / 2) / num_timesteps) ** 2)
        y_shifted = y - y.min()
        _DEFAULT_WEIGHING_SCHEME = y_shifted * (num_timesteps / y_shifted.sum())
    return _DEFAULT_WEIGHING_SCHEME


def linear_timestep_sampling(
    batch_size: int,
    num_train_timesteps: int = 1000,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    timesteps = torch.randint(0, num_train_timesteps, (batch_size,), device=device)
    weights = torch.ones(batch_size, device=device)
    return timesteps, weights


def sigmoid_timestep_sampling(
    batch_size: int,
    num_train_timesteps: int = 1000,
    device: torch.device = None,
    sigmoid_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    u = torch.rand(batch_size, device=device)
    eps = 1e-4
    u = u * (1 - 2 * eps) + eps
    logit_u = torch.log(u / (1 - u))
    logit_u = logit_u / sigmoid_scale
    t_normalized = torch.sigmoid(logit_u)
    timesteps = (t_normalized * num_train_timesteps).long()
    timesteps = torch.clamp(timesteps, 0, num_train_timesteps - 1)
    weights = torch.ones(batch_size, device=device)
    return timesteps, weights


def weighted_timestep_sampling(
    batch_size: int,
    num_train_timesteps: int = 1000,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    timesteps = torch.randint(0, num_train_timesteps, (batch_size,), device=device)
    weighing_scheme = _get_default_weighing_scheme(num_train_timesteps).to(device)
    weights = weighing_scheme[timesteps]
    return timesteps, weights


def shift_timestep_sampling(
    batch_size: int,
    num_train_timesteps: int = 1000,
    device: torch.device = None,
    shift: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    u = torch.rand(batch_size, device=device)
    sigmas = shift * u / (1 + (shift - 1) * u)
    timesteps = (sigmas * num_train_timesteps).long()
    timesteps = torch.clamp(timesteps, 0, num_train_timesteps - 1)
    weights = torch.ones(batch_size, device=device)
    return timesteps, weights


def lognorm_blend_timestep_sampling(
    batch_size: int,
    num_train_timesteps: int = 1000,
    device: torch.device = None,
    alpha: float = 0.75,
    loc: float = 0.0,
    scale: float = 0.333,
) -> tuple[torch.Tensor, torch.Tensor]:
    lognormal = LogNormal(loc=loc, scale=scale)
    num_lognorm = int(batch_size * alpha)
    num_linear = batch_size - num_lognorm
    timesteps_list = []
    if num_lognorm > 0:
        t1 = lognormal.sample((num_lognorm,)).to(device)
        t1 = torch.clamp(t1, 0, 1)
        t1 = ((1 - t1) * num_train_timesteps).long()
        t1 = torch.clamp(t1, 0, num_train_timesteps - 1)
        timesteps_list.append(t1)
    if num_linear > 0:
        t2 = torch.randint(0, num_train_timesteps, (num_linear,), device=device)
        timesteps_list.append(t2)
    timesteps = torch.cat(timesteps_list, dim=0) if len(timesteps_list) > 1 else timesteps_list[0]
    perm = torch.randperm(batch_size, device=device)
    timesteps = timesteps[perm]
    weights = torch.ones(batch_size, device=device)
    return timesteps, weights


def sample_timesteps(
    batch_size: int,
    num_train_timesteps: int = 1000,
    device: torch.device = None,
    timestep_type: TimestepType = "sigmoid",
    sigmoid_scale: float = 1.0,
    shift: float = 3.0,
    lognorm_alpha: float = 0.75,
    min_timestep: int = None,
    max_timestep: int = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if timestep_type == "linear":
        timesteps, weights = linear_timestep_sampling(batch_size, num_train_timesteps, device)
    elif timestep_type == "sigmoid":
        timesteps, weights = sigmoid_timestep_sampling(batch_size, num_train_timesteps, device, sigmoid_scale)
    elif timestep_type == "weighted":
        timesteps, weights = weighted_timestep_sampling(batch_size, num_train_timesteps, device)
    elif timestep_type == "shift":
        timesteps, weights = shift_timestep_sampling(batch_size, num_train_timesteps, device, shift)
    elif timestep_type == "lognorm_blend":
        timesteps, weights = lognorm_blend_timestep_sampling(batch_size, num_train_timesteps, device, lognorm_alpha)
    else:
        raise ValueError(f"Unknown timestep type: {timestep_type}")
    
    if min_timestep is not None or max_timestep is not None:
        t_min = min_timestep if min_timestep is not None else 0
        t_max = max_timestep if max_timestep is not None else num_train_timesteps
        timesteps = torch.clamp(timesteps, min=t_min, max=t_max - 1)
    
    return timesteps, weights


# ============================================================
# 全局 Trainer 实例和停止信号
# ============================================================
_global_trainer_instance = None
_script_dir = os.path.dirname(os.path.abspath(__file__))
STOP_SIGNAL_FILE = os.path.join(_script_dir, ".stop_training")
STOP_COMPLETE_FILE = os.path.join(_script_dir, ".stop_complete")


def check_stop_signal_file() -> bool:
    return os.path.exists(STOP_SIGNAL_FILE)


def notify_stop_complete():
    try:
        with open(STOP_COMPLETE_FILE, 'w') as f:
            f.write(str(time.time()))
    except:
        pass


# ============================================================
# 全局线程/Executor 管理器
# ============================================================
import threading
import weakref

_registered_executors = []
_registered_threads = []
_executor_lock = threading.Lock()


def register_executor(executor):
    with _executor_lock:
        _registered_executors.append(weakref.ref(executor))


def register_thread(thread):
    with _executor_lock:
        _registered_threads.append(weakref.ref(thread))


def _signal_handler(signum, frame):
    print(f"\n[!] 收到停止信号 (signal={signum})，立即退出...")
    os._exit(1)


def setup_signal_handlers():
    try:
        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)
    except Exception:
        pass


# ============================================================
# Windows 控制台编码修复
# ============================================================
def fix_windows_encoding():
    """修复 Windows 控制台编码问题（支持中文路径和 Unicode 字符）"""
    if sys.platform == 'win32':
        try:
            if hasattr(sys.stdout, 'buffer'):
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
            if hasattr(sys.stderr, 'buffer'):
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        except Exception:
            pass


# ============================================================
# 分布式训练辅助函数
# ============================================================
def get_rank() -> int:
    """获取当前进程的 rank"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def is_main_process() -> bool:
    """判断是否为主进程"""
    return get_rank() == 0


def print_main(*args, **kwargs):
    """只在主进程打印"""
    if is_main_process():
        print(*args, **kwargs)


# ============================================================
# Quanto Workaround
# ============================================================
def apply_quanto_fix(verbose: bool = True):
    """应用 quanto 反向传播修复"""
    def fixed_quanto_backward(ctx, gO):
        input_gO = other_gO = bias_gO = None
        input, other = ctx.saved_tensors
        out_features, in_features = other.shape
        if ctx.needs_input_grad[0]:
            input_gO = torch.matmul(gO, other)
        if ctx.needs_input_grad[1]:
            other_gO = torch.matmul(
                gO.reshape(-1, out_features).t(),
                input.to(gO.dtype).reshape(-1, in_features),
            )
        if ctx.needs_input_grad[2]:
            dim = tuple(range(gO.ndim - 1))
            bias_gO = gO.sum(dim)
        return input_gO, other_gO, bias_gO
    
    optimum.quanto.tensor.function.QuantizedLinearFunction.backward = fixed_quanto_backward
    if verbose:
        print("✓ 应用 quanto workaround")


# ============================================================
# Packing / Unpacking 辅助函数
# ============================================================
def pack_latents(latents: torch.Tensor, batch_size: int, num_channels: int, height: int, width: int) -> torch.Tensor:
    """将 latents 打包为序列格式"""
    latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)
    return latents


def unpack_latents(latents: torch.Tensor, height: int, width: int, vae_scale_factor: int) -> torch.Tensor:
    """将序列格式的 latents 解包"""
    batch_size, num_patches, channels = latents.shape
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))
    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)
    return latents


# ============================================================
# 基础配置类
# ============================================================
class BaseTrainerConfig:
    """训练基础配置类"""
    
    def __init__(
        self,
        model_id: str,
        output_dir: str,
        num_train_steps: int = 5000,
        checkpoint_every_n_steps: int = 500,
        checkpoints_total_limit: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-4,
        lr_warmup_steps: int = 500,
        resolution: int = 1024,
        device: str = None,
        dtype: torch.dtype = torch.bfloat16,
        quantize_transformer: bool = True,
        quantize_text_encoder: bool = True,
        quantize_level = None,
        vae_scale_factor: int = 8,
        latent_channels: int = 16,
        use_tensorboard: bool = True,
        tensorboard_dir: str = None,
        gradient_accumulation_steps: int = 1,
        mixed_precision: str = "bf16",
        seed: int = 42,
        max_grad_norm: float = 1.0,
        resume_from_checkpoint: str = None,
    ):
        self.model_id = model_id
        self.output_dir = output_dir
        
        self.num_train_steps = num_train_steps
        self.checkpoint_every_n_steps = checkpoint_every_n_steps
        self.checkpoints_total_limit = checkpoints_total_limit
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lr_warmup_steps = lr_warmup_steps
        self.resolution = resolution
        
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        
        self.quantize_transformer = quantize_transformer
        self.quantize_text_encoder = quantize_text_encoder
        self.quantize_level = quantize_level
        
        self.vae_scale_factor = vae_scale_factor
        self.latent_channels = latent_channels
        
        self.use_tensorboard = use_tensorboard
        self.tensorboard_dir = tensorboard_dir
        
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        self.seed = seed
        self.max_grad_norm = max_grad_norm
        self.resume_from_checkpoint = resume_from_checkpoint
    
    @property
    def torch_dtype(self) -> torch.dtype:
        """获取 torch dtype"""
        return self.dtype


# ============================================================
# 基础训练器抽象类
# ============================================================
class BaseTrainer(ABC):
    """
    基础训练器抽象类
    
    子类需要实现：
    - load_models(): 加载模型
    - create_dataset(): 创建数据集
    - get_trainable_params(): 获取可训练参数
    - train_step(): 执行一个训练步骤
    - save_checkpoint(): 保存检查点
    - save_final_model(): 保存最终模型
    """
    
    def __init__(self, config: BaseTrainerConfig):
        global _global_trainer_instance
        _global_trainer_instance = self
        
        self.config = config
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 设置信号处理器（用于优雅停止）
        setup_signal_handlers()
        
        # 初始化 Accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
        )
        
        # 设置随机种子
        if config.seed is not None:
            set_seed(config.seed)
        
        # 配置日志级别
        self._configure_logging()
        
        # 更新设备
        self.config.device = str(self.accelerator.device)
        
        # 打印初始化信息
        if self.accelerator.is_main_process:
            print(f"✓ Accelerator 初始化完成")
            print(f"  - 设备: {self.accelerator.device}")
            print(f"  - 进程数: {self.accelerator.num_processes}")
            print(f"  - 混合精度: {self.config.mixed_precision}")
            print(f"  - 梯度累积步数: {self.config.gradient_accumulation_steps}")
        
        # 应用 quanto 修复 (只要有任何模型使用 quanto 就需要)
        use_fp8 = getattr(self.config, 'use_fp8', False)
        quantize_text_encoder = getattr(self.config, 'quantize_text_encoder', True)
        quantize_transformer = getattr(self.config, 'quantize_transformer', True)
        needs_quanto = (not use_fp8 and quantize_transformer) or quantize_text_encoder
        if needs_quanto:
            apply_quanto_fix(verbose=self.accelerator.is_main_process)
        
        # 初始化变量
        self.vae = None
        self.transformer = None
        self.text_encoder = None
        self.tokenizer = None
        self.processor = None
        self.noise_scheduler = None
        self.dataset = None
        self.dataloader = None
        self.optimizer = None
        self.lr_scheduler = None
        
        self.should_stop = False
        self.current_step = 0
        self.resume_step = 0
        self.tensorboard_writer = None
        
        self.effective_batch_size = None
        self.adjusted_num_train_steps = None
        self.adjusted_checkpoint_every = None
        self.adjusted_warmup_steps = None
    
    def _configure_logging(self):
        """配置日志级别"""
        if not self.accelerator.is_main_process:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("diffusers").setLevel(logging.ERROR)
            from functools import partialmethod
            tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
        else:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_warning()
    
    def flush_memory(self):
        """清理 GPU 内存"""
        torch.cuda.empty_cache()
        gc.collect()
    
    def print_status(self, message: str):
        """只在主进程打印状态消息"""
        if self.accelerator.is_main_process:
            print(message)
    
    # ============================================================
    # 抽象方法 - 子类必须实现
    # ============================================================
    
    @abstractmethod
    def load_models(self):
        """加载所有需要的模型（VAE、Transformer、Text Encoder 等）"""
        pass
    
    @abstractmethod
    def create_dataset(self):
        """创建数据集和 DataLoader"""
        pass
    
    @abstractmethod
    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        """获取可训练参数列表"""
        pass
    
    @abstractmethod
    def train_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """执行一个训练步骤，返回 loss"""
        pass
    
    @abstractmethod
    def save_checkpoint(self, step: int):
        """保存检查点"""
        pass
    
    @abstractmethod
    def save_final_model(self):
        """保存最终模型"""
        pass
    
    def save_accelerate_state(self, checkpoint_dir: str, step: int):
        """使用 accelerate 保存完整训练状态"""
        accelerate_state_dir = os.path.join(checkpoint_dir, "accelerate_state")
        self.accelerator.save_state(accelerate_state_dir)
        
        state_info = {
            "step": step,
            "num_train_steps": self.config.num_train_steps,
            "adjusted_num_train_steps": self.adjusted_num_train_steps,
        }
        if self.accelerator.is_main_process:
            state_path = os.path.join(checkpoint_dir, "training_state.json")
            with open(state_path, "w") as f:
                json.dump(state_info, f, indent=2)
    
    def load_checkpoint(self, checkpoint_dir: str):
        """从检查点恢复训练状态"""
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
        
        transformer_dir = os.path.join(checkpoint_dir, "transformer")
        if os.path.exists(transformer_dir) and self.transformer is not None:
            if self.accelerator.is_main_process:
                print(f"  - 加载 transformer 权重...")
            state_dict = {}
            from safetensors.torch import load_file
            for f in os.listdir(transformer_dir):
                if f.endswith(".safetensors") and not f.endswith(".index.json"):
                    file_path = os.path.join(transformer_dir, f)
                    state_dict.update(load_file(file_path))
            if state_dict:
                self.transformer.load_state_dict(state_dict, strict=False)
                if self.accelerator.is_main_process:
                    print(f"  - Transformer 权重已恢复")
        
        accelerate_state_dir = os.path.join(checkpoint_dir, "accelerate_state")
        if os.path.exists(accelerate_state_dir):
            self.accelerator.load_state(accelerate_state_dir)
            if self.accelerator.is_main_process:
                print(f"  - Accelerate 状态已恢复 (optimizer, scheduler, RNG)")
    
    # ============================================================
    # 可选覆盖的方法
    # ============================================================
    
    def pre_training_hook(self):
        """训练开始前的钩子（可选覆盖）"""
        pass
    
    def post_training_hook(self):
        """训练结束后的钩子（可选覆盖）"""
        pass
    
    def _handle_stop(self):
        """处理停止训练 - 返回后会执行 save_final_model()"""
        # 这个方法只是为了让代码更清晰
        # 返回后 run() 会继续执行 save_final_model()
        pass
    
    def create_optimizer(self, trainable_params: List[torch.nn.Parameter]):
        """创建优化器（可覆盖以使用不同的优化器）"""
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(
            trainable_params,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-6,
        )
    
    def create_lr_scheduler(self, optimizer):
        """创建学习率调度器（可覆盖）"""
        return get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.adjusted_warmup_steps,
        )
    
    # ============================================================
    # 核心训练方法
    # ============================================================
    
    def setup_training(self):
        """设置训练组件"""
        if self.accelerator.is_main_process:
            print("\n设置训练组件")
            print("=" * 60)
        
        # 计算有效批次大小和调整后的步数
        scale_factor = self.accelerator.num_processes * self.config.gradient_accumulation_steps
        self.effective_batch_size = self.config.batch_size * scale_factor
        self.adjusted_num_train_steps = self.config.num_train_steps // scale_factor
        self.adjusted_checkpoint_every = max(1, self.config.checkpoint_every_n_steps // scale_factor)
        self.adjusted_warmup_steps = max(1, self.config.lr_warmup_steps // scale_factor)
        
        if self.accelerator.is_main_process:
            print(f"✓ 多卡训练步数调整:")
            print(f"  - 原始步数: {self.config.num_train_steps}")
            print(f"  - 调整后步数: {self.adjusted_num_train_steps} (÷{scale_factor})")
            print(f"  - 有效批次大小: {self.effective_batch_size}")
            print(f"  - 检查点间隔: {self.adjusted_checkpoint_every} 步")
            print(f"  - 预热步数: {self.adjusted_warmup_steps}")
        
        # 获取可训练参数
        trainable_params = self.get_trainable_params()
        
        # 创建优化器和调度器
        self.optimizer = self.create_optimizer(trainable_params)
        self.lr_scheduler = self.create_lr_scheduler(self.optimizer)
        
        # 使用 Accelerator 准备
        self.optimizer, self.dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.optimizer, self.dataloader, self.lr_scheduler
        )
        
        if self.accelerator.is_main_process:
            print(f"✓ Accelerator 已准备优化器和数据加载器")
        
        self._setup_tensorboard()
        
        if self.config.resume_from_checkpoint:
            self.load_checkpoint(self.config.resume_from_checkpoint)
        
        if self.accelerator.is_main_process:
            print("✓ 训练组件设置完成")
    
    def _setup_tensorboard(self):
        """设置 TensorBoard"""
        if self.config.use_tensorboard and self.accelerator.is_main_process:
            from datetime import datetime
            
            if self.config.tensorboard_dir:
                tensorboard_base_dir = self.config.tensorboard_dir
            elif self.config.output_dir:
                tensorboard_base_dir = os.path.join(self.config.output_dir, "tensorboard")
            else:
                tensorboard_base_dir = os.path.join(self.script_dir, "output", "tensorboard")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            num_gpus = self.accelerator.num_processes
            run_name = f"train_{self.config.num_train_steps}steps_{num_gpus}gpu_{timestamp}"
            tensorboard_dir = os.path.join(tensorboard_base_dir, run_name)
            
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(log_dir=tensorboard_dir)
            print(f"✓ TensorBoard 日志目录: {tensorboard_dir}")
    
    def cleanup_old_checkpoints(self):
        """清理旧检查点"""
        output_dir = self.config.output_dir or os.path.join(self.script_dir, "output")
        checkpoints_dir = os.path.join(output_dir, "checkpoints")
        
        if not os.path.exists(checkpoints_dir):
            return
        
        checkpoints = []
        for d in os.listdir(checkpoints_dir):
            if d.startswith("checkpoint-"):
                try:
                    step = int(d.split("-")[1])
                    checkpoints.append((step, d))
                except:
                    continue
        
        checkpoints.sort()
        if len(checkpoints) > self.config.checkpoints_total_limit:
            num_to_remove = len(checkpoints) - self.config.checkpoints_total_limit
            for i in range(num_to_remove):
                step, dirname = checkpoints[i]
                checkpoint_path = os.path.join(checkpoints_dir, dirname)
                shutil.rmtree(checkpoint_path, ignore_errors=True)
    
    def train(self):
        """核心训练循环"""
        if self.accelerator.is_main_process:
            if os.path.exists(STOP_SIGNAL_FILE):
                os.remove(STOP_SIGNAL_FILE)
            if os.path.exists(STOP_COMPLETE_FILE):
                os.remove(STOP_COMPLETE_FILE)
        self.accelerator.wait_for_everyone()
        
        num_train_steps = self.adjusted_num_train_steps
        checkpoint_every = self.adjusted_checkpoint_every
        start_step = self.resume_step
        
        import psutil
        import psutil
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        print(f"[DEBUG] 训练开始前进程统计: 当前进程 PID={current_process.pid}, 子进程数={len(children)}")
        for i, child in enumerate(children[:10]):  # 只显示前10个
            try:
                print(f"  [子进程 {i}] PID={child.pid}, name={child.name()}, status={child.status()}")
            except:
                pass
        if len(children) > 10:
            print(f"  ... 还有 {len(children) - 10} 个子进程")
        
        if self.accelerator.is_main_process:
            print("\n开始训练")
            print("=" * 60)
            print(f"✓ 分布式训练: {self.accelerator.num_processes} 个 GPU")
            print(f"✓ 每GPU批次: {self.config.batch_size}, 有效批次: {self.effective_batch_size}")
            print(f"✓ 梯度累积步数: {self.config.gradient_accumulation_steps}")
            print(f"✓ 调整后总步数: {num_train_steps} (原始: {self.config.num_train_steps})")
            if start_step > 0:
                print(f"✓ 从第 {start_step} 步恢复训练")
            print(f"✓ 保存频率: 每 {checkpoint_every} 步\n")
        
        self.pre_training_hook()
        
        if self.transformer is not None:
            self.transformer.train()
        
        initial_weight_sample = None
        weight_sample_name = None
        weight_sample_source = None
        if self.accelerator.is_main_process:
            lokr_modules = getattr(self, 'lokr_modules', None)
            if lokr_modules and len(lokr_modules) > 0:
                for lokr_module in lokr_modules:
                    for name, param in lokr_module.named_parameters():
                        if param.requires_grad and param.numel() > 100:
                            initial_weight_sample = param.data.detach().clone().cpu().float()
                            weight_sample_name = name
                            weight_sample_source = lokr_module
                            print(f"[Weight Monitor] Tracking LoKr: {name}")
                            print(f"  Initial: mean={initial_weight_sample.mean().item():.6f}, std={initial_weight_sample.std().item():.6f}")
                            break
                    if initial_weight_sample is not None:
                        break
            elif self.transformer is not None:
                for name, param in self.transformer.named_parameters():
                    if param.requires_grad and "weight" in name and param.numel() > 1000:
                        initial_weight_sample = param.data.detach().clone().cpu().float()
                        weight_sample_name = name
                        weight_sample_source = self.transformer
                        print(f"[Weight Monitor] Tracking: {name}")
                        print(f"  Initial: mean={initial_weight_sample.mean().item():.6f}, std={initial_weight_sample.std().item():.6f}")
                        break
        
        start_time = time.time()
        step_times = []
        global_step = start_step
        
        trainable_params = self.get_trainable_params()
        
        progress_bar = None
        if self.accelerator.is_main_process:
            progress_bar = tqdm(total=num_train_steps, initial=start_step, desc="训练进度")
        
        self.accelerator.wait_for_everyone()
        
        for epoch in range(1000):
            for batch in self.dataloader:
                if self.should_stop or check_stop_signal_file():
                    self.should_stop = True
                    self.current_step = global_step
                    if self.accelerator.is_main_process:
                        print(f"\n[stop] 训练在第 {global_step} 步被停止，正在保存模型...")
                        if self.tensorboard_writer:
                            self.tensorboard_writer.close()
                    return self._handle_stop()
                
                if global_step >= num_train_steps:
                    break
                
                step_start_time = time.time()
                current_step = global_step + 1
                
                with self.accelerator.accumulate(self.transformer):
                    loss = self.train_step(batch)
                    
                    use_fused = getattr(self, 'use_fused_backward', False)
                    
                    if use_fused:
                        self.lr_scheduler.step()
                    
                    self.accelerator.backward(loss)
                    
                    # 对于 fused backward 模式：
                    # - 梯度处理已经在 grad_hook 中完成（reduce + step_param）
                    # - 如果子类定义了 _process_pending_gradients，则调用它
                    if use_fused and hasattr(self, '_process_pending_gradients'):
                        self._process_pending_gradients()
                    
                    if not use_fused:
                        if self.accelerator.sync_gradients:
                            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=self.config.max_grad_norm)
                        
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.lr_scheduler.step()
                
                avg_loss = self.accelerator.gather(loss.repeat(self.config.batch_size)).mean()
                
                # 多卡 reduce 检查（只在第一步打印）
                if current_step == 1:
                    self._first_step_reduce_checked = True
                    
                    if self.accelerator.is_main_process:
                        reduce_count = getattr(self, '_grad_hook_reduce_count', 0)
                        total_count = getattr(self, '_grad_hook_total_count', 0)
                        if total_count > 0:
                            print(f"\n[Multi-GPU Reduce Stats @ step 1]")
                            print(f"  Total grad hooks called: {total_count}")
                            print(f"  Reduce operations: {reduce_count}")
                            if reduce_count > 0:
                                print(f"  [OK] Multi-GPU gradient sync is working!")
                            else:
                                print(f"  [INFO] Single GPU mode or reduce not needed")
                
                # 调试：定期打印进程数量
                if current_step in [1, 5, 10, 20, 50]:
                    import psutil
                    current_process = psutil.Process()
                    children = current_process.children(recursive=True)
                    print(f"[DEBUG step {current_step}] 子进程数量: {len(children)}")
                
                step_time = time.time() - step_start_time
                step_times.append(step_time)
                avg_step_time = sum(step_times[-100:]) / len(step_times[-100:])
                remaining_steps = num_train_steps - current_step
                eta_seconds = avg_step_time * remaining_steps
                eta_minutes = int(eta_seconds // 60)
                eta_secs = int(eta_seconds % 60)
                eta_str = f"{eta_minutes}m{eta_secs}s" if eta_minutes > 0 else f"{eta_secs}s"
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # 日志记录
                if self.accelerator.is_main_process:
                    if progress_bar:
                        progress_bar.update(1)
                        postfix_dict = {
                            'loss': f'{avg_loss.item():.4f}',
                            'lr': f'{current_lr:.2e}',
                            'ETA': eta_str
                        }
                        if hasattr(self, 'last_image_name') and self.last_image_name:
                            postfix_dict['img'] = self.last_image_name[:20]
                        if hasattr(self, 'last_aux_loss') and self.last_aux_loss is not None:
                            postfix_dict['aux'] = f'{self.last_aux_loss:.4f}'
                        if hasattr(self, 'last_recon_loss') and self.last_recon_loss is not None:
                            postfix_dict['recon'] = f'{self.last_recon_loss:.4f}'
                        progress_bar.set_postfix(postfix_dict)
                    
                    if self.tensorboard_writer:
                        self.tensorboard_writer.add_scalar('train/loss', avg_loss.item(), current_step)
                        self.tensorboard_writer.add_scalar('train/learning_rate', current_lr, current_step)
                        self.tensorboard_writer.add_scalar('train/step_time', step_time, current_step)
                
                if current_step % 50 == 0 and initial_weight_sample is not None and self.accelerator.is_main_process:
                    params_to_check = []
                    if weight_sample_source is not None and weight_sample_source is not self.transformer:
                        params_to_check = list(weight_sample_source.named_parameters())
                    elif self.transformer is not None:
                        params_to_check = list(self.transformer.named_parameters())
                    
                    for name, param in params_to_check:
                        if name == weight_sample_name:
                            current_weight = param.data.detach().cpu().float()
                            weight_diff = (current_weight - initial_weight_sample).abs().mean().item()
                            current_mean = current_weight.mean().item()
                            current_std = current_weight.std().item()
                            source_type = "LoKr" if weight_sample_source is not self.transformer else "Transformer"
                            print(f"\n[Weight Monitor @ step {current_step}] {source_type}: {name}")
                            print(f"  Current: mean={current_mean:.6f}, std={current_std:.6f}")
                            print(f"  Diff from initial: {weight_diff:.8f}")
                            if weight_diff < 1e-8:
                                print(f"  [WARNING] Weight not changing!")
                            break
                
                # 保存检查点
                if current_step % checkpoint_every == 0:
                    self.accelerator.wait_for_everyone()
                    if self.accelerator.is_main_process:
                        self.save_checkpoint(current_step)
                        self.cleanup_old_checkpoints()
                        print(f"  ✓ 检查点已保存: checkpoint-{current_step}")
                
                global_step += 1
            
            if global_step >= num_train_steps:
                break
        
        self.current_step = global_step
        
        if progress_bar:
            progress_bar.close()
        
        if self.accelerator.is_main_process:
            total_time = time.time() - start_time
            total_minutes = int(total_time // 60)
            total_secs = int(total_time % 60)
            print(f"\n✓ 训练完成! 总时间: {total_minutes}m{total_secs}s")
            print(f"  - 实际训练步数: {global_step}")
            print(f"  - 等效单卡步数: {global_step * self.accelerator.num_processes * self.config.gradient_accumulation_steps}")
            
            if self.tensorboard_writer:
                self.tensorboard_writer.close()
        
        # 调用训练后钩子
        self.post_training_hook()
    
    def run(self):
        try:
            if self.accelerator.is_main_process:
                print("=" * 60)
                print(f"开始训练 (使用 {self.accelerator.num_processes} 个 GPU)")
                print("=" * 60)
            
            # 子类实现的方法 - 注意顺序：先创建数据集，再加载模型
            self.create_dataset()
            self.load_models()
            
            # 基类方法
            self.setup_training()
            
            # 等待同步
            self.accelerator.wait_for_everyone()
            
            self.train()
            
            try:
                self.save_final_model()
            except Exception as e:
                if self.accelerator.is_main_process:
                    print(f"\n[!] 保存模型时出错: {e}")
                    import traceback
                    traceback.print_exc()
            
            self.accelerator.wait_for_everyone()
            
            if self.accelerator.is_main_process:
                if self.should_stop:
                    notify_stop_complete()
                    print("\n[ok] 停止处理完成")
                else:
                    print("\n[ok] 训练完成")
        finally:
            if self.accelerator.is_main_process and self.should_stop:
                notify_stop_complete()
            if dist.is_initialized():
                dist.destroy_process_group()

