import math
import torch
import torch.nn.functional as F
from optimum.quanto import QTensor, QBytesTensor


def factorization(dimension: int, factor: int = -1) -> tuple:
    if factor > 0 and (dimension % factor) == 0:
        m = factor
        n = dimension // factor
        return m, n
    if factor == -1:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        else:
            m, n = new_m, new_n
    if m > n:
        n, m = m, n
    return m, n


def make_kron(w1: torch.Tensor, w2: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if len(w2.shape) == 4:
        w1 = w1.unsqueeze(2).unsqueeze(2)
    w2 = w2.contiguous()
    rebuild = torch.kron(w1, w2)
    return rebuild * scale


class LokrModule(torch.nn.Module):
    
    def __init__(
        self,
        org_module: torch.nn.Module,
        lora_dim: int = 4,
        alpha: int = 1,
        factor: int = 4,
        multiplier: float = 1.0,
        decompose_both: bool = False,
        full_matrix: bool = False,
    ):
        super().__init__()
        factor = int(factor)
        self.lora_dim = lora_dim
        self.use_w1 = False
        self.use_w2 = False
        self.full_matrix = full_matrix
        
        self.shape = org_module.weight.shape
        if org_module.__class__.__name__ in ['Linear', 'LoRACompatibleLinear', 'QLinear']:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            
            in_m, in_n = factorization(in_dim, factor)
            out_l, out_k = factorization(out_dim, factor)
            shape = ((out_l, out_k), (in_m, in_n))
            
            if decompose_both and lora_dim < max(shape[0][0], shape[1][0])/2 and not self.full_matrix:
                self.lokr_w1_a = torch.nn.Parameter(torch.empty(shape[0][0], lora_dim))
                self.lokr_w1_b = torch.nn.Parameter(torch.empty(lora_dim, shape[1][0]))
            else:
                self.use_w1 = True
                self.lokr_w1 = torch.nn.Parameter(torch.empty(shape[0][0], shape[1][0]))
            
            if lora_dim < max(shape[0][1], shape[1][1])/2 and not self.full_matrix:
                self.lokr_w2_a = torch.nn.Parameter(torch.empty(shape[0][1], lora_dim))
                self.lokr_w2_b = torch.nn.Parameter(torch.empty(lora_dim, shape[1][1]))
            else:
                self.use_w2 = True
                self.lokr_w2 = torch.nn.Parameter(torch.empty(shape[0][1], shape[1][1]))
            
            self.op = F.linear
            self.extra_args = {}
        else:
            raise NotImplementedError("Only Linear layers supported")
        
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        if self.use_w2 and self.use_w1:
            alpha = lora_dim
        self.alpha = torch.tensor(float(alpha))
        self.scale = alpha / self.lora_dim
        
        if self.use_w2:
            torch.nn.init.constant_(self.lokr_w2, 0)
        else:
            torch.nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))
            torch.nn.init.constant_(self.lokr_w2_b, 0)
        
        if self.use_w1:
            torch.nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
        else:
            torch.nn.init.kaiming_uniform_(self.lokr_w1_a, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.lokr_w1_b, a=math.sqrt(5))
        
        self.multiplier = multiplier
        self.org_module = [org_module]
        self.org_forward = org_module.forward
        self.torch_multiplier = torch.tensor([multiplier], dtype=torch.float32)
        
        weight = make_kron(
            self.lokr_w1 if self.use_w1 else self.lokr_w1_a @ self.lokr_w1_b,
            self.lokr_w2 if self.use_w2 else self.lokr_w2_a @ self.lokr_w2_b,
            torch.tensor(self.multiplier * self.scale)
        )
        assert torch.sum(torch.isnan(weight)) == 0, "weight is nan"
    
    def get_orig_weight(self) -> torch.Tensor:
        weight = self.org_module[0].weight
        if isinstance(weight, (QTensor, QBytesTensor)):
            weight = weight.dequantize()
        return weight.data.detach()
    
    def get_lokr_weight(self, orig_weight: torch.Tensor = None) -> torch.Tensor:
        w1 = self.lokr_w1 if self.use_w1 else self.lokr_w1_a @ self.lokr_w1_b
        w2 = self.lokr_w2 if self.use_w2 else self.lokr_w2_a @ self.lokr_w2_b
        weight = make_kron(w1, w2, torch.tensor(self.scale))
        if orig_weight is not None:
            weight = weight.reshape(orig_weight.shape)
        return weight
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if isinstance(x, (QTensor, QBytesTensor)):
            x = x.dequantize()
        
        orig_dtype = x.dtype
        orig_weight = self.org_module[0].weight
        
        # 检查是否是 FP8 权重
        is_fp8 = orig_weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
        
        if is_fp8:
            # FP8 模式：先调用原始 forward（通过 monkey patch 处理反量化），再加上 LoKr delta
            # 使用 no_grad 包装原始 forward，因为：
            # 1. FP8 权重是冻结的，不需要梯度
            # 2. FP8 tensor 不支持梯度操作 (ufunc_add_CUDA 未实现)
            # 3. 只有 LoKr 参数需要训练
            with torch.no_grad():
                org_forwarded = self.org_forward(x)
            
            # 使用 bfloat16 计算 LoKr
            compute_dtype = torch.bfloat16
            lokr_weight = self.get_lokr_weight()
            lokr_weight = lokr_weight.to(dtype=compute_dtype, device=x.device)
            
            x_compute = x.to(compute_dtype) if x.dtype != compute_dtype else x
            lokr_out = self.op(x_compute, lokr_weight.view(self.shape), None, **self.extra_args)
            
            # detach org_forwarded 确保梯度只流向 lokr_out
            return org_forwarded.detach() + lokr_out.to(org_forwarded.dtype) * self.multiplier
        else:
            # 非 FP8 模式：权重合并方式
            orig_weight = self.get_orig_weight()
            lokr_weight = self.get_lokr_weight(orig_weight).to(dtype=orig_weight.dtype)
            
            if x.dtype != orig_weight.dtype:
                x = x.to(dtype=orig_weight.dtype)
            
            weight = orig_weight + lokr_weight * self.multiplier
            
            bias = None
            if hasattr(self.org_module[0], 'bias') and self.org_module[0].bias is not None:
                bias = self.org_module[0].bias
                if isinstance(bias, (QTensor, QBytesTensor)):
                    bias = bias.dequantize()
                bias = bias.to(weight.device, dtype=weight.dtype)
            
            output = self.op(x, weight.view(self.shape), bias, **self.extra_args)
            return output.to(orig_dtype)


def apply_lokr_to_transformer(
    transformer: torch.nn.Module,
    lora_dim: int = 4,
    alpha: int = 1,
    factor: int = 4,
    full_matrix: bool = False,
    decompose_both: bool = False,
    parameter_threshold: int = 0,
    verbose: bool = True
) -> tuple:
    lokr_modules = []
    lokr_module_names = []
    count = 0
    full_matrix_count = 0
    
    if verbose:
        print(">>> 搜索可训练的 Linear 层...")
        print(f"Transformer 类型: {transformer.__class__.__name__}")
        print(f"LoKr 配置: dim={lora_dim}, alpha={alpha}, factor={factor}, full_matrix={full_matrix}")
    
    LINEAR_MODULES = ['Linear', 'LoRACompatibleLinear', 'QLinear']
    
    # 自动检测 block 命名方式（支持 WAN 和 Qwen）
    block_attr = None
    if hasattr(transformer, 'transformer_blocks'):
        block_attr = 'transformer_blocks'
    elif hasattr(transformer, 'blocks'):
        block_attr = 'blocks'
    
    if block_attr and verbose:
        print(f"  找到 {block_attr}，仅对内部层应用 LoKr")
    
    for name, module in transformer.named_modules():
        if name == "":
            continue
        
        skip = False
        
        if block_attr:
            if block_attr not in name:
                skip = True
        
        if module.__class__.__name__ in LINEAR_MODULES and not skip:
            if hasattr(module, 'weight'):
                num_params = module.weight.numel()
                if num_params < parameter_threshold:
                    skip = True
            
            if not skip:
                try:
                    lokr = LokrModule(
                        module,
                        lora_dim=lora_dim,
                        alpha=alpha,
                        factor=factor,
                        multiplier=1.0,
                        full_matrix=full_matrix,
                        decompose_both=decompose_both
                    )
                    lokr.org_forward = module.forward
                    module.forward = lokr.forward
                    lokr_modules.append(lokr)
                    lokr_module_names.append(name)
                    count += 1
                    
                    if lokr.use_w2:
                        full_matrix_count += 1
                    
                    if count <= 3 and verbose:
                        mode = "full" if lokr.use_w2 else "low-rank"
                        w1_mode = "full" if lokr.use_w1 else "low-rank"
                        print(f"  [{count}] {name} - w1: {w1_mode}, w2: {mode}")
                except Exception as e:
                    if count <= 3 and verbose:
                        print(f"  跳过 {name}: {str(e)[:80]}")
    
    if verbose:
        print(f"已应用 {count} 个 LoKr 模块")
        print(f"  - w2 全矩阵: {full_matrix_count}")
        print(f"  - w2 低秩: {count - full_matrix_count}")
    
    return lokr_modules, lokr_module_names

