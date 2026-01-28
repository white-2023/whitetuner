import math
import torch
from transformers import Adafactor

def copy_stochastic_(target: torch.Tensor, source: torch.Tensor):
    result = torch.randint_like(source, dtype=torch.int32, low=0, high=(1 << 16))
    result.add_(source.view(dtype=torch.int32))
    result.bitwise_and_(-65536)
    target.copy_(result.view(dtype=torch.float32))
    del result


@torch.no_grad()
def adafactor_step_param(self, p, group):
    if p.grad is None:
        return
    
    grad = p.grad
    grad_device = grad.device
    
    if grad.dtype in {torch.float16, torch.bfloat16}:
        grad = grad.float()
    if grad.is_sparse:
        raise RuntimeError("Adafactor does not support sparse gradients.")

    state = self.state[p]
    grad_shape = grad.shape

    factored, use_first_moment = Adafactor._get_options(group, grad_shape)
    
    if len(state) == 0:
        state["step"] = 0
        if use_first_moment:
            state["exp_avg"] = torch.zeros_like(grad)
        if factored:
            state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(grad)
            state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).to(grad)
        else:
            state["exp_avg_sq"] = torch.zeros_like(grad)
        state["RMS"] = 0
    else:
        if use_first_moment:
            state["exp_avg"] = state["exp_avg"].to(grad)
        if factored:
            state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
            state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
        else:
            state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

    p_data_fp32 = p.data.to(grad_device)
    if p.dtype in {torch.float16, torch.bfloat16}:
        p_data_fp32 = p_data_fp32.float()

    state["step"] += 1
    state["RMS"] = Adafactor._rms(p_data_fp32)
    lr = Adafactor._get_lr(group, state)

    beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
    update = (grad**2) + group["eps"][0]
    if factored:
        exp_avg_sq_row = state["exp_avg_sq_row"]
        exp_avg_sq_col = state["exp_avg_sq_col"]

        exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=(1.0 - beta2t))
        exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=(1.0 - beta2t))

        update = Adafactor._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
        update.mul_(grad)
    else:
        exp_avg_sq = state["exp_avg_sq"]
        exp_avg_sq.mul_(beta2t).add_(update, alpha=(1.0 - beta2t))
        update = exp_avg_sq.rsqrt().mul_(grad)

    update.div_((Adafactor._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
    update.mul_(lr)

    if use_first_moment:
        exp_avg = state["exp_avg"]
        exp_avg.mul_(group["beta1"]).add_(update, alpha=(1 - group["beta1"]))
        update = exp_avg

    if group["weight_decay"] != 0:
        p_data_fp32.add_(p_data_fp32, alpha=(-group["weight_decay"] * lr))

    p_data_fp32.add_(-update)

    # 使用确定性舍入（round-to-nearest）而不是随机舍入
    # 随机舍入在多卡训练时会导致不同 GPU 上的权重不一致
    if p.dtype == torch.bfloat16:
        p.data.copy_(p_data_fp32.to(p.device, torch.bfloat16))
    elif p.dtype == torch.float16:
        p.data.copy_(p_data_fp32.to(p.device, p.dtype))
    else:
        p.data.copy_(p_data_fp32.to(p.device))


@torch.no_grad()
def adafactor_step(self, closure=None):
    loss = None
    if closure is not None:
        loss = closure()

    for group in self.param_groups:
        for p in group["params"]:
            adafactor_step_param(self, p, group)

    return loss


def patch_adafactor_fused(optimizer: Adafactor):
    optimizer.step_param = adafactor_step_param.__get__(optimizer)
    optimizer.step = adafactor_step.__get__(optimizer)

