from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import gc
import time
import atexit
from typing import Optional, Any, Callable
import torch
import torch.nn as nn

try:
    from base_trainer import register_executor
except ImportError:
    def register_executor(executor):
        pass


def clean_memory_on_device(device: torch.device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "xpu":
        torch.xpu.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()


def synchronize_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def weighs_to_device(layer: nn.Module, device: torch.device):
    for module in layer.modules():
        if hasattr(module, "weight") and module.weight is not None and module.__class__.__name__.endswith("Linear"):
            module.weight.data = module.weight.data.to(device, non_blocking=device.type != "cpu")


def _move_non_linear_weight_to_device(layer: nn.Module, device: torch.device):
    """
    移动非 Linear 层的所有参数和 buffers 到指定设备
    
    Block swap 只交换 Linear 层的 weight，但其他层（如 RMSNorm、LayerNorm）的参数
    需要始终在 GPU 上才能正确执行 forward
    """
    for module in layer.modules():
        is_linear = module.__class__.__name__.endswith("Linear")
        
        if is_linear:
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data = module.bias.data.to(device, non_blocking=True)
        else:
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data = module.weight.data.to(device, non_blocking=True)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data = module.bias.data.to(device, non_blocking=True)
        
        if hasattr(module, "scale_weight") and module.scale_weight is not None:
            module.scale_weight.data = module.scale_weight.data.to(device, non_blocking=True)
        
        for name, buf in module._buffers.items():
            if buf is not None and buf.device != device:
                module._buffers[name] = buf.to(device, non_blocking=True)


def to_device(x: Any, device: torch.device) -> Any:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, list):
        return [to_device(elem, device) for elem in x]
    elif isinstance(x, tuple):
        return tuple(to_device(elem, device) for elem in x)
    elif isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        return x


def to_cpu(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.cpu()
    elif isinstance(x, list):
        return [to_cpu(elem) for elem in x]
    elif isinstance(x, tuple):
        return tuple(to_cpu(elem) for elem in x)
    elif isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    else:
        return x


def create_cpu_offloading_wrapper(func: Callable, device: torch.device) -> Callable:
    def wrapper(orig_func: Callable) -> Callable:
        def custom_forward(*inputs):
            nonlocal device, orig_func
            cuda_inputs = to_device(inputs, device)
            outputs = orig_func(*cuda_inputs)
            return to_cpu(outputs)
        return custom_forward
    return wrapper(func)


_offloader_instance_count = 0

class Offloader:
    def __init__(
        self,
        block_type: str,
        num_blocks: int,
        blocks_to_swap: int,
        device: torch.device,
        use_pinned_memory: bool = False,
        debug: bool = False,
    ):
        global _offloader_instance_count
        _offloader_instance_count += 1
        self._instance_id = _offloader_instance_count
        print(f"[DEBUG] Offloader 创建 (实例 #{self._instance_id}): block_type={block_type}, num_blocks={num_blocks}, blocks_to_swap={blocks_to_swap}")
        
        self.block_type = block_type
        self.num_blocks = num_blocks
        self.blocks_to_swap = blocks_to_swap
        self.device = device
        self.use_pinned_memory = use_pinned_memory

        import os
        if not debug:
            debug = os.getenv("WHITETUNER_OFFLOADER_DEBUG", "0") == "1"

        self.debug = debug
        self.debug_block_count = 0

        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        print(f"[DEBUG] ThreadPoolExecutor 创建 (Offloader #{self._instance_id})")
        register_executor(self.thread_pool)
        self.futures = {}
        self.cuda_available = device.type == "cuda"
        self.stream = torch.cuda.Stream(device=device) if self.cuda_available else None
        if self.cuda_available:
            print(f"[DEBUG] CUDA Stream 创建 (Offloader #{self._instance_id})")

        self.staging_buffer_a = None
        self.staging_buffer_b = None
        self.pinned_buffer = None

    def swap_weight_devices_cuda(self, device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
        assert layer_to_cpu.__class__ == layer_to_cuda.__class__

        debug_print = False
        if self.debug:
            debug_print = self.debug_block_count % 10 == 0
            self.debug_block_count += 1

        class Timer:
            def __init__(self, enabled=False):
                self.enabled = enabled
                self.totals = defaultdict(float)
                self.start_time = time.perf_counter()

            @contextmanager
            def section(self, name):
                if not self.enabled:
                    yield
                    return
                t0 = time.perf_counter()
                try:
                    yield
                finally:
                    self.totals[name] += time.perf_counter() - t0

        T = Timer(enabled=debug_print)

        weight_swap_jobs = []

        with T.section("find modules"):
            modules_to_cpu = {k: v for k, v in layer_to_cpu.named_modules()}
            for module_to_cuda_name, module_to_cuda in layer_to_cuda.named_modules():
                if (
                    hasattr(module_to_cuda, "weight")
                    and module_to_cuda.weight is not None
                    and module_to_cuda.__class__.__name__.endswith("Linear")
                ):
                    module_to_cpu = modules_to_cpu.get(module_to_cuda_name, None)
                    if module_to_cpu is not None and module_to_cpu.weight.shape == module_to_cuda.weight.shape:
                        weight_swap_jobs.append(
                            (module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data)
                        )
                    else:
                        if module_to_cuda.weight.data.device.type != device.type:
                            module_to_cuda.weight.data = module_to_cuda.weight.data.to(device)

        with T.section("synchronize before swap"):
            torch.cuda.current_stream().synchronize()

        if not self.use_pinned_memory:
            stream = self.stream
            with torch.cuda.stream(stream):
                if self.staging_buffer_a is None:
                    self.staging_buffer_a = [
                        torch.empty_like(cuda_data_view, device="cpu").pin_memory(device=device)
                        for _, _, cuda_data_view, _ in weight_swap_jobs
                    ]
                    self.staging_buffer_b = [
                        torch.empty_like(cuda_data_view, device="cpu").pin_memory(device=device)
                        for _, _, cuda_data_view, _ in weight_swap_jobs
                    ]

                event_b = None
                for sbuf_a, sbuf_b, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(
                    self.staging_buffer_a, self.staging_buffer_b, weight_swap_jobs
                ):
                    event_a = torch.cuda.Event()
                    with T.section("cuda to staging A"):
                        sbuf_a.copy_(cuda_data_view.data, non_blocking=True)
                        event_a.record(stream)

                    if event_b is not None:
                        with T.section("wait staging B"):
                            event_b.synchronize()

                    with T.section("cpu to staging B"):
                        sbuf_b.copy_(module_to_cuda.weight.data)

                    with T.section("wait staging A"):
                        event_a.synchronize()

                    event_b = torch.cuda.Event()
                    with T.section("staging B to CUDA"):
                        cuda_data_view.copy_(sbuf_b, non_blocking=True)
                        event_b.record(stream)

                    with T.section("staging A to CPU"):
                        cpu_data_view.copy_(sbuf_a)

            for sbuf_a, sbuf_b, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(
                self.staging_buffer_a, self.staging_buffer_b, weight_swap_jobs
            ):
                module_to_cuda.weight.data = cuda_data_view
                module_to_cpu.weight.data = cpu_data_view

            sync_event = event_b

        else:
            if self.pinned_buffer is None:
                with torch.cuda.stream(self.stream):
                    self.pinned_buffer = [
                        torch.empty_like(cuda_data_view, device="cpu").pin_memory(device=device)
                        for _, _, cuda_data_view, _ in weight_swap_jobs
                    ]
                self.stream.synchronize()
            released_pinned_buffer = []

            events = [torch.cuda.Event() for _ in weight_swap_jobs]

            for event, module_pin_buf, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(
                events, self.pinned_buffer, weight_swap_jobs
            ):
                with torch.cuda.stream(self.stream):
                    with T.section("cuda to cpu"):
                        module_pin_buf.copy_(cuda_data_view, non_blocking=True)
                        event.record(self.stream)

            for event, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(events, weight_swap_jobs):
                with torch.cuda.stream(self.stream):
                    with T.section("wait cpu"):
                        self.stream.wait_event(event)

                    with T.section("cpu to cuda"):
                        cuda_data_view.copy_(cpu_data_view, non_blocking=True)

            for module_pin_buf, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(
                self.pinned_buffer, weight_swap_jobs
            ):
                module_to_cuda.weight.data = cuda_data_view
                module_to_cpu.weight.data = module_pin_buf
                released_pinned_buffer.append(cpu_data_view)

            if not released_pinned_buffer[0].is_pinned():
                with torch.cuda.stream(self.stream):
                    released_pinned_buffer = [
                        torch.empty_like(cuda_data_view, device="cpu").pin_memory(device=device)
                        for _, _, cuda_data_view, _ in weight_swap_jobs
                    ]
            self.pinned_buffer = released_pinned_buffer

            sync_event = self.stream.record_event()

        if debug_print:
            print(f"[{self.block_type}] Weight swap timing at {self.debug_block_count - 1}:")
            for name, total in T.totals.items():
                print(f"  {name}: {total * 1000:.2f}ms")
            print(
                f"Overall time: {(time.perf_counter() - T.start_time) * 1000:.2f}ms, total time in sections: {sum(T.totals.values()) * 1000:.2f}ms"
            )

        return sync_event

    def swap_weight_devices_no_cuda(self, device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
        assert layer_to_cpu.__class__ == layer_to_cuda.__class__

        weight_swap_jobs = []
        for module_to_cpu, module_to_cuda in zip(layer_to_cpu.modules(), layer_to_cuda.modules()):
            if hasattr(module_to_cpu, "weight") and module_to_cpu.weight is not None:
                weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))

        for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
            module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True)

        synchronize_device(device)

        for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
            cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
            module_to_cuda.weight.data = cuda_data_view

        synchronize_device(device)

    def swap_weight_devices(self, block_to_cpu: nn.Module, block_to_cuda: nn.Module):
        if self.cuda_available:
            sync_event = self.swap_weight_devices_cuda(self.device, block_to_cpu, block_to_cuda)
        else:
            self.swap_weight_devices_no_cuda(self.device, block_to_cpu, block_to_cuda)
            sync_event = None
        return sync_event

    def _submit_move_blocks(self, blocks, block_idx_to_cpu, block_idx_to_cuda):
        def move_blocks(bidx_to_cpu, block_to_cpu, bidx_to_cuda, block_to_cuda):
            if self.debug:
                start_time = time.perf_counter()
                print(
                    f"[{self.block_type}] Move block {bidx_to_cpu} to CPU and block {bidx_to_cuda} to {'CUDA' if self.cuda_available else 'device'}"
                )

            dev = self.device.index if self.device.index is not None else torch.cuda.current_device()
            torch.cuda.set_device(dev)

            sync_event = self.swap_weight_devices(block_to_cpu, block_to_cuda)

            if self.debug:
                print(
                    f"[{self.block_type}] Moved blocks {bidx_to_cpu} to CPU and {bidx_to_cuda} to {'CUDA' if self.cuda_available else 'device'} in {time.perf_counter() - start_time:.2f}s"
                )
            return bidx_to_cpu, bidx_to_cuda, sync_event

        block_to_cpu = blocks[block_idx_to_cpu]
        block_to_cuda = blocks[block_idx_to_cuda]

        self.futures[block_idx_to_cuda] = self.thread_pool.submit(
            move_blocks, block_idx_to_cpu, block_to_cpu, block_idx_to_cuda, block_to_cuda
        )

    def _wait_blocks_move(self, block_idx):
        if block_idx not in self.futures:
            return

        if self.debug:
            print(f"[{self.block_type}] Wait for block {block_idx}")
            start_time = time.perf_counter()

        future = self.futures.pop(block_idx)
        _, bidx_to_cuda, sync_event = future.result()

        assert block_idx == bidx_to_cuda

        if self.cuda_available and sync_event is not None:
            torch.cuda.current_stream().wait_event(sync_event)

        if self.debug:
            print(f"[{self.block_type}] Waited for block {block_idx}: {time.perf_counter() - start_time:.2f}s")

    def shutdown(self):
        instance_id = getattr(self, '_instance_id', 'unknown')
        print(f"[DEBUG] Offloader shutdown 被调用 (实例 #{instance_id})")
        
        for block_idx in list(self.futures.keys()):
            try:
                future = self.futures[block_idx]
                if not future.done():
                    future.cancel()
            except:
                pass
        self.futures.clear()
        
        if hasattr(self, 'thread_pool') and self.thread_pool is not None:
            print(f"[DEBUG] ThreadPoolExecutor shutdown (实例 #{instance_id})")
            try:
                self.thread_pool.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                try:
                    self.thread_pool.shutdown(wait=False)
                except:
                    pass
            except:
                pass
            self.thread_pool = None


class ModelOffloader(Offloader):
    def __init__(
        self,
        block_type: str,
        blocks: list[nn.Module],
        num_blocks: int,
        blocks_to_swap: int,
        supports_backward: bool,
        device: torch.device,
        use_pinned_memory: bool = False,
        debug: bool = False,
    ):
        super().__init__(block_type, num_blocks, blocks_to_swap, device, use_pinned_memory, debug)

        self.supports_backward = supports_backward
        self.forward_only = not supports_backward

        if self.supports_backward:
            self.remove_handles = []
            for i, block in enumerate(blocks):
                hook = self.create_backward_hook(blocks, i)
                if hook is not None:
                    handle = block.register_full_backward_hook(hook)
                    self.remove_handles.append(handle)

    def set_forward_only(self, forward_only: bool):
        for block_idx in list(self.futures.keys()):
            self._wait_blocks_move(block_idx)

        self.forward_only = forward_only

    def __del__(self):
        if hasattr(self, 'supports_backward') and self.supports_backward:
            for handle in getattr(self, 'remove_handles', []):
                try:
                    handle.remove()
                except:
                    pass
        self.shutdown()

    def create_backward_hook(self, blocks: list[nn.Module], block_index: int) -> Optional[callable]:
        num_blocks_propagated = self.num_blocks - block_index - 1
        num_gpu_blocks = self.num_blocks - self.blocks_to_swap
        
        # 修复边界情况：当 GPU 上只有 1 个 block 时，需要使用简化的交换逻辑
        # 原因：原始公式假设 GPU 上有多个 block 可以"流水线"交换，
        # 但当只有 1 个 block 时，每次都需要：当前 block → CPU，下一个 block → GPU
        if num_gpu_blocks == 1:
            # GPU 上只有 1 个 block（即 blocks_to_swap = num_blocks - 1）
            if block_index > 0:
                swapping = True
                block_idx_to_cpu = block_index  # 当前 block 移到 CPU
                block_idx_to_cuda = block_index - 1  # 下一个需要的 block 移到 GPU
            else:
                # block 0 不需要 swap（没有下一个 block 了）
                swapping = False
                block_idx_to_cpu = 0
                block_idx_to_cuda = 0
        else:
            # 原始逻辑：GPU 上有多个 block 时使用流水线交换
            swapping = num_blocks_propagated > 0 and num_blocks_propagated <= self.blocks_to_swap
            block_idx_to_cpu = self.num_blocks - num_blocks_propagated
            block_idx_to_cuda = self.blocks_to_swap - num_blocks_propagated
        
        waiting = block_index > 0 and block_index <= self.blocks_to_swap

        if not swapping and not waiting:
            return None

        block_idx_to_wait = block_index - 1

        _backward_hook_call_count = [0]  # 使用列表以便在闭包中修改
        
        def backward_hook(module, grad_input, grad_output):
            _backward_hook_call_count[0] += 1
            call_count = _backward_hook_call_count[0]
            
            # 只打印前 2 次，减少日志量
            if call_count <= 2:
                import torch.distributed as dist
                rank = dist.get_rank() if dist.is_initialized() else 0
                print(f"[rank{rank} backward_hook] block={block_index}, swapping={swapping}, waiting={waiting}", flush=True)
            
            if swapping:
                self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
            if waiting:
                self._wait_blocks_move(block_idx_to_wait)
            return None

        return backward_hook

    def prepare_block_devices_before_forward(self, blocks: list[nn.Module]):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        if self.debug:
            print(f"[{self.block_type}] Prepare block devices before forward")

        cpu_device = torch.device("cpu")
        
        for b in blocks[self.num_blocks - self.blocks_to_swap :]:
            weighs_to_device(b, cpu_device)
        
        synchronize_device(self.device)
        clean_memory_on_device(self.device)
        
        for b in blocks[0 : self.num_blocks - self.blocks_to_swap]:
            b.to(self.device)
            weighs_to_device(b, self.device)
        
        for b in blocks[self.num_blocks - self.blocks_to_swap :]:
            _move_non_linear_weight_to_device(b, self.device)

        synchronize_device(self.device)
        clean_memory_on_device(self.device)

    def wait_for_block(self, block_idx: int):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self._wait_blocks_move(block_idx)

    def submit_move_blocks_forward(self, blocks: list[nn.Module], block_idx: int):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        if not self.forward_only:
            if block_idx >= self.blocks_to_swap:
                return
            block_idx_to_cpu = block_idx
            block_idx_to_cuda = self.num_blocks - self.blocks_to_swap + block_idx
            block_idx_to_cuda = block_idx_to_cuda % self.num_blocks
            self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
            return

        block_idx_to_cpu = block_idx

        if self.blocks_to_swap < (self.num_blocks // 2):
            if self.blocks_to_swap <= block_idx < self.num_blocks - self.blocks_to_swap:
                return
            if block_idx < self.blocks_to_swap:
                block_idx_to_cuda = (self.num_blocks - self.blocks_to_swap + block_idx) % self.num_blocks
            else:
                block_idx_to_cuda = block_idx - (self.num_blocks - self.blocks_to_swap)
        else:
            block_idx_to_cuda = self.num_blocks - self.blocks_to_swap + block_idx
            block_idx_to_cuda = block_idx_to_cuda % self.num_blocks

        self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)

