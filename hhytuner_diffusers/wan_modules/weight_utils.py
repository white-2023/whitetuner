"""Weight loading utilities (simplified from musubi-tuner, without LoRA)"""
import os
from typing import List, Optional, Union
import torch

import logging

from tqdm import tqdm

from .device_utils import synchronize_device
from .fp8_optimization_utils import load_safetensors_with_fp8_optimization
from .safetensors_utils import (
    MemoryEfficientSafeOpen,
    TensorWeightAdapter,
    WeightTransformHooks,
    get_split_weight_filenames,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_safetensors_with_dtype(
    model_files: Union[str, List[str]],
    fp8_optimization: bool,
    calc_device: torch.device,
    move_to_device: bool = False,
    weight_dtype: Optional[torch.dtype] = None,
    target_keys: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
    disable_numpy_memmap: bool = False,
    weight_transform_hooks: Optional[WeightTransformHooks] = None,
) -> dict[str, torch.Tensor]:
    """
    Load safetensors files with proper dtype conversion and optional FP8 optimization.

    Args:
        model_files (Union[str, List[str]]): Path to the model file or list of paths.
            If the path matches a pattern like `00001-of-00004`, it will load all files with the same prefix.
        fp8_optimization (bool): Whether to apply FP8 optimization.
        calc_device (torch.device): Device to calculate on.
        move_to_device (bool): Whether to move tensors to the calculation device after loading.
        weight_dtype (Optional[torch.dtype]): Target dtype for weights. Required if fp8_optimization is False.
        target_keys (Optional[List[str]]): Keys to target for FP8 optimization.
        exclude_keys (Optional[List[str]]): Keys to exclude from FP8 optimization.
        disable_numpy_memmap (bool): Whether to disable numpy memmap when loading safetensors.
        weight_transform_hooks (Optional[WeightTransformHooks]): Hooks for transforming weights during loading.
    
    Returns:
        dict[str, torch.Tensor]: Loaded state dict with proper dtype.
    """
    # Handle split weight files (e.g., 00001-of-00004)
    if isinstance(model_files, str):
        model_files = [model_files]

    extended_model_files = []
    for model_file in model_files:
        split_filenames = get_split_weight_filenames(model_file)
        if split_filenames is not None:
            extended_model_files.extend(split_filenames)
        else:
            extended_model_files.append(model_file)
    model_files = extended_model_files
    logger.info(f"Loading model files: {model_files}")

    # Load with or without FP8 optimization
    if fp8_optimization:
        logger.info(f"Loading state dict with FP8 optimization.")
        state_dict = load_safetensors_with_fp8_optimization(
            model_files,
            calc_device,
            target_keys,
            exclude_keys,
            move_to_device=move_to_device,
            weight_hook=None,
            disable_numpy_memmap=disable_numpy_memmap,
            weight_transform_hooks=weight_transform_hooks,
        )
    else:
        logger.info(f"Loading state dict with dtype={weight_dtype}")
        state_dict = {}
        for model_file in model_files:
            with MemoryEfficientSafeOpen(model_file, disable_numpy_memmap=disable_numpy_memmap) as original_f:
                f = TensorWeightAdapter(weight_transform_hooks, original_f) if weight_transform_hooks is not None else original_f
                for key in tqdm(f.keys(), desc=f"Loading {os.path.basename(model_file)}", leave=False):
                    if move_to_device:
                        # Load directly to device with target dtype
                        value = f.get_tensor(key, device=calc_device, dtype=weight_dtype)
                    else:
                        # Load to CPU, then convert dtype
                        value = f.get_tensor(key)
                        if weight_dtype is not None:
                            value = value.to(weight_dtype)
                    state_dict[key] = value
        
        if move_to_device:
            synchronize_device(calc_device)

    return state_dict

