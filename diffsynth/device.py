import importlib
from typing import Any

import torch


def is_torch_npu_available() -> bool:
    return importlib.util.find_spec("torch_npu") is not None


IS_CUDA_AVAILABLE = torch.cuda.is_available()
IS_NPU_AVAILABLE = is_torch_npu_available() and hasattr(torch, "npu") and torch.npu.is_available()

if IS_NPU_AVAILABLE:
    import torch_npu  # noqa: F401

    torch.npu.config.allow_internal_format = False


def get_device_type() -> str:
    if IS_CUDA_AVAILABLE:
        return "cuda"
    if IS_NPU_AVAILABLE:
        return "npu"
    return "cpu"


def get_torch_device(device_type: str | None = None) -> Any:
    device_type = get_device_type() if device_type is None else device_type
    try:
        return getattr(torch, device_type)
    except AttributeError:
        return torch.cuda


def get_device_id(device_type: str | None = None) -> int:
    return get_torch_device(device_type).current_device()


def get_device_name(device: Any = None) -> str:
    if isinstance(device, str):
        if ":" in device:
            return device
        return f"{device}:{get_device_id(device)}" if device != "cpu" else "cpu"
    if isinstance(device, torch.device):
        if device.index is not None:
            return str(device)
        return device.type if device.type == "cpu" else f"{device.type}:{get_device_id(device.type)}"
    device_type = get_device_type()
    return device_type if device_type == "cpu" else f"{device_type}:{get_device_id(device_type)}"


def synchronize(device_type: str | None = None) -> None:
    resolved_type = get_device_type() if device_type is None else device_type
    if resolved_type == "cpu":
        return
    get_torch_device(resolved_type).synchronize()


def empty_cache(device_type: str | None = None) -> None:
    resolved_type = get_device_type() if device_type is None else device_type
    if resolved_type == "cpu":
        return
    get_torch_device(resolved_type).empty_cache()


def get_mem_info(device: Any = None):
    resolved_name = get_device_name(device)
    resolved_type = parse_device_type(resolved_name)
    if resolved_type == "cpu":
        raise RuntimeError("Memory info is not available for cpu.")
    return get_torch_device(resolved_type).mem_get_info(resolved_name)


def enable_high_precision_for_bf16() -> None:
    if IS_CUDA_AVAILABLE:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if IS_NPU_AVAILABLE:
        torch.npu.matmul.allow_tf32 = False
        torch.npu.matmul.allow_bf16_reduced_precision_reduction = False


def parse_device_type(device: Any) -> str:
    if isinstance(device, torch.device):
        return device.type
    if isinstance(device, str):
        if device.startswith("cuda"):
            return "cuda"
        if device.startswith("npu"):
            return "npu"
        return "cpu"
    return get_device_type()


def parse_nccl_backend(device_type: str) -> str:
    if device_type == "cuda":
        return "nccl"
    if device_type == "npu":
        return "hccl"
    raise RuntimeError(f"No available distributed communication backend found on device type {device_type}.")


def get_available_device_type() -> str:
    return get_device_type()
