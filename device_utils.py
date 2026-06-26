import os

import torch


def _mps_available():
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def get_torch_device():
    """
    Select the fastest available PyTorch device.

    SICKLESIGHT_DEVICE can override auto selection with cpu, cuda, cuda:0, or mps.
    Auto mode prefers CUDA, then Apple MPS, then CPU.
    """
    requested = os.environ.get("SICKLESIGHT_DEVICE", "auto").strip().lower()
    if requested and requested != "auto":
        if requested.startswith("cuda"):
            if torch.cuda.is_available():
                return torch.device(requested)
            print(f"Warning: SICKLESIGHT_DEVICE={requested} requested but CUDA is unavailable; falling back.")
        elif requested == "mps":
            if _mps_available():
                return torch.device("mps")
            print("Warning: SICKLESIGHT_DEVICE=mps requested but MPS is unavailable; falling back.")
        elif requested == "cpu":
            return torch.device("cpu")
        else:
            print(f"Warning: Unknown SICKLESIGHT_DEVICE={requested!r}; using auto device selection.")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_ultralytics_device(torch_device=None):
    """Return the device value expected by Ultralytics predict/track calls."""
    torch_device = torch_device or get_torch_device()
    if torch_device.type == "cuda":
        return torch_device.index if torch_device.index is not None else 0
    if torch_device.type == "mps":
        return "mps"
    return "cpu"


def get_cellpose_gpu_enabled(torch_device=None):
    """Cellpose exposes GPU usage as a boolean in this codebase."""
    torch_device = torch_device or get_torch_device()
    return torch_device.type in {"cuda", "mps"}


def print_device_summary(prefix="Using"):
    torch_device = get_torch_device()
    yolo_device = get_ultralytics_device(torch_device)
    cellpose_gpu = get_cellpose_gpu_enabled(torch_device)
    print(f"{prefix} PyTorch device: {torch_device}")
    print(f"{prefix} Ultralytics device: {yolo_device}")
    print(f"{prefix} Cellpose gpu={cellpose_gpu}")
    return torch_device
