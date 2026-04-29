import os

import torch


def _cpu_only_from_env():
    return os.environ.get("FORCE_CPU", "").lower() in ("1", "true", "yes") or os.environ.get(
        "TORCH_FORCE_CPU", ""
    ).lower() in ("1", "true", "yes")


def get_torch_device() -> torch.device:
    """For inference and dry-run. Trainer selects its own device."""
    if _cpu_only_from_env():
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            return f"cuda ({torch.cuda.get_device_name(0)})"
        except Exception:
            return "cuda"
    if device.type == "mps":
        return "mps (Apple GPU)"
    return str(device)
