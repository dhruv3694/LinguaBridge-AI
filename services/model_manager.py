"""Runtime status helpers (lazy torch import)."""
from __future__ import annotations


def release_model_memory() -> None:
    """Collect model references before returning cached GPU memory to CUDA."""
    import gc

    gc.collect()
    try:
        import torch
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def runtime_status() -> dict:
    try:
        import torch
        cuda = torch.cuda.is_available()
        return {
            "device": "cuda" if cuda else "cpu",
            "cuda_available": cuda,
            "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
            "hf_token_present": _token_present(),
        }
    except Exception:
        return {"device": "cpu", "cuda_available": False, "gpu_name": None, "hf_token_present": _token_present()}


def _token_present():
    try:
        from modules.hf_auth import get_huggingface_token
        return bool(get_huggingface_token())
    except Exception:
        return False
