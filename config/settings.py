"""Central settings (no heavy imports at module load)."""
from __future__ import annotations
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
ASSETS_DIR = BASE_DIR / "assets"
DOCS_DIR = BASE_DIR / "docs"
NOTEBOOK_PATH = BASE_DIR / "notebooks" / "LinguaBridge_AI_Development.ipynb"


def configure_model_storage() -> None:
    """Keep model downloads and temporary extraction off the usually smaller C: drive."""
    default_cache = Path(r"D:\LinguaBridge_Environment\hf-cache")
    if not default_cache.parent.exists():
        default_cache = Path.home() / ".cache" / "huggingface"

    cache_root = Path(os.environ.get("HF_HOME") or os.environ.get("LINGUABRIDGE_CACHE_DIR") or default_cache)
    os.environ.setdefault("HF_HOME", str(cache_root))
    hub_cache = Path(os.environ.get("HF_HUB_CACHE") or cache_root / "hub")
    os.environ.setdefault("HF_HUB_CACHE", str(hub_cache))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))

    temp_dir = Path(os.environ.get("LINGUABRIDGE_TEMP_DIR") or cache_root / "tmp")
    os.environ["TMPDIR"] = str(temp_dir)
    os.environ["TEMP"] = str(temp_dir)
    os.environ["TMP"] = str(temp_dir)

    for path in (cache_root, hub_cache, cache_root / "torch", temp_dir):
        path.mkdir(parents=True, exist_ok=True)

    # tempfile may already be imported by Streamlit before a page loads.
    import tempfile
    tempfile.tempdir = str(temp_dir)


configure_model_storage()

for _d in (
    DATA_DIR, OUTPUTS_DIR / "translations", OUTPUTS_DIR / "evaluations",
    OUTPUTS_DIR / "audio", OUTPUTS_DIR / "captions", OUTPUTS_DIR / "bias",
    OUTPUTS_DIR / "visualizations", ASSETS_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)


def get_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


DEVICE = get_device()
