"""Environment and dependency diagnostic for LinguaBridge AI.

Checks the active Python interpreter, core/optional dependencies, and GPU status.
Does not crash on missing optional dependencies.
"""
from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class CheckResult:
    name: str
    status: str  # ok | missing | conflict | optional-missing
    detail: str = ""


CORE = [
    "torch",
    "transformers",
    "huggingface_hub",
    "tokenizers",
    "safetensors",
    "accelerate",
    "sentencepiece",
    "streamlit",
    "pandas",
    "numpy",
    "PIL",
    "sacrebleu",
]

OPTIONAL = [
    ("soundfile", "TTS WAV output"),
    ("parler_tts", "Indic Parler-TTS (notebook TTS model)"),
    ("matplotlib", "Bias/evaluation charts"),
    ("sklearn", "Notebook dependency (used if required)"),
    ("sacremoses", "MarianMT/tokenization support in some setups"),
    ("dotenv", "Local env loading (python-dotenv)"),
]

IMPORT_NAMES = {
    "PIL": "PIL",
    "sklearn": "sklearn",
    "dotenv": "dotenv",
}


def _version(mod_name: str) -> str:
    try:
        mod = importlib.import_module(mod_name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return ""


def check_import(name: str, optional_reason: Optional[str] = None) -> CheckResult:
    mod_name = IMPORT_NAMES.get(name, name)
    try:
        importlib.import_module(mod_name)
        ver = _version(mod_name)
        return CheckResult(name, "ok", ver)
    except Exception as exc:
        status = "optional-missing" if optional_reason else "missing"
        detail = f"{optional_reason + ': ' if optional_reason else ''}{exc.__class__.__name__}: {exc}"
        return CheckResult(name, status, detail)


def check_torch_cuda() -> list[CheckResult]:
    results = []
    try:
        import torch

        results.append(CheckResult("torch.cuda", "ok", f"available={torch.cuda.is_available()} version={torch.version.cuda}"))
        if torch.cuda.is_available():
            results.append(CheckResult("torch.gpu", "ok", torch.cuda.get_device_name(0)))
            try:
                mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                results.append(CheckResult("torch.vram_gb", "ok", f"{mem:.2f}"))
            except Exception as exc:
                results.append(CheckResult("torch.vram_gb", "optional-missing", str(exc)))
        else:
            results.append(CheckResult("torch.gpu", "optional-missing", "CUDA unavailable; CPU fallback possible"))
    except Exception as exc:
        results.append(CheckResult("torch.cuda", "missing", str(exc)))
    return results


def main() -> int:
    print("=== LinguaBridge AI Environment Diagnostic ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"Platform: {platform.platform()} {platform.machine()}")
    print(f"pip/python same env: run `python -m pip` and `python -m streamlit`")
    print()
    results = [check_import(n) for n in CORE]
    results += [check_import(n, reason) for n, reason in OPTIONAL]
    results += check_torch_cuda()

    width = max(len(r.name) for r in results)
    for r in results:
        icon = {"ok": "OK", "missing": "MISSING", "conflict": "CONFLICT", "optional-missing": "OPTIONAL"}.get(r.status, r.status)
        print(f"{r.name:<{width}}  {icon:<9}  {r.detail}")
    print()
    missing_core = [r for r in results if r.status == "missing" and r.name in CORE]
    if missing_core:
        print("Core missing:", ", ".join(r.name for r in missing_core))
        if "torch" in {r.name for r in missing_core}:
            print("PyTorch is intentionally installed separately; install a CPU or CUDA wheel for this interpreter first.")
            print("For the validated Python 3.11/CUDA setup, see docs/installation_guide.md.")
        print("Then install the remaining core dependencies in the same interpreter:")
        print("python -m pip install -r requirements-core.txt")
        print("For GPU (RTX/CUDA 12.x typical): install PyTorch per https://pytorch.org/get-started/locally/ first.")
        return 1
    print("Core dependencies present. Optional features may still need packages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
