# Dependency Compatibility Notes

## Critical pairs
- **Python 3.14**: verify wheel availability for `torch`, `tokenizers`, `sentencepiece`, and `parler-tts`. If a package lacks wheels, use a supported Python version (3.11 is validated here).
- **PyTorch + Transformers**: install compatible versions; avoid blind upgrades.
- **Hugging Face Hub + Transformers**: `huggingface_hub>=0.26` used; ensure `transformers` supports `token=`/auth API in your installed version.
- **parler-tts** (git): depends on specific transformers/torch behavior; test after install.

## This environment (observed during repair)
- Python 3.11, Windows AMD64, RTX 4050 (6 GB VRAM), CUDA-enabled PyTorch 2.11.0+cu128.
- The project environment has Transformers 4.46.1 and Parler-TTS installed. Python 3.14 on PATH is not the validated environment and does not have PyTorch.

## Strategy
1. Install compatible PyTorch for your platform.
2. Install `requirements-core.txt`.
3. Run `python -m pip check` and `python scripts/check_environment.py`.
4. Install optional deps only for features you need.
