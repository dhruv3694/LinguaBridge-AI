# LinguaBridge AI (P19) — Notebook-First Reconstruction

Open-source English → Hindi/Marathi translation platform based on `notebooks/LinguaBridge_AI_Development.ipynb`.

## Notebook model allowlist (only these HF models)
1. `Helsinki-NLP/opus-mt-en-hi`
2. `Helsinki-NLP/opus-mt-en-mr`
3. `Qwen/Qwen2.5-1.5B-Instruct`
4. `Salesforce/blip-image-captioning-base`
5. `ai4bharat/indic-parler-tts` (TTS — **not** Vyom)

## Features (Streamlit)
- Hugging Face Authentication (session-only token)
- Translator: MarianMT or Qwen (zero/few/role + styles)
- Prompt Engineering Studio (U1)
- MarianMT vs Qwen comparison + SacreBLEU/chrF when valid references exist
- Image captioning (BLIP) + translation
- Text-to-Speech (Indic Parler-TTS, notebook speakers/styles)
- Gender bias evaluation (heuristic markers, human review required)
- U1–U5 documentation and model inventory

## Run (Windows)

This project was validated with **Python 3.11**. Use the project environment
and launcher so Streamlit, PyTorch/CUDA, and TTS dependencies come from the
same interpreter:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_app.ps1
```

The launcher uses `D:\LinguaBridge_Environment\venv311` by default, checks the
Python version and required dependencies, and can be run without Administrator
privileges. To use another Python 3.11 environment:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_app.ps1 -VenvRoot "D:\path\to\venv311"
```

For a fresh setup (including the matching PyTorch CUDA wheel and optional
Parler-TTS dependency), follow the [Windows installation guide](docs/installation_guide.md).
Do not launch with a different global Python; auth can work even when that
interpreter lacks PyTorch and model inference cannot.

## Hugging Face token
Use the **Hugging Face Authentication** page. Tokens are stored **only in session state** and never written to disk by the app.

## Tests
```powershell
python -m pytest -q
```
Unit/mock tests do not prove real model inference. Real inference requires model downloads, dependencies, and possibly HF access.

## Hardware target
Ryzen 7 7735HS, 16 GB RAM, RTX 4050 6 GB — use lazy loading; run heavy models sequentially.

## Security
See `docs/security.md`. The original notebook contained a plaintext token; treat it as compromised and revoke it.
