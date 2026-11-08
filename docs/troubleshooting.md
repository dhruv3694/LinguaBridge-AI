# Troubleshooting

## `ModuleNotFoundError: No module named 'torch'`
- PyTorch is not installed in the interpreter running Streamlit. Authentication can still work, but model inference cannot.
- Fix: run `.\scripts\run_app.ps1` to use the configured Python 3.11 environment, or follow `docs/installation_guide.md` to install PyTorch in the same environment as Streamlit.
- Diagnostic: `python scripts/check_environment.py`.

## `ModuleNotFoundError: No module named 'huggingface_hub'`
- Install: `python -m pip install huggingface_hub transformers` (same interpreter).
- Auth page shows dependency guidance if missing.

## `No space left on device` during pip install
- CUDA torch wheels are very large. Free disk space, move pip cache/temp (`$env:PIP_CACHE_DIR`, `$env:TEMP`), or install CPU torch if GPU acceleration is not required immediately.

## `No space left on device` during Parler-TTS model load
- The app configures Hugging Face model downloads and Python temporary files under `D:\LinguaBridge_Environment\hf-cache` by default, before model loading.
- Restart Streamlit after changing the cache configuration. Confirm that D: has free space and that `HF_HOME`, `HF_HUB_CACHE`, and `TEMP` resolve to D: in the running process.
- The project pins `sentencepiece<0.2.3` because SentencePiece 0.2.3 generated Protobuf code that is incompatible with the Protobuf 6 runtime required by current Streamlit.
- Keep the Streamlit-compatible Protobuf version; do not downgrade Protobuf to fix this SentencePiece error.
- The first cached-model load can take a few minutes. TTS should report a load/generation error rather than create a success result without a valid WAV.

## `pip` and `streamlit` use different environments
- Avoid global `streamlit.exe`. Use `python -m streamlit run app.py`.

## CUDA unavailable
- App should run on CPU (slower). Check `torch.cuda.is_available()` and NVIDIA driver/CUDA-compatible wheel.

## Out of memory (GPU)
- RTX 4050 has 6 GB VRAM. Load models one at a time, use smaller batches, close other GPU apps, or use CPU fallback.

## Model download/auth errors
- First run downloads weights to Hugging Face cache. Some models may require accepting terms or a token (session-only in app).

## Parler-TTS missing
- Optional dependencies; MarianMT/Qwen/BLIP/auth can still be used if their deps are installed. Install `requirements-optional.txt` when ready.
