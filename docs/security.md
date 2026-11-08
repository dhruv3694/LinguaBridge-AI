# Security

## Hugging Face tokens
- Tokens are entered in the **Hugging Face Authentication** Streamlit page (masked input).
- Default storage is **Streamlit session state only** (`hf_token`). The app does **not** write tokens to `.env`, source code, JSON, CSV, logs, or outputs.
- Disconnect clears the session token. Session restarts require re-entry.
- The original Colab notebook contained a plaintext token in a markdown/code context; it must be treated as **compromised** and **revoked**. Do not copy it.

## Local files
- `.env` is sanitized and gitignored. `.env.example` contains no real token.
- Model weights are stored in the Hugging Face cache (user profile), not committed to git.
- Generated audio/images/CSVs may be written under `outputs/`; review before sharing.

## Model/runtime risks
- Large models (Qwen 1.5B, Indic Parler-TTS, BLIP) can consume significant RAM/VRAM. Use lazy loading and close other GPU workloads on 6 GB VRAM systems.

## Safe cleanup
- Clear session token via the app Disconnect button.
- Remove generated outputs manually from `outputs/` if needed.
- Do not auto-delete Hugging Face cache unless you intend to re-download models.
