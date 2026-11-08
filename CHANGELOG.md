# Changelog

## Runtime recovery
- Standardized Windows startup on the validated Python 3.11 CUDA/TTS environment and added dependency checks to the launcher.
- Made dashboard readiness checks opt-in and unload each model after its smoke test to avoid holding all weights in VRAM.
- Fixed BLIP captioning to accept PIL images, including the dashboard readiness check.
- Removed the experimental IndicTrans2/IndicTransToolkit feature; MarianMT and Qwen remain the translation engines, with Indic Parler-TTS as the speech engine.

## Notebook-first reconstruction
- Rebuilt project from `notebooks/LinguaBridge_AI_Development.ipynb` (sole implementation source).
- Implemented notebook model allowlist: MarianMT hi/mr, Qwen2.5-1.5B-Instruct, BLIP, Indic Parler-TTS, IndicTrans2 (experimental).
- Replaced IDE TTS (Vyom) with notebook Indic Parler-TTS.
- Added session-only Hugging Face authentication; removed plaintext token persistence.
- Reconstructed modules: translator, prompt_engineering, llm_translator, evaluation, image_caption, image_translation, speech, translation (IndicTrans2), bias_evaluation.
- Built Streamlit multipage UI covering notebook features and U1–U5 documentation.
- Added docs: model inventory, U1–U5 matrix, security, architecture, licenses, dataset docs, model card, limitations.
- Preserved original notebook and datasets; sanitized `.env`.
