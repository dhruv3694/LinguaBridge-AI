# Architecture (Notebook-First Reconstruction)
Source: `notebooks/LinguaBridge_AI_Development.ipynb`.

## Layers
- **config/**: paths, device, notebook model allowlist.
- **modules/**: notebook-derived inference/evaluation logic (MarianMT, prompt engineering, Qwen, BLIP, image pipeline, Indic Parler-TTS, bias evaluation).
- **services/**: runtime/auth helpers (lightweight; lazy loading lives in modules).
- **frontend/pages/**: Streamlit pages; each calls module functions (no dummy inference).
- **app.py**: navigation, sidebar auth status.
- **data/**: benchmark and bias datasets.
- **outputs/**: generated CSV/audio/images/worksheets.

## Data flow examples
- Translation (MarianMT): UI → `modules.translator.translate_text` → MarianMT → result.
- Translation (Qwen): UI → prompt module → `modules.llm_translator.translate_with_llm` → Qwen chat generation.
- Image pipeline: upload → temp file → BLIP caption → selected translator → UI.
- Bias: dataset → per-model translation functions → marker detection → summaries/exports.
- TTS: text → Indic Parler-TTS → WAV in `outputs/audio/` → Streamlit audio/download.

## Memory strategy (6 GB VRAM target)
Lazy load per model, cache globals/singletons in modules, CPU fallback where supported, sequential heavy tasks in UI, clear guidance on OOM. Dashboard readiness checks release each model after its smoke test so the checks do not keep all models resident on the GPU.
