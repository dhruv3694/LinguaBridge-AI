"""Notebook-derived Hugging Face model allowlist and generation defaults."""
from __future__ import annotations

# Sole source: LinguaBridge_AI_Development.ipynb (163 cells)

MARIAN_MODELS = {
    "Hindi": "Helsinki-NLP/opus-mt-en-hi",
    "Marathi": "Helsinki-NLP/opus-mt-en-mr",
}

QWEN_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
BLIP_MODEL_ID = "Salesforce/blip-image-captioning-base"
PARLER_TTS_MODEL_ID = "ai4bharat/indic-parler-tts"

MODEL_ALLOWLIST = [
    MARIAN_MODELS["Hindi"],
    MARIAN_MODELS["Marathi"],
    QWEN_MODEL_ID,
    BLIP_MODEL_ID,
    PARLER_TTS_MODEL_ID,
]

SUPPORTED_LANGUAGES = ("Hindi", "Marathi")
