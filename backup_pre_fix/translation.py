"""IndicTrans2 experiment (notebook cells 114,122,124). Lazy imports."""
from __future__ import annotations
from typing import Any, Dict, Optional
from config.model_config import INDICTRANS2_MODEL_ID
from modules.hf_auth import get_huggingface_token

MODEL_NAME = INDICTRANS2_MODEL_ID
LANGUAGE_CODES = {"Hindi": "hin_Deva", "Marathi": "mar_Deva"}
_tokenizer = _model = _processor = None
DEVICE = None


def load_translation_model(token: Optional[str] = None):
    global _tokenizer, _model, _processor, DEVICE
    if _model is not None:
        return True
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    try:
        from IndicTransToolkit.processor import IndicProcessor
    except Exception as exc:
        raise RuntimeError(f"IndicTransToolkit is required for IndicTrans2: {exc}") from exc
    tok = token if token is not None else get_huggingface_token()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, token=tok)
    _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True, low_cpu_mem_usage=True, token=tok)
    _model = _model.to(DEVICE); _model.eval()
    _processor = IndicProcessor(inference=True)
    return True


def translate_text(text: str, target_language: str, token: Optional[str] = None) -> Dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        return {"success": False, "translated": None, "error": "Input text cannot be empty."}
    if target_language not in LANGUAGE_CODES:
        return {"success": False, "translated": None, "error": f"Unsupported language: {target_language}"}
    try:
        import torch
        if _model is None:
            load_translation_model(token=token)
        target_code = LANGUAGE_CODES[target_language]
        batch = _processor.preprocess_batch([text.strip()], src_lang="eng_Latn", tgt_lang=target_code)
        inputs = _tokenizer(batch, truncation=True, padding="longest", return_tensors="pt", return_attention_mask=True).to(DEVICE)
        with torch.inference_mode():
            generated_tokens = _model.generate(**inputs, use_cache=True, min_length=1, max_length=256, num_beams=5, num_return_sequences=1)
        generated_tokens = _tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        translations = _processor.postprocess_batch(generated_tokens, lang=target_code)
        return {"success": True, "translated": translations[0], "translated_text": translations[0],
                "source_language": "English", "target_language": target_language, "model": MODEL_NAME, "error": None}
    except Exception as exc:
        return {"success": False, "translated": None, "error": str(exc)}


def unload_translation_model():
    global _tokenizer, _model, _processor
    _tokenizer = _model = _processor = None
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception:
        pass
