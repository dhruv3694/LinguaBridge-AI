"""Indic Parler-TTS (notebook cell 90). Lazy heavy imports."""
from __future__ import annotations
import time
from pathlib import Path
from typing import Any, Dict, Optional

from config.model_config import PARLER_TTS_MODEL_ID
from config.settings import OUTPUTS_DIR
from modules.hf_auth import get_huggingface_token

MODEL_NAME = PARLER_TTS_MODEL_ID
SUPPORTED_SPEAKERS = {
    "Hindi": ["Divya", "Rohit", "Aman", "Rani"],
    "Marathi": ["Sunita", "Sanjay", "Nikhil", "Radha", "Varun", "Isha"],
}
SPEAKER_DEFAULTS = {"Hindi": "Divya", "Marathi": "Sunita"}
VOICE_STYLES = {
    "Normal": "speaks at a moderate speed and pitch in a clear, neutral tone",
    "Slow": "speaks slowly with clear pronunciation and a calm tone",
    "Expressive": "speaks with an expressive, engaging and natural tone",
}
_model = _tokenizer = _description_tokenizer = _device = None


def load_tts(token: Optional[str] = None):
    global _model, _tokenizer, _description_tokenizer, _device
    if _model is not None:
        return True
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    tok = token if token is not None else get_huggingface_token()
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if _device == "cuda" else torch.float32
    _model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=dtype, token=tok).to(_device)
    _model.eval()
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=tok)
    text_enc = getattr(_model.config, "text_encoder", None)
    enc_name = getattr(text_enc, "_name_or_path", None) if text_enc else None
    _description_tokenizer = AutoTokenizer.from_pretrained(enc_name or MODEL_NAME, token=tok)
    return True


def generate_speech(text, language="Hindi", speaker=None, style="Normal", output_path=None, token: Optional[str] = None) -> Dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        return {"success": False, "error": "Text cannot be empty."}
    if language not in SUPPORTED_SPEAKERS:
        return {"success": False, "error": "Only Hindi and Marathi are supported."}
    if style not in VOICE_STYLES:
        return {"success": False, "error": "Invalid speech style."}
    speaker = speaker or SPEAKER_DEFAULTS[language]
    if speaker not in SUPPORTED_SPEAKERS[language]:
        return {"success": False, "error": f"Unsupported {language} speaker: {speaker}"}
    try:
        import numpy as np
        import soundfile as sf
        import torch
        load_tts(token=token)
        description = f"{speaker} {VOICE_STYLES[style]}. The recording is of very high quality, with very clear audio and no background noise."
        prompt_inputs = _tokenizer(text.strip(), return_tensors="pt").to(_device)
        description_inputs = _description_tokenizer(description, return_tensors="pt").to(_device)
        start_time = time.time()
        with torch.inference_mode():
            generation = _model.generate(
                input_ids=description_inputs.input_ids, attention_mask=description_inputs.attention_mask,
                prompt_input_ids=prompt_inputs.input_ids, prompt_attention_mask=prompt_inputs.attention_mask,
            )
        audio = generation.detach().float().cpu().numpy().squeeze()
        if audio.ndim != 1 or audio.size == 0:
            raise ValueError("Model returned invalid audio output.")
        output_dir = OUTPUTS_DIR / "audio"; output_dir.mkdir(parents=True, exist_ok=True)
        if output_path is None:
            output_path = output_dir / f"{language.lower()}_{int(time.time())}.wav"
        else:
            output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
        sample_rate = getattr(_model.config, "sampling_rate", 44100)
        sf.write(str(output_path), audio, sample_rate)
        latency = round(time.time() - start_time, 3)
        return {"success": True, "audio_path": str(output_path), "language": language, "speaker": speaker,
                "style": style, "model": MODEL_NAME, "device": _device, "sample_rate": sample_rate,
                "duration_seconds": round(len(audio) / sample_rate, 2), "latency_seconds": latency}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def unload_tts():
    global _model, _tokenizer, _description_tokenizer, _device
    _model = _tokenizer = _description_tokenizer = _device = None
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception:
        pass
