"""Indic Parler-TTS synthesis with lazy loading and safe fallbacks."""
from __future__ import annotations

import time
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from config.model_config import PARLER_TTS_MODEL_ID
from config.settings import OUTPUTS_DIR, configure_model_storage
from modules.hf_auth import get_huggingface_token

MODEL_NAME = PARLER_TTS_MODEL_ID
MODEL_REPOSITORY_URL = f"https://huggingface.co/{MODEL_NAME}"
SPEECH_MODELS = {
    "Hindi": {"model_id": PARLER_TTS_MODEL_ID, "default_speaker": "Divya"},
    "Marathi": {"model_id": PARLER_TTS_MODEL_ID, "default_speaker": "Sunita"},
}
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
SUPPORTED_LANGUAGES = tuple(SPEECH_MODELS.keys())

_model = _tokenizer = _description_tokenizer = _device = None
_model_load_lock = threading.Lock()


def _normalized_language(language: Optional[str]) -> str:
    if not language:
        return "Hindi"
    normalized = str(language).strip().title()
    if normalized not in SUPPORTED_SPEAKERS:
        raise ValueError(f"Unsupported language: {language}")
    return normalized


def _sanitize_error(message: str, token: Optional[str] = None) -> str:
    if not message:
        return "Speech generation failed."
    for secret in (token, get_huggingface_token()):
        if secret:
            message = message.replace(secret, "[REDACTED]")
    message = message.replace("\n", " ")
    if "no space left" in message.lower() or "errno 28" in message.lower():
        return (
            "The TTS model load ran out of storage. Model downloads and temporary files are now "
            "configured to use D:\\LinguaBridge_Environment\\hf-cache; verify that drive has free space "
            "and restart Streamlit before retrying."
        )
    if any(term in message.lower() for term in ("401", "403", "gated repo", "access to model")):
        return f"Access to the TTS model was denied. Verify that your Hugging Face account has accepted the model terms: {MODEL_REPOSITORY_URL}. Details: {message}"
    if "out of memory" in message.lower() or "cuda" in message.lower() and "memory" in message.lower():
        return "Insufficient GPU memory for speech generation. Try a shorter sentence or close other GPU workloads."
    return message


def _ensure_tensor_audio(audio: Any):
    import numpy as np
    import torch

    if isinstance(audio, dict):
        audio = audio.get("audio")
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().float().numpy()
    if isinstance(audio, np.ndarray):
        arr = np.asarray(audio, dtype=np.float32)
    else:
        arr = np.asarray(audio, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = np.mean(arr, axis=0 if arr.shape[0] <= 2 else 1)
    if arr.ndim != 1:
        raise ValueError(f"Model returned audio with unsupported shape {arr.shape}.")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError("Model returned empty audio data.")
    return arr


def load_tts(token: Optional[str] = None):
    global _model, _tokenizer, _description_tokenizer, _device
    if _model is not None:
        return True
    with _model_load_lock:
        if _model is not None:
            return True
        configure_model_storage()
        tok = token if token is not None else get_huggingface_token()
        try:
            import torch
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer

            _device = "cpu"
            if torch.cuda.is_available():
                free_bytes, _ = torch.cuda.mem_get_info()
                if free_bytes >= 4 * 1024**3:
                    _device = "cuda"
            dtype = torch.float16 if _device == "cuda" else torch.float32
            model = ParlerTTSForConditionalGeneration.from_pretrained(
                MODEL_NAME,
                torch_dtype=dtype,
                token=tok,
            ).to(_device)
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=tok)
            text_encoder = getattr(model.config, "text_encoder", None)
            encoder_name = getattr(text_encoder, "_name_or_path", None) if text_encoder else None
            description_tokenizer = AutoTokenizer.from_pretrained(encoder_name or MODEL_NAME, token=tok)
        except Exception as exc:
            raise RuntimeError(_sanitize_error(str(exc), tok)) from exc

        _model = model
        _tokenizer = tokenizer
        _description_tokenizer = description_tokenizer
        return True


class SpeechSynthesizer:
    def __init__(
        self,
        output_dir: Optional[Path | str] = None,
        default_device: Optional[str] = None,
        token: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir) if output_dir is not None else OUTPUTS_DIR / "audio"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.default_device = default_device or ("cuda" if self._cuda_available() else "cpu")
        self.token = token
        self.model = None
        self.tokenizer = None
        self.description_tokenizer = None
        self.device = self.default_device
        self.model_name = MODEL_NAME

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def is_model_loaded(self):
        return self.model is not None and self.tokenizer is not None

    def load_model(self, language: str = "Hindi") -> Tuple[Any, Any]:
        normalized = _normalized_language(language)
        if self.model is not None and self.tokenizer is not None:
            return self.model, self.tokenizer
        try:
            import torch
            load_tts(token=self.token)
            self.model = _model
            self.tokenizer = _tokenizer
            self.description_tokenizer = _description_tokenizer
            self.device = _device or self.default_device
            self.model_name = MODEL_NAME
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            return self.model, self.tokenizer
        except Exception as exc:
            raise RuntimeError(_sanitize_error(str(exc))) from exc

    def synthesize(
        self,
        text: str,
        language: str = "Hindi",
        speaker: Optional[str] = None,
        style: str = "Normal",
        output_path: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            return {"success": False, "audio_path": None, "error": "Text cannot be empty."}
        try:
            language = _normalized_language(language)
        except ValueError as exc:
            return {"success": False, "audio_path": None, "error": str(exc)}
        if style not in VOICE_STYLES:
            return {"success": False, "audio_path": None, "error": "Invalid speech style."}
        speaker = speaker or SPEAKER_DEFAULTS[language]
        if speaker not in SUPPORTED_SPEAKERS[language]:
            return {"success": False, "audio_path": None, "error": f"Unsupported {language} speaker: {speaker}"}
        stage = "loading model"
        try:
            started = time.perf_counter()
            model, tokenizer = self.load_model(language)
            load_seconds = round(time.perf_counter() - started, 3)

            stage = "preparing text and voice description"
            if self.description_tokenizer is None:
                raise RuntimeError("The Parler-TTS description tokenizer was not loaded.")
            description = (
                f"{speaker} {VOICE_STYLES[style]}. "
                "The recording is of very high quality, with very clear audio and no background noise."
            )
            prompt_inputs = tokenizer(text.strip(), return_tensors="pt").to(self.device)
            description_inputs = self.description_tokenizer(description, return_tensors="pt").to(self.device)

            stage = "generating speech"
            import torch
            if self.device == "cuda":
                torch.cuda.synchronize()
            generation_started = time.perf_counter()
            with torch.inference_mode():
                generation = model.generate(
                    input_ids=description_inputs.input_ids,
                    attention_mask=description_inputs.attention_mask,
                    prompt_input_ids=prompt_inputs.input_ids,
                    prompt_attention_mask=prompt_inputs.attention_mask,
                )
            if self.device == "cuda":
                torch.cuda.synchronize()
            generation_seconds = round(time.perf_counter() - generation_started, 3)

            stage = "converting generated waveform"
            audio_arr = _ensure_tensor_audio(generation)
            sample_rate = int(getattr(model.config, "sampling_rate", 0))
            if sample_rate <= 0:
                raise ValueError(f"Model returned an invalid sample rate: {sample_rate}.")

            stage = "saving and validating WAV"
            output_path = Path(output_path) if output_path is not None else (
                self.output_dir / f"{language.lower()}_{uuid.uuid4().hex[:12]}.wav"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            import soundfile as sf
            sf.write(str(output_path), audio_arr.astype("float32"), sample_rate)
            info = sf.info(str(output_path))
            if info.format != "WAV" or info.frames <= 0 or info.samplerate != sample_rate:
                raise ValueError("Saved audio failed WAV format, frame-count, or sample-rate validation.")
            sf.read(str(output_path), frames=min(info.frames, sample_rate))
            latency = round(time.perf_counter() - started, 3)
            duration = round(len(audio_arr) / sample_rate, 3)
            if output_path.stat().st_size == 0:
                raise ValueError("Saved audio file is empty.")
            return {
                "success": True,
                "audio_path": str(output_path),
                "language": language,
                "speaker": speaker,
                "style": style,
                "model": self.model_name,
                "device": self.device,
                "sample_rate": sample_rate,
                "duration_seconds": duration,
                "latency_seconds": latency,
                "load_seconds": load_seconds,
                "generation_time_seconds": generation_seconds,
                "model_name": self.model_name,
            }
        except MemoryError:
            return {"success": False, "audio_path": None, "error": "Insufficient memory for speech generation. Try a shorter sentence or switch to CPU."}
        except Exception as exc:
            return {"success": False, "audio_path": None, "error": f"Failed during {stage}: {_sanitize_error(str(exc))}"}


def generate_speech(text, language="Hindi", speaker=None, style="Normal", output_path=None, token: Optional[str] = None) -> Dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        return {"success": False, "audio_path": None, "error": "Text cannot be empty."}
    try:
        language = _normalized_language(language)
    except ValueError as exc:
        return {"success": False, "audio_path": None, "error": str(exc)}
    if style not in VOICE_STYLES:
        return {"success": False, "audio_path": None, "error": "Invalid speech style."}
    try:
        out_dir = Path(output_path).parent if output_path is not None else OUTPUTS_DIR / "audio"
        return SpeechSynthesizer(output_dir=out_dir, token=token).synthesize(
            text,
            language=language,
            speaker=speaker,
            style=style,
            output_path=output_path,
        )
    except Exception as exc:
        return {"success": False, "audio_path": None, "error": _sanitize_error(str(exc))}


def unload_tts():
    global _model, _tokenizer, _description_tokenizer, _device
    _model = _tokenizer = _description_tokenizer = _device = None
    from services.model_manager import release_model_memory

    release_model_memory()
