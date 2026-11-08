"""
LinguaBridge AI - Qwen LLM Translation Engine (notebook cell 26).
Model: Qwen/Qwen2.5-1.5B-Instruct. Greedy decoding as notebook.
Memory-aware loading: tries CUDA first, then falls back to device_map="auto"
with offload on D: so Qwen still runs when 6 GB VRAM / page file is constrained.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional
from pathlib import Path

from config.model_config import QWEN_MODEL_ID
from config.settings import OUTPUTS_DIR
from modules.hf_auth import get_huggingface_token
from modules.prompt_engineering import build_translation_prompt

SUPPORTED_LANGUAGES = ["Hindi", "Marathi"]
MODEL_NAME = QWEN_MODEL_ID

_tokenizer = None
_model = None
_device = None
_load_mode = None  # "cuda" | "auto_offload" | "cpu"


def dependency_error() -> Optional[str]:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return None
    except Exception as exc:
        return (
            f"{exc.__class__.__name__}: {exc}. Install compatible PyTorch and Transformers in the same "
            "Python environment used by Streamlit (see docs/installation_guide.md)."
        )


def _offload_dir() -> Path:
    d = OUTPUTS_DIR / "model_offload" / "qwen"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_llm(token: Optional[str] = None):
    """
    Load Qwen once and reuse it.

    Strategy (memory-aware for RTX 4050 6 GB VRAM):
    1. Direct CUDA load (float16) — fastest when VRAM/page file allow.
    2. device_map="auto" with offload folder on D: — keeps Qwen usable when
       direct CUDA allocation is constrained (Windows paging file too small).
    """
    global _tokenizer, _model, _device, _load_mode
    if _model is not None:
        return True
    dep = dependency_error()
    if dep:
        raise RuntimeError(dep)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = token if token is not None else get_huggingface_token()
    tok_kwargs = {"trust_remote_code": False}
    if tok:
        tok_kwargs["token"] = tok

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, **tok_kwargs)

    load_attempts: list[str] = []
    last_error: Optional[Exception] = None

    # Attempt 1: direct CUDA (notebook-intended when resources allow)
    if torch.cuda.is_available():
        try:
            load_kwargs = {
                "torch_dtype": torch.float16,
                "trust_remote_code": False,
                "low_cpu_mem_usage": True,
            }
            if tok:
                load_kwargs["token"] = tok
            _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
            _model.to("cuda")
            _model.eval()
            _device = "cuda"
            _load_mode = "cuda"
            return True
        except Exception as exc:
            last_error = exc
            load_attempts.append(f"direct CUDA failed: {exc}")
            _model = None
            _device = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Attempt 2: device_map="auto" with offload to D: (CPU/offloaded fallback)
    try:
        load_kwargs = {
            "torch_dtype": torch.float16,
            "trust_remote_code": False,
            "low_cpu_mem_usage": True,
            "device_map": "auto",
            "offload_folder": str(_offload_dir()),
        }
        if tok:
            load_kwargs["token"] = tok
        _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
        _model.eval()
        # device_map="auto" may place the model on cuda/cpu mix; report best effort
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _load_mode = "auto_offload"
        return True
    except Exception as exc:
        last_error = exc
        load_attempts.append(f"auto_offload failed: {exc}")
        _model = None
        _tokenizer = None
        _device = None
        _load_mode = None
        raise RuntimeError(
            "Could not load Qwen/Qwen2.5-1.5B-Instruct. " + "; ".join(load_attempts)
            + f". Last error: {last_error.__class__.__name__}: {last_error}"
        ) from last_error


def _result(success: bool, **kwargs) -> Dict[str, Any]:
    base = {
        "success": success,
        "original": kwargs.get("original", ""),
        "translated": kwargs.get("translated"),
        "translated_text": kwargs.get("translated_text"),
        "target_language": kwargs.get("target_language"),
        "style": kwargs.get("style"),
        "technique": kwargs.get("technique"),
        "model": kwargs.get("model", MODEL_NAME),
        "device": kwargs.get("device"),
        "latency_seconds": kwargs.get("latency_seconds"),
        "load_seconds": kwargs.get("load_seconds"),
        "load_mode": kwargs.get("load_mode", _load_mode),
        "error": kwargs.get("error"),
    }
    return base


def translate_with_llm(
    text,
    target_language="Hindi",
    style="Neutral",
    technique="zero_shot",
    max_new_tokens=160,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    original = text if isinstance(text, str) else str(text)
    if not isinstance(text, str) or not text.strip():
        return _result(False, original=original, target_language=target_language, style=style, technique=technique,
                       latency_seconds=0.0, error="Input text cannot be empty or whitespace-only.")
    if target_language not in SUPPORTED_LANGUAGES:
        return _result(False, original=original, target_language=target_language, style=style, technique=technique,
                       latency_seconds=0.0, error=f"Unsupported target language: {target_language}")

    dep = dependency_error()
    if dep:
        return _result(False, original=original, target_language=target_language, style=style, technique=technique,
                       latency_seconds=0.0, error=dep)
    try:
        import torch

        prompt = build_translation_prompt(text=text, target_language=target_language, style=style, technique=technique)
        load_start = time.perf_counter()
        load_llm(token=token)
        load_seconds = round(time.perf_counter() - load_start, 3) if _model is not None else None
        messages = [
            {
                "role": "system",
                "content": "You are a professional English-to-Indian-language translator. Follow the translation instructions carefully. Return only the translated text.",
            },
            {"role": "user", "content": prompt},
        ]
        chat_text = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = _tokenizer(chat_text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(_device) for k, v in inputs.items()}
        if _device == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.inference_mode():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=_tokenizer.eos_token_id,
                eos_token_id=_tokenizer.eos_token_id,
            )
        if _device == "cuda":
            torch.cuda.synchronize()
        latency = round(time.perf_counter() - start_time, 4)
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        translated_text = _tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        if not translated_text:
            return _result(False, original=original, target_language=target_language, style=style, technique=technique,
                           model=MODEL_NAME, device=_device, latency_seconds=latency, load_seconds=load_seconds,
                           load_mode=_load_mode,
                           error="Model returned an empty translation.")
        return _result(True, original=original, translated=translated_text, translated_text=translated_text,
                       target_language=target_language, style=style, technique=technique, model=MODEL_NAME,
                       device=_device, latency_seconds=latency, load_seconds=load_seconds, load_mode=_load_mode)
    except Exception as error:
        return _result(False, original=original, target_language=target_language, style=style, technique=technique,
                       model=MODEL_NAME, device=_device, latency_seconds=None, load_mode=_load_mode,
                       error=f"{error.__class__.__name__}: {error}")


def unload_llm():
    global _tokenizer, _model, _device, _load_mode
    _model = None
    _tokenizer = None
    _device = None
    _load_mode = None
    from services.model_manager import release_model_memory

    release_model_memory()
