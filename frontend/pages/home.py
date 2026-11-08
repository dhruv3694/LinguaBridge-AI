"""Home / Dashboard with honest per-model readiness status."""
from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any

import streamlit as st

from config.model_config import MODEL_ALLOWLIST
from services.model_manager import runtime_status
from services.model_readiness import check_all


def _quick_import(name):
    try:
        __import__(name)
        return True, None
    except Exception as exc:
        return False, f"{exc.__class__.__name__}"


def render():
    st.title("LinguaBridge AI — P19 Dashboard")
    st.caption("Notebook-first reconstruction (source: `notebooks/LinguaBridge_AI_Development.ipynb`).")

    st.subheader("Runtime diagnostic (Streamlit process)")
    st.code(
        f"Python executable : {sys.executable}\n"
        f"Python version    : {sys.version.split()[0]}\n"
        f"Working dir       : {os.getcwd()}\n"
        f"Virtual env       : {sys.prefix}\n"
        f"Base Python       : {sys.base_prefix}\n"
        f"torch spec        : {importlib.util.find_spec('torch')}\n"
        f"transformers spec : {importlib.util.find_spec('transformers')}\n"
        f"huggingface_hub   : {importlib.util.find_spec('huggingface_hub')}\n"
        f"torch.__version__ : {_torch_version()}\n"
        f"cuda available    : {_cuda_available()}\n"
        f"gpu name          : {_gpu_name()}",
        language="text",
    )

    status = runtime_status()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Device", str(status.get("device", "cpu")).upper())
    c2.metric("CUDA", "Yes" if status.get("cuda_available") else "No")
    c3.metric("HF token (session)", "Present" if status.get("hf_token_present") else "Missing")
    c4.metric("GPU", "Detected" if status.get("gpu_name") else "N/A")
    if status.get("gpu_name"):
        st.info(f"GPU: {status['gpu_name']}")

    st.subheader("Dependency health (quick checks)")
    checks = [
        ("torch", "torch"), ("huggingface_hub", "huggingface_hub"), ("transformers", "transformers"),
        ("streamlit", "streamlit"), ("sacrebleu", "sacrebleu"), ("PIL", "PIL"),
        ("sentencepiece", "sentencepiece"), ("soundfile", "soundfile"), ("parler_tts", "parler_tts"),
    ]
    rows = []
    for label, mod in checks:
        ok, err = _quick_import(mod)
        rows.append({"component": label, "status": "OK" if ok else "missing/error", "note": "" if ok else err})
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption("Run `python scripts/check_environment.py` for full diagnostics.")

    st.subheader("Model readiness (real inference test)")
    st.caption("These checks load models and can take several minutes. Each model is released after its test.")
    if st.button("Run model readiness checks", key="run_model_readiness"):
        with st.spinner("Testing models one at a time..."):
            st.session_state["model_readiness"] = check_all()

    readiness = st.session_state.get("model_readiness")
    labels = {
        "marianmt": "MarianMT (HI/MR)",
        "qwen": "Qwen2.5-1.5B-Instruct",
        "blip": "BLIP image captioning",
        "parler_tts": "Indic Parler-TTS",
        "hf_auth": "Hugging Face auth",
    }
    if readiness is None:
        st.info("Run the checks when you want to verify model downloads and inference.")
    else:
        for key, label in labels.items():
            r = readiness.get(key, {})
            status_val = r.get("status", "unknown")
            icon = "OK" if status_val == "ready" else ("BLOCKED" if status_val == "blocked" else "?")
            st.write(f"**{label}**: {icon}")
            detail = r.get("detail", "")
            if detail:
                st.caption(detail)
            meta = []
            for k in ("device", "load_mode", "latency_seconds"):
                v = r.get(k)
                if v is not None:
                    meta.append(f"{k}={v}")
            if meta:
                st.caption(" | ".join(meta))

    with st.expander("Application model allowlist (5 IDs)"):
        for m in MODEL_ALLOWLIST:
            st.code(m)
    st.subheader("Available functionality")
    st.markdown(
        "- Translator (MarianMT/Qwen), Prompt Studio, MarianMT vs Qwen comparison\n"
        "- Image captioning (BLIP) + translation, Text-to-Speech (Indic Parler-TTS)\n"
        "- Gender bias evaluation (heuristic), U1–U5 docs"
    )


def _torch_version():
    try:
        import torch
        return torch.__version__
    except Exception as exc:
        return f"ERROR: {exc}"


def _cuda_available():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception as exc:
        return f"ERROR: {exc}"


def _gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except Exception as exc:
        return f"ERROR: {exc}"
