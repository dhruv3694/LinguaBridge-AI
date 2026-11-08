"""Primary Translator page: MarianMT and Qwen."""
from __future__ import annotations

import streamlit as st

from modules.llm_translator import translate_with_llm
from modules.prompt_engineering import SUPPORTED_STYLES, SUPPORTED_TECHNIQUES
from modules.translator import translate_text as marian_translate
from services.model_manager import runtime_status


def _word_count(text: str) -> int:
    return len([w for w in (text or "").split() if w.strip()])


def render():
    st.title("English → Hindi / Marathi Translator")
    st.caption("MarianMT (seq2seq) and Qwen2.5-1.5B-Instruct (prompt-based LLM) — notebook model allowlist only.")
    rs = runtime_status()
    st.info(
        f"Runtime: device={rs.get('device')} | CUDA={rs.get('cuda_available')} | HF token in session={rs.get('hf_token_present')}. "
        "If imports fail, run `python scripts/check_environment.py` using the same interpreter as Streamlit."
    )
    engine = st.radio("Engine", ["MarianMT (Helsinki-NLP)", "Qwen2.5-1.5B-Instruct"], horizontal=True)
    target = st.selectbox("Target language", ["Hindi", "Marathi"])
    technique = "zero_shot"
    style = "Neutral"
    if engine.startswith("Qwen"):
        c1, c2 = st.columns(2)
        technique = c1.selectbox(
            "Prompt technique", SUPPORTED_TECHNIQUES, format_func=lambda x: x.replace("_", " ").title()
        )
        style = c2.selectbox("Style / register", list(SUPPORTED_STYLES.keys()))
        st.caption(SUPPORTED_STYLES[style])
    text = st.text_area("English input", height=140, placeholder="Enter English text to translate...")
    col_a, col_b = st.columns([1, 3])
    do = col_a.button("Translate", type="primary", disabled=not (text and text.strip()))
    col_b.caption(f"Input words: {_word_count(text)} | characters: {len(text or '')}")
    if do:
        with st.spinner("Running inference (first run may download/load model weights)..."):
            if engine.startswith("Marian"):
                result = marian_translate(text, target)
            else:
                result = translate_with_llm(text, target_language=target, style=style, technique=technique)
        if result.get("success"):
            out = result.get("translated") or result.get("translated_text") or ""
            st.subheader("Translated output")
            st.text_area("Output", value=out, height=140)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Output words", _word_count(out))
            m2.metric("Latency (s)", result.get("latency_seconds", "—"))
            m3.metric("Device", str(result.get("device") or "—").upper())
            m4.metric("Model", (result.get("model") or "—").split("/")[-1])
            if result.get("load_seconds") is not None:
                st.caption(f"Model load/cached access time observed in this call: {result.get('load_seconds')}s (excluded from inference latency).")
            st.code(out, language=None)
        else:
            st.error(result.get("error", "Translation failed."))
            st.json({k: v for k, v in result.items() if k not in ("original",)})
