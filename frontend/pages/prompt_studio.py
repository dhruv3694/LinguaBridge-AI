"""Prompt Engineering Studio (U1)."""
from __future__ import annotations
import streamlit as st
from modules.prompt_engineering import (
    SUPPORTED_STYLES, SUPPORTED_TECHNIQUES, build_translation_prompt, compare_prompt_techniques,
)
from modules.llm_translator import translate_with_llm


def render():
    st.title("Prompt Engineering Studio")
    st.markdown(
        "Explore notebook prompt techniques (zero-shot, few-shot, role-based) and styles. "
        "Prompts are sent to Qwen2.5-1.5B-Instruct; MarianMT is a separate seq2seq baseline."
    )
    with st.expander("U1 concepts (summary)"):
        st.markdown(
            "- **Translation lifecycle:** input → tokenization → model inference → decoding → postprocessing.\n"
            "- **Generative vs discriminative:** Qwen generates target tokens; MarianMT is a conditional seq2seq generator trained for translation.\n"
            "- **Token generation:** next-token probabilities (logits/softmax) guide decoding; notebook uses greedy decoding for Qwen.\n"
            "- **Latent representations:** transformers encode meaning in contextual vectors; this UI shows prompts/outputs, not hidden states."
        )
    target = st.selectbox("Target language", ["Hindi", "Marathi"], key="pe_lang")
    style = st.selectbox("Style", list(SUPPORTED_STYLES.keys()), key="pe_style")
    text = st.text_area("English text", "Please submit your project report by tomorrow.", key="pe_text")
    tab1, tab2 = st.tabs(["Build prompt", "Compare techniques & run"])
    with tab1:
        technique = st.selectbox("Technique", SUPPORTED_TECHNIQUES, key="pe_t")
        if st.button("Generate prompt"):
            try:
                prompt = build_translation_prompt(text, target, style, technique)
                st.code(prompt)
            except Exception as e:
                st.error(str(e))
    with tab2:
        if st.button("Compare all techniques", type="primary"):
            try:
                prompts = compare_prompt_techniques(text, target, style)
                for tech, p in prompts.items():
                    st.subheader(tech.replace("_", " ").title())
                    st.code(p)
            except Exception as e:
                st.error(str(e))
        run = st.selectbox("Run Qwen with technique", SUPPORTED_TECHNIQUES, key="pe_run_t")
        if st.button("Translate with selected technique"):
            with st.spinner("Running Qwen..."):
                res = translate_with_llm(text, target, style=style, technique=run)
            if res.get("success"):
                st.success(res.get("translated") or res.get("translated_text"))
                st.caption(f"Model: {res.get('model')} | latency: {res.get('latency_seconds')}s | style: {res.get('style')}")
            else:
                st.error(res.get("error"))
