"""U1–U5 documentation and assignment mapping."""
from __future__ import annotations
from pathlib import Path
import streamlit as st

DOCS = Path(__file__).resolve().parents[2] / "docs"


def render():
    st.title("U1–U5 Assignment Documentation")
    tabs = st.tabs(["U1", "U2", "U3", "U4", "U5", "Requirement matrix", "Security"])
    with tabs[0]:
        st.markdown(
            """
### U1 — Prompt Engineering & Generative AI
- **Implemented (notebook):** zero-shot, few-shot, role-based prompts; tones Neutral/Formal/Informal/Conversational/Academic/Professional; Qwen chat-template inference.
- **Lifecycle:** source text → prompt construction → tokenization → causal generation → decode new tokens.
- **Generative framing:** Qwen generates target tokens conditioned on prompt; MarianMT is seq2seq translation model.
- **Probability/latent space:** discussed conceptually; no hidden-state extraction in UI.
"""
        )
    with tabs[1]:
        st.markdown(
            """
### U2 — Transformers & Translation
- **MarianMT:** encoder–decoder seq2seq (`Helsinki-NLP/opus-mt-en-hi/mr`), beam search.
- **Qwen:** decoder-only instruction model (`Qwen/Qwen2.5-1.5B-Instruct`), prompt-based translation (no fine-tuning claimed).
- **Attention:** self-attention in transformers; cross-attention in encoder–decoder; causal attention in decoder-only LLMs.
"""
        )
    with tabs[2]:
        st.markdown(
            """
### U3 — Multimodal AI
- **BLIP:** `Salesforce/blip-image-captioning-base` generates English captions.
- **Pipeline:** image → caption → translation (Hindi/Marathi) via selected notebook translation function.
"""
        )
    with tabs[3]:
        st.markdown(
            """
### U4 — Python AI Application
- Streamlit UI exposes translation, prompt studio, comparison, image pipeline, TTS, bias evaluation, and docs.
- Audio via Indic Parler-TTS; evaluation via SacreBLEU/pandas/matplotlib; local inference with lazy loading.
"""
        )
    with tabs[4]:
        st.markdown(
            """
### U5 — Evaluation & Responsible AI
- MarianMT vs Qwen benchmark/latency and BLEU/chrF when human references exist.
- Gender bias screening uses heuristic markers on profession dataset; requires human review.
- Limitations: marker heuristics incomplete, small datasets, model hallucinations/bias possible, reference translations needed for valid automatic metrics.
"""
        )
    with tabs[5]:
        p = DOCS / "u1_u5_requirement_matrix.md"
        st.markdown(p.read_text(encoding="utf-8") if p.exists() else "Matrix file not found.")
    with tabs[6]:
        p = DOCS / "security.md"
        st.markdown(p.read_text(encoding="utf-8") if p.exists() else "Security doc not found.")
