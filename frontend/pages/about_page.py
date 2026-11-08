"""About / Model Inventory."""
from __future__ import annotations
from pathlib import Path
import streamlit as st
from config.model_config import MODEL_ALLOWLIST

DOCS = Path(__file__).resolve().parents[2] / "docs"


def render():
    st.title("About / Model Inventory")
    st.markdown("All AI models below are extracted from the notebook allowlist. No additional Hugging Face models are introduced.")
    for m in MODEL_ALLOWLIST:
        st.code(m)
    inv = DOCS / "model_inventory.md"
    lic = DOCS / "licenses.md"
    if inv.exists():
        st.subheader("Model inventory")
        st.markdown(inv.read_text(encoding="utf-8"))
    if lic.exists():
        st.subheader("Licences")
        st.markdown(lic.read_text(encoding="utf-8"))
