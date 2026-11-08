"""LinguaBridge AI Streamlit app (notebook-first reconstruction)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import configure_model_storage

configure_model_storage()

import streamlit as st

st.set_page_config(page_title="LinguaBridge AI — P19", page_icon="🌐", layout="wide")

from frontend.components.status import auth_sidebar_status
from frontend.pages import (
    about_page, auth, bias_page, comparison, docs_page, home, image_page,
    prompt_studio, translator_page, tts_page,
)

PAGES = {
    "Home / Dashboard": home.render,
    "Hugging Face Authentication": auth.render,
    "Translator (MarianMT / Qwen)": translator_page.render,
    "Prompt Engineering Studio": prompt_studio.render,
    "MarianMT vs Qwen Comparison": comparison.render,
    "Image Captioning & Translation": image_page.render,
    "Text-to-Speech": tts_page.render,
    "Gender Bias Evaluation": bias_page.render,
    "U1–U5 Documentation": docs_page.render,
    "About / Model Inventory": about_page.render,
}

st.sidebar.title("LinguaBridge AI")
page = st.sidebar.radio("Navigation", list(PAGES.keys()))
auth_sidebar_status()
st.sidebar.divider()
st.sidebar.caption("Source: notebooks/LinguaBridge_AI_Development.ipynb")
PAGES[page]()
