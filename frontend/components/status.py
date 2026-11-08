"""Streamlit UI helpers."""
from __future__ import annotations

import streamlit as st

from modules.hf_auth import get_huggingface_token, SESSION_USER_KEY


def auth_sidebar_status():
    st.sidebar.markdown("### Hugging Face")
    token = get_huggingface_token()
    if token:
        user = st.session_state.get(SESSION_USER_KEY) or "authenticated"
        st.sidebar.success(f"Authenticated: {user}")
        st.sidebar.caption("Token stored in session only.")
    else:
        st.sidebar.warning("No HF token in session")
        st.sidebar.caption("Some models may require a token; use the Authentication page.")
