"""Hugging Face Authentication (session-only)."""
from __future__ import annotations

import streamlit as st

from modules.hf_auth import SESSION_USER_KEY, clear_session_token, get_huggingface_token, set_session_token, validate_token
from services.model_manager import runtime_status


def render():
    st.title("Hugging Face Authentication")
    st.markdown(
        "Enter your Hugging Face access token for gated/private model access when required. "
        "**Session-only:** token is stored only in Streamlit session state and never written to `.env`, source, CSV/JSON, logs, or outputs."
    )
    rs = runtime_status()
    st.caption(f"huggingface_hub available check runs during validation; Python: same interpreter as Streamlit recommended (`python -m streamlit run app.py`).")
    st.markdown(
        "1. Create token: [Hugging Face Settings → Access Tokens](https://huggingface.co/settings/tokens).\n"
        "2. Accept model terms for gated repos if needed.\n"
        "3. Validate & connect. Public models may work without a token."
    )
    current = get_huggingface_token()
    if current:
        st.success(f"Authenticated in session: {st.session_state.get(SESSION_USER_KEY, 'authenticated')}")
        if st.button("Disconnect / clear session token", type="primary"):
            clear_session_token()
            st.rerun()
    else:
        st.warning("No token in this session.")

    token_in = st.text_input("Hugging Face token", type="password", placeholder="hf_...")
    c1, c2 = st.columns(2)
    if c1.button("Validate & connect", disabled=not (token_in and token_in.strip())):
        with st.spinner("Validating token via Hugging Face Hub (no persistence)..."):
            res = validate_token(token_in)
        if res.get("valid"):
            set_session_token(token_in.strip(), res.get("username"))
            st.success(f"Connected as {res.get('username')}.")
            st.rerun()
        st.error(res.get("error", "Validation failed."))
        if res.get("dependency_missing"):
            st.code("python -m pip install --upgrade huggingface_hub", language="powershell")
    if c2.button("Clear input field"):
        st.rerun()
    st.info("The notebook contained a plaintext token; treat it as compromised and revoke it. Never reuse it.")
