"""Session-only Hugging Face authentication for Streamlit (no plaintext persistence)."""
from __future__ import annotations

from typing import Optional

SESSION_TOKEN_KEY = "hf_token"
SESSION_AUTH_KEY = "hf_authenticated"
SESSION_USER_KEY = "hf_username"


def _get_streamlit():
    try:
        import streamlit as st
        return st
    except Exception:
        return None


def get_huggingface_token() -> Optional[str]:
    st = _get_streamlit()
    if st is not None and hasattr(st, "session_state"):
        token = st.session_state.get(SESSION_TOKEN_KEY)
        if token and str(token).strip():
            return str(token).strip()
    return None


def set_session_token(token: str, username: Optional[str] = None) -> None:
    st = _get_streamlit()
    if st is None:
        raise RuntimeError("Streamlit session is required for session-only token storage.")
    st.session_state[SESSION_TOKEN_KEY] = token.strip()
    st.session_state[SESSION_AUTH_KEY] = True
    st.session_state[SESSION_USER_KEY] = username or ""


def clear_session_token() -> None:
    st = _get_streamlit()
    if st is None:
        return
    for key in (SESSION_TOKEN_KEY, SESSION_AUTH_KEY, SESSION_USER_KEY):
        st.session_state.pop(key, None)


def validate_token(token: str) -> dict:
    """Validate token with Hugging Face Hub without persisting credentials."""
    token = (token or "").strip()
    if not token:
        return {"valid": False, "error": "Token cannot be empty."}
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        return {
            "valid": False,
            "error": (
                f"huggingface_hub is not available in this Python environment: {exc}. "
                "Install dependencies with the same interpreter used by Streamlit: "
                "`python -m pip install huggingface_hub transformers`."
            ),
            "dependency_missing": True,
        }
    try:
        info = HfApi().whoami(token=token)
        username = info.get("name") or info.get("fullname") or info.get("email") or "authenticated"
        return {"valid": True, "username": username}
    except Exception as exc:
        msg = str(exc)
        # Never include token in messages
        return {"valid": False, "error": msg}


# Compatibility aliases (do NOT persist to disk)
def save_huggingface_token(token: str, *args, **kwargs) -> None:
    set_session_token(token)


save_hf_token = save_huggingface_token


def hf_from_pretrained(cls, *args, token=None, **kwargs):
    tok = token if token is not None else get_huggingface_token()
    if tok:
        try:
            return cls.from_pretrained(*args, token=tok, **kwargs)
        except TypeError:
            return cls.from_pretrained(*args, use_auth_token=tok, **kwargs)
    return cls.from_pretrained(*args, **kwargs)
