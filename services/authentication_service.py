"""Security & credential handling."""
from __future__ import annotations
from typing import Optional
from huggingface_hub import HfApi

from modules.hf_auth import (
    SESSION_TOKEN_KEY, clear_session_token, get_huggingface_token, set_session_token, validate_token,
)


def validate_and_connect(token: str) -> dict:
    res = validate_token(token)
    if res.get("valid"):
        set_session_token(token, res.get("username"))
    return res


def auth_kwargs(token: Optional[str] = None) -> dict:
    tok = token if token is not None else get_huggingface_token()
    return {"token": tok} if tok else {}
