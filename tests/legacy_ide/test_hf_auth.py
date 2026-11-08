"""Tests for local Hugging Face token lookup."""

from modules.hf_auth import get_huggingface_token


def test_token_is_loaded_from_local_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=hf_test_local_token\n", encoding="utf-8")

    assert get_huggingface_token(env_file) == "hf_test_local_token"


def test_environment_token_takes_precedence(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=hf_file_token\n", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "hf_environment_token")

    assert get_huggingface_token(env_file) == "hf_environment_token"


def test_missing_or_empty_token_returns_none(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=  \n", encoding="utf-8")

    assert get_huggingface_token(env_file) is None
