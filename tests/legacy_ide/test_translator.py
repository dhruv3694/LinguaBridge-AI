"""
Unit and Integration Tests for MarianMT Translation Engine.
"""

import pytest
from modules.translator import MarianTranslationEngine, translate_text


@pytest.fixture(scope="module")
def engine():
    return MarianTranslationEngine()


# ============================================================================
# 1. Validation & Schema Tests (Fast, offline, no model downloads)
# ============================================================================

def test_empty_string_input(engine):
    """Test that empty string input is rejected with a structured error."""
    res = engine.translate(text="", target_language="Hindi")
    assert res["success"] is False
    assert res["translated_text"] is None
    assert "empty" in res["error"].lower()


def test_whitespace_only_input(engine):
    """Test that whitespace-only input is rejected."""
    res = engine.translate(text="   \n\t   ", target_language="Marathi")
    assert res["success"] is False
    assert res["translated_text"] is None
    assert "empty" in res["error"].lower()


def test_invalid_target_language(engine):
    """Test that unsupported target languages are rejected."""
    res = engine.translate(text="Hello world", target_language="German")
    assert res["success"] is False
    assert res["translated_text"] is None
    assert "unsupported" in res["error"].lower()


def test_non_string_input(engine):
    """Test that non-string inputs are rejected gracefully."""
    res = engine.translate(text=99999, target_language="Hindi")  # type: ignore
    assert res["success"] is False
    assert "string" in res["error"].lower()


def test_structured_result_schema(engine):
    """Verify that translate() returns all required schema keys."""
    res = engine.translate(text="", target_language="Hindi")
    expected_keys = {
        "success",
        "original_text",
        "translated_text",
        "target_language",
        "model_name",
        "device",
        "latency_seconds",
        "error",
    }
    assert expected_keys.issubset(res.keys())


# ============================================================================
# 2. Live Translation Integration Tests (Requires model download / network)
# ============================================================================

@pytest.mark.integration
def test_english_to_hindi_translation(engine):
    """Verify valid translation from English to Hindi producing Devanagari script."""
    text = "Good morning."
    res = engine.translate(text=text, target_language="Hindi")
    assert res["success"] is True
    assert res["translated_text"] is not None
    assert len(res["translated_text"]) > 0
    assert res["target_language"] == "Hindi"
    assert res["latency_seconds"] >= 0
    assert res["error"] is None
    # Verify Devanagari Unicode characters (range U+0900 to U+097F)
    assert any("\u0900" <= char <= "\u097f" for char in res["translated_text"])


@pytest.mark.integration
def test_english_to_marathi_translation(engine):
    """Verify valid translation from English to Marathi producing Devanagari script."""
    text = "How are you?"
    res = engine.translate(text=text, target_language="Marathi")
    assert res["success"] is True
    assert res["translated_text"] is not None
    assert len(res["translated_text"]) > 0
    assert res["target_language"] == "Marathi"
    assert res["error"] is None
    assert any("\u0900" <= char <= "\u097f" for char in res["translated_text"])


@pytest.mark.integration
def test_convenience_translate_text_function():
    """Verify functional interface translate_text()."""
    res = translate_text("Education is important.", "Hindi")
    assert res["success"] is True
    assert isinstance(res["translated_text"], str)


@pytest.mark.integration
def test_model_in_memory_caching(engine):
    """Verify that once a model is loaded, subsequent checks reflect loaded state."""
    assert engine.is_model_loaded("Hindi") is True
