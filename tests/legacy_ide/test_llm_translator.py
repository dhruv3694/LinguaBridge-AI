"""
Unit and Integration Tests for Phase 3: Qwen LLM Translation Engine.
Tests cover validation, schema conformity, output cleanup, lazy loading, and live generation.
"""

import pytest
from modules.llm_translator import (
    QwenLLMTranslator,
    clean_llm_translation_output,
    translate_with_llm,
)


@pytest.fixture(scope="module")
def translator_instance():
    """Provides a fresh QwenLLMTranslator instance for unit testing."""
    return QwenLLMTranslator()


# ============================================================================
# 1. Validation & Schema Tests (Fast, offline, no model download)
# ============================================================================

def test_lazy_loading_on_instantiation(translator_instance):
    """Verify that instantiating the translator does NOT load weights into memory."""
    fresh_instance = QwenLLMTranslator()
    assert fresh_instance.is_model_loaded() is False
    assert fresh_instance.model is None
    assert fresh_instance.tokenizer is None


def test_empty_input_rejection(translator_instance):
    """Verify that empty strings return a structured error result."""
    res = translator_instance.translate(text="", target_language="Hindi")
    assert res["success"] is False
    assert res["translated_text"] is None
    assert "empty" in res["error"].lower()


def test_whitespace_only_rejection(translator_instance):
    """Verify that whitespace-only strings return a structured error result."""
    res = translator_instance.translate(text="   \n\t   ", target_language="Marathi")
    assert res["success"] is False
    assert res["translated_text"] is None
    assert "empty" in res["error"].lower()


def test_non_string_rejection(translator_instance):
    """Verify that non-string inputs are safely rejected."""
    res = translator_instance.translate(text=12345, target_language="Hindi")  # type: ignore
    assert res["success"] is False
    assert "string" in res["error"].lower()


def test_invalid_language_rejection(translator_instance):
    """Verify that unsupported languages return a structured error result."""
    res = translator_instance.translate(text="Hello world", target_language="Spanish")
    assert res["success"] is False
    assert "unsupported target language" in res["error"].lower()


def test_invalid_technique_rejection(translator_instance):
    """Verify that unsupported prompt techniques return a structured error result."""
    res = translator_instance.translate(
        text="Hello world",
        target_language="Hindi",
        technique="chain-of-thought",
    )
    assert res["success"] is False
    assert "unsupported technique" in res["error"].lower()


def test_invalid_tone_rejection(translator_instance):
    """Verify that unsupported tones return a structured error result."""
    res = translator_instance.translate(
        text="Hello world",
        target_language="Marathi",
        tone="poetic",
    )
    assert res["success"] is False
    assert "unsupported tone" in res["error"].lower()


def test_result_schema_conformity(translator_instance):
    """Verify that translate() returns all required schema keys."""
    res = translator_instance.translate(text="", target_language="Hindi")
    expected_keys = {
        "success",
        "original_text",
        "translated_text",
        "target_language",
        "technique",
        "tone",
        "model_name",
        "device",
        "latency_seconds",
        "error",
    }
    assert expected_keys.issubset(res.keys())
    assert res["technique"] == "zero-shot"
    assert res["tone"] == "formal"


# ============================================================================
# 2. Output Sanitization & Cleanup Unit Tests
# ============================================================================

def test_clean_output_markdown_fences():
    """Verify stripping of markdown code fences."""
    raw = "```\nनमस्ते दुनिया\n```"
    cleaned = clean_llm_translation_output(raw, "Hindi")
    assert cleaned == "नमस्ते दुनिया"


def test_clean_output_preamble_prefixes():
    """Verify stripping of common translation preambles."""
    raw1 = "Translation: सुप्रभात"
    assert clean_llm_translation_output(raw1, "Hindi") == "सुप्रभात"

    raw2 = "Target Translation: शुभ प्रभात"
    assert clean_llm_translation_output(raw2, "Marathi") == "शुभ प्रभात"

    raw3 = "अनुवाद: आप कैसे हैं?"
    assert clean_llm_translation_output(raw3, "Hindi") == "आप कैसे हैं?"


def test_clean_output_disclaimer_removal():
    """Verify filtering out explanatory postscripts."""
    raw = "भारत एक महान देश है।\nNote: This is in formal register."
    cleaned = clean_llm_translation_output(raw, "Hindi")
    assert cleaned == "भारत एक महान देश है।"


# ============================================================================
# 3. Live Model Integration Tests (Requires model weights / network)
# ============================================================================

@pytest.mark.integration
def test_qwen_live_hindi_translation():
    """Live integration test: English to Hindi translation with Qwen2.5-1.5B."""
    res = translate_with_llm(
        text="Good morning, how are you today?",
        target_language="Hindi",
        technique="zero-shot",
        tone="formal",
    )
    assert res["success"] is True
    assert res["translated_text"] is not None
    assert len(res["translated_text"]) > 0
    assert res["latency_seconds"] > 0
    # Check for Devanagari Unicode characters (U+0900 to U+097F)
    assert any("\u0900" <= char <= "\u097f" for char in res["translated_text"])


@pytest.mark.integration
def test_qwen_live_marathi_translation():
    """Live integration test: English to Marathi translation with Qwen2.5-1.5B."""
    res = translate_with_llm(
        text="Education is essential for social development.",
        target_language="Marathi",
        technique="few-shot",
        tone="academic",
    )
    assert res["success"] is True
    assert res["translated_text"] is not None
    assert any("\u0900" <= char <= "\u097f" for char in res["translated_text"])


@pytest.mark.integration
def test_qwen_caching_after_first_load(translator_instance):
    """Verify that once the model is loaded, is_model_loaded() is True."""
    # Only test if model has been loaded
    if translator_instance.is_model_loaded():
        assert translator_instance.model is not None
        assert translator_instance.tokenizer is not None
