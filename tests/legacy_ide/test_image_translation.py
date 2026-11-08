"""
Unit and Functional Tests for Phase 6: Image-to-Translation Pipeline.
Validates end-to-end chaining of BLIP captioning to MarianMT translation,
independent stage error handling, latency tracking, and schema conformity.
"""

from unittest.mock import patch
from PIL import Image
import pytest

from modules.image_translation import image_to_translation


@pytest.fixture
def sample_pil_image():
    """Provides a synthetic 100x100 RGB image for testing."""
    return Image.new("RGB", (100, 100), color=(100, 150, 200))


# ============================================================================
# 1. Pipeline Validation Tests
# ============================================================================

def test_invalid_target_language_rejection(sample_pil_image):
    """Verify that unsupported languages are rejected at validation stage."""
    res = image_to_translation(sample_pil_image, target_language="Russian")
    assert res["success"] is False
    assert res["stage_failed"] == "validation"
    assert res["caption"] is None
    assert "unsupported target language" in res["error"].lower()


def test_invalid_image_input_fails_at_captioning_stage():
    """Verify that corrupt or invalid image inputs fail gracefully at Stage 1."""
    res = image_to_translation("NON_EXISTENT_FILE.jpg", target_language="Hindi")
    assert res["success"] is False
    assert res["stage_failed"] == "captioning"
    assert res["caption"] is None
    assert res["translated_caption"] is None
    assert "stage 1" in res["error"].lower()


def test_result_schema_conformity(sample_pil_image):
    """Verify that image_to_translation returns all required schema keys."""
    res = image_to_translation(None, target_language="Hindi")
    expected_keys = {
        "success",
        "stage_failed",
        "caption",
        "translated_caption",
        "target_language",
        "blip_model",
        "translator_model",
        "blip_latency_seconds",
        "translation_latency_seconds",
        "total_latency_seconds",
        "device_caption",
        "device_translation",
        "error",
    }
    assert expected_keys.issubset(res.keys())


# ============================================================================
# 2. Mocked End-to-End Pipeline Tests
# ============================================================================

@patch("modules.image_translation.generate_caption")
@patch("modules.image_translation.translate_text")
def test_successful_image_to_translation_pipeline(mock_translate, mock_caption, sample_pil_image):
    """Verify successful end-to-end execution of Stage 1 (BLIP) -> Stage 2 (MarianMT)."""
    # Mock Stage 1: BLIP Captioning
    mock_caption.return_value = {
        "success": True,
        "caption": "A person reading a book in a quiet library.",
        "model_name": "Salesforce/blip-image-captioning-base",
        "device": "cuda",
        "latency_seconds": 0.25,
        "error": None,
    }

    # Mock Stage 2: MarianMT Translation
    mock_translate.return_value = {
        "success": True,
        "original_text": "A person reading a book in a quiet library.",
        "translated_text": "एक व्यक्ति शांत पुस्तकालय में पुस्तक पढ़ रहा है।",
        "target_language": "Hindi",
        "model_name": "Helsinki-NLP/opus-mt-en-hi",
        "device": "cuda",
        "latency_seconds": 0.05,
        "error": None,
    }

    res = image_to_translation(sample_pil_image, target_language="Hindi")

    assert res["success"] is True
    assert res["stage_failed"] is None
    assert res["caption"] == "A person reading a book in a quiet library."
    assert res["translated_caption"] == "एक व्यक्ति शांत पुस्तकालय में पुस्तक पढ़ रहा है।"
    assert res["target_language"] == "Hindi"
    assert res["blip_latency_seconds"] == 0.25
    assert res["translation_latency_seconds"] == 0.05
    assert res["total_latency_seconds"] > 0
    assert res["error"] is None

    # Verify BLIP called with image and MarianMT called with generated caption
    mock_caption.assert_called_once_with(sample_pil_image)
    mock_translate.assert_called_once_with(
        text="A person reading a book in a quiet library.",
        target_language="Hindi",
    )


@patch("modules.image_translation.generate_caption")
@patch("modules.image_translation.translate_text")
def test_stage_2_failure_preserves_caption(mock_translate, mock_caption, sample_pil_image):
    """Verify that if Stage 2 (Translation) fails, the Stage 1 caption is still preserved."""
    mock_caption.return_value = {
        "success": True,
        "caption": "A laptop on a wooden desk with a notebook.",
        "model_name": "Salesforce/blip-image-captioning-base",
        "device": "cuda",
        "latency_seconds": 0.21,
        "error": None,
    }

    mock_translate.return_value = {
        "success": False,
        "original_text": "A laptop on a wooden desk with a notebook.",
        "translated_text": None,
        "target_language": "Marathi",
        "model_name": "Helsinki-NLP/opus-mt-en-mr",
        "device": "cuda",
        "latency_seconds": 0.02,
        "error": "CUDA out-of-memory during translation.",
    }

    res = image_to_translation(sample_pil_image, target_language="Marathi")

    assert res["success"] is False
    assert res["stage_failed"] == "translation"
    assert res["caption"] == "A laptop on a wooden desk with a notebook."
    assert res["translated_caption"] is None
    assert "stage 2" in res["error"].lower()


# ============================================================================
# 3. Live Integration Tests (Requires model weights / network)
# ============================================================================

@pytest.mark.integration
def test_live_image_to_translation_pipeline(sample_pil_image):
    """Live integration test: image -> BLIP -> MarianMT Hindi translation."""
    res = image_to_translation(sample_pil_image, target_language="Hindi")
    assert res["success"] is True
    assert res["caption"] is not None
    assert res["translated_caption"] is not None
    assert len(res["translated_caption"]) > 0
