"""
Unit and Integration Tests for Phase 5: BLIP Multimodal Image Captioning.
Validates input sanitization, image verification, corruption handling, schema, and lazy loading.
"""

import io
from PIL import Image
import pytest

from modules.image_caption import (
    BLIPImageCaptioner,
    generate_caption,
    validate_and_load_image,
)


@pytest.fixture(scope="module")
def captioner_instance():
    """Provides a fresh BLIPImageCaptioner instance for testing."""
    return BLIPImageCaptioner()


# ============================================================================
# 1. Image Validation & Error Handling Tests (Fast, offline)
# ============================================================================

def test_lazy_loading_on_instantiation(captioner_instance):
    """Verify that instantiating BLIPImageCaptioner does NOT load weights on startup."""
    fresh_instance = BLIPImageCaptioner()
    assert fresh_instance.is_model_loaded() is False
    assert fresh_instance.processor is None
    assert fresh_instance.model is None


def test_none_image_rejection(captioner_instance):
    """Verify that None input returns a structured error."""
    res = captioner_instance.generate_caption(None)
    assert res["success"] is False
    assert res["caption"] is None
    assert "no image provided" in res["error"].lower()


def test_invalid_type_rejection(captioner_instance):
    """Verify that non-image types (e.g. int) are safely rejected."""
    res = captioner_instance.generate_caption(123456)
    assert res["success"] is False
    assert res["caption"] is None
    assert "unsupported image input type" in res["error"].lower()


def test_corrupted_image_bytes(captioner_instance):
    """Verify that corrupted byte streams return a clear error without unhandled crashes."""
    corrupted_bytes = b"NOT_A_VALID_IMAGE_HEADER_1234567890"
    res = captioner_instance.generate_caption(corrupted_bytes)
    assert res["success"] is False
    assert res["caption"] is None
    assert "corrupted" in res["error"].lower() or "could not parse" in res["error"].lower()


def test_oversized_dimension_image(captioner_instance):
    """Verify that images exceeding max dimensions are rejected."""
    # Create an image exceeding max_dimension (4096)
    is_valid, img, err = validate_and_load_image(
        Image.new("RGB", (5000, 100)),
        max_dimension=4096,
    )
    assert is_valid is False
    assert img is None
    assert "exceed" in err.lower()


def test_valid_pil_image_normalization():
    """Verify that valid PIL images in various color modes (RGBA, L) normalize to RGB."""
    # Grayscale image
    img_gray = Image.new("L", (100, 100), color=128)
    is_valid, norm_gray, err = validate_and_load_image(img_gray)
    assert is_valid is True
    assert norm_gray.mode == "RGB"
    assert err is None

    # RGBA image
    img_rgba = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    is_valid, norm_rgba, err = validate_and_load_image(img_rgba)
    assert is_valid is True
    assert norm_rgba.mode == "RGB"
    assert err is None


def test_result_schema_conformity(captioner_instance):
    """Verify that generate_caption returns all required schema keys."""
    res = captioner_instance.generate_caption(None)
    expected_keys = {
        "success",
        "caption",
        "model_name",
        "device",
        "latency_seconds",
        "error",
    }
    assert expected_keys.issubset(res.keys())
    assert res["model_name"] == "Salesforce/blip-image-captioning-base"


# ============================================================================
# 2. Live Model Integration Tests (Requires model weights / network)
# ============================================================================

@pytest.mark.integration
def test_live_blip_caption_generation():
    """Generate caption on a synthetic PIL canvas using live BLIP model."""
    synthetic_image = Image.new("RGB", (224, 224), color=(73, 109, 137))
    res = generate_caption(synthetic_image)
    assert res["success"] is True
    assert res["caption"] is not None
    assert len(res["caption"]) > 0
    assert res["latency_seconds"] > 0
    assert res["error"] is None
