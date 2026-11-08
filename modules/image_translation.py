"""
LinguaBridge AI - Image-to-Translation Pipeline (notebook cell 71).
Pipeline: Image -> BLIP caption -> translation function.
"""
from __future__ import annotations

from modules.image_caption import caption_image


def process_image(image_path, target_language="Hindi", translation_function=None):
    if translation_function is None:
        raise ValueError("A translation_function must be provided.")
    caption_result = caption_image(image_path)
    if not caption_result.get("success"):
        return {"success": False, "stage": "captioning", "error": caption_result.get("error")}
    caption = caption_result["caption"]
    translation_result = translation_function(caption, target_language)
    if not translation_result.get("success"):
        return {
            "success": False,
            "stage": "translation",
            "caption": caption,
            "error": translation_result.get("error"),
        }
    return {
        "success": True,
        "image_path": str(image_path),
        "english_caption": caption,
        "translated_caption": translation_result.get("translated") or translation_result.get("translated_text"),
        "target_language": target_language,
        "caption_model": caption_result.get("model"),
        "translation_model": translation_result.get("model", "Unknown"),
        "caption_latency_seconds": caption_result.get("latency_seconds"),
        "translation_latency_seconds": translation_result.get("latency_seconds"),
    }
