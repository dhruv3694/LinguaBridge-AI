"""BLIP Image Captioning (notebook cell 59). Lazy heavy imports."""
from __future__ import annotations
import time
from typing import Any, Dict, Optional
from config.model_config import BLIP_MODEL_ID
from modules.hf_auth import get_huggingface_token

MODEL_NAME = BLIP_MODEL_ID
_processor = None
_model = None
_device = None


def _load_rgb_image(image_input):
    from PIL import Image

    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    with Image.open(image_input) as source:
        return source.convert("RGB")


def load_blip(token: Optional[str] = None):
    global _processor, _model, _device
    if _model is not None:
        return True
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor
    tok = token if token is not None else get_huggingface_token()
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _processor = BlipProcessor.from_pretrained(MODEL_NAME, token=tok)
    dtype = torch.float16 if _device == "cuda" else torch.float32
    _model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=dtype, token=tok)
    _model.to(_device); _model.eval()
    return True


def caption_image(image_path, max_new_tokens=60, token: Optional[str] = None) -> Dict[str, Any]:
    if not image_path:
        return {"success": False, "error": "Please provide an image path."}
    try:
        import torch
        image = _load_rgb_image(image_path)
    except Exception as exc:
        return {"success": False, "error": f"Unable to open image: {exc}"}
    try:
        load_blip(token=token)
        inputs = _processor(images=image, return_tensors="pt")
        inputs = {k: v.to(_device) for k, v in inputs.items()}
        if _device == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].half()
        start = time.time()
        with torch.inference_mode():
            output = _model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=4)
        caption = _processor.decode(output[0], skip_special_tokens=True).strip()
        latency = round(time.time() - start, 3)
        return {"success": True, "caption": caption, "model": MODEL_NAME, "device": _device,
                "latency_seconds": latency, "image_size": image.size}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def unload_blip():
    global _processor, _model, _device
    _model = None; _processor = None; _device = None
    from services.model_manager import release_model_memory

    release_model_memory()
