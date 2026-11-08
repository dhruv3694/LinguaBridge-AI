from io import BytesIO

from PIL import Image

from modules.image_caption import _load_rgb_image


def test_load_rgb_image_accepts_pil_image():
    image = Image.new("L", (8, 8), color=128)

    result = _load_rgb_image(image)

    assert result.mode == "RGB"
    assert result.size == (8, 8)


def test_load_rgb_image_accepts_file_like_image():
    image_data = BytesIO()
    Image.new("RGBA", (8, 8), color=(255, 0, 0, 128)).save(image_data, format="PNG")
    image_data.seek(0)

    result = _load_rgb_image(image_data)

    assert result.mode == "RGB"
    assert result.size == (8, 8)
