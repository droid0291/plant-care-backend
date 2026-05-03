import base64
import io
from PIL import Image
from config import get_settings


def decode_and_validate_image(image_base64: str) -> Image.Image:
    """Decode base64 string, validate it's an image, return PIL Image."""
    try:
        image_bytes = base64.b64decode(image_base64)
    except Exception:
        raise ValueError("Invalid base64 encoding")

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
        img = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise ValueError("Provided data is not a valid image")

    return img


def resize_and_encode(image_base64: str) -> str:
    """Resize image to max dimension and re-encode as JPEG base64."""
    settings = get_settings()
    img = decode_and_validate_image(image_base64)

    img = img.convert("RGB")

    max_px = settings.max_image_size_px
    if img.width > max_px or img.height > max_px:
        img.thumbnail((max_px, max_px), Image.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")
