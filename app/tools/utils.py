import base64
import hashlib
import re
from pathlib import Path

from PIL import Image, UnidentifiedImageError

MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_DIMENSION = 5000
IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}
IMAGE_FORMAT_MIME_TYPES = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
}


def hash_password(password: str) -> str:
    """Хэширует пароль с использованием SHA-256."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def clean_text(text: str) -> str:
    """Очищает текст от markdown-стилей: **, *, ###, ##, #."""
    cleaned_text = re.sub(r"(\*\*|\*|###|##|#)", "", text)
    cleaned_text = re.sub(r"\n+", " ", cleaned_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    return cleaned_text


def encode_image_to_base64(image_path: str) -> str:
    """Кодирует изображение в формат Base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def validate_image_file(image_path: str | Path) -> tuple[str, int, int]:
    """Проверяет размер, формат и разрешение изображения."""
    path = Path(image_path)
    extension_mime = IMAGE_MIME_TYPES.get(path.suffix.lower())
    if extension_mime is None:
        raise ValueError("Поддерживаются только изображения JPG и PNG")

    if path.stat().st_size > MAX_IMAGE_SIZE_BYTES:
        raise ValueError("Размер изображения не должен превышать 5 МБ")

    try:
        with Image.open(path) as image:
            image_format = image.format
            width, height = image.size
            image.verify()
    except (Image.DecompressionBombError, OSError, UnidentifiedImageError) as error:
        raise ValueError("Файл не является корректным изображением") from error

    content_mime = IMAGE_FORMAT_MIME_TYPES.get(image_format)
    if content_mime is None or content_mime != extension_mime:
        raise ValueError("Расширение файла не соответствует его содержимому")

    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise ValueError("Разрешение изображения не должно превышать 5000x5000 пикселей")

    return content_mime, width, height
