import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.tools.utils import (
    MAX_IMAGE_DIMENSION,
    MAX_IMAGE_SIZE_BYTES,
    validate_image_file,
)


class ImageValidationTests(unittest.TestCase):
    def create_image_file(self, suffix: str, image_format: str, size: tuple[int, int]) -> Path:
        temporary_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        temporary_file.close()
        path = Path(temporary_file.name)
        image = Image.new("RGB", size, color="white")
        image.save(path, format=image_format)
        return path

    def test_valid_png_returns_png_mime_and_dimensions(self) -> None:
        path = self.create_image_file(".png", "PNG", (640, 480))
        self.addCleanup(path.unlink, missing_ok=True)

        result = validate_image_file(path)

        self.assertEqual(result, ("image/png", 640, 480))

    def test_valid_jpeg_returns_jpeg_mime(self) -> None:
        path = self.create_image_file(".jpg", "JPEG", (640, 480))
        self.addCleanup(path.unlink, missing_ok=True)

        result = validate_image_file(path)

        self.assertEqual(result, ("image/jpeg", 640, 480))

    def test_valid_png_with_temporary_suffix_uses_original_extension(self) -> None:
        path = self.create_image_file(".tmp", "PNG", (640, 480))
        self.addCleanup(path.unlink, missing_ok=True)

        result = validate_image_file(path, ".png")

        self.assertEqual(result, ("image/png", 640, 480))

    def test_rejects_extension_and_content_mismatch(self) -> None:
        path = self.create_image_file(".jpg", "PNG", (640, 480))
        self.addCleanup(path.unlink, missing_ok=True)

        with self.assertRaisesRegex(ValueError, "не соответствует"):
            validate_image_file(path)

    def test_rejects_invalid_image_content(self) -> None:
        temporary_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temporary_file.write(b"not an image")
        temporary_file.close()
        path = Path(temporary_file.name)
        self.addCleanup(path.unlink, missing_ok=True)

        with self.assertRaisesRegex(ValueError, "корректным изображением"):
            validate_image_file(path)

    def test_rejects_files_over_five_megabytes(self) -> None:
        temporary_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temporary_file.write(b"0" * (MAX_IMAGE_SIZE_BYTES + 1))
        temporary_file.close()
        path = Path(temporary_file.name)
        self.addCleanup(path.unlink, missing_ok=True)

        with self.assertRaisesRegex(ValueError, "5 МБ"):
            validate_image_file(path)

    def test_rejects_images_over_5000_pixels(self) -> None:
        path = self.create_image_file(
            ".png",
            "PNG",
            (MAX_IMAGE_DIMENSION + 1, 100),
        )
        self.addCleanup(path.unlink, missing_ok=True)

        with self.assertRaisesRegex(ValueError, "5000"):
            validate_image_file(path)


if __name__ == "__main__":
    unittest.main()
