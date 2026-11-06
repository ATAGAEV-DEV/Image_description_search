"""Utility behavior and image validation boundaries using real local files."""

import base64
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.tools import utils


class UtilityTests(unittest.TestCase):
    """Check deterministic helpers without external dependencies."""

    def test_hash_password_uses_utf8_sha256(self) -> None:
        password = "пароль с пробелом 🔒"
        expected = hashlib.sha256(password.encode("utf-8")).hexdigest()
        self.assertEqual(utils.hash_password(password), expected)
        self.assertNotEqual(utils.hash_password(password), utils.hash_password(password + "x"))

    def test_clean_text_removes_markdown_and_collapses_whitespace(self) -> None:
        self.assertEqual(utils.clean_text("**Кот**\n\n### на\tковре"), "Кот на ковре")
        self.assertEqual(utils.clean_text(""), "")
        self.assertEqual(utils.clean_text("нет разметки"), "нет разметки")

    def test_base64_encoding_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.bin"
            payload = bytes(range(256))
            path.write_bytes(payload)
            self.assertEqual(base64.b64decode(utils.encode_image_to_base64(str(path))), payload)

    def test_missing_file_propagates(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaises(FileNotFoundError),
        ):
            utils.encode_image_to_base64(str(Path(directory) / "missing.png"))


class ImageBoundaryTests(unittest.TestCase):
    """Validate inclusive byte and dimension limits plus format mismatches."""

    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.directory = Path(temporary_directory.name)

    def image(self, name: str, size: tuple[int, int] = (16, 16), fmt: str = "PNG") -> Path:
        path = self.directory / name
        with Image.new("RGB", size, "white") as image:
            image.save(path, format=fmt)
        return path

    def test_unsupported_extension_is_rejected(self) -> None:
        path = self.image("image.gif", fmt="GIF")
        with self.assertRaisesRegex(ValueError, "JPG и PNG"):
            utils.validate_image_file(path)

    def test_temporary_file_requires_the_original_extension(self) -> None:
        path = self.image("image.tmp")
        with self.assertRaisesRegex(ValueError, "JPG и PNG"):
            utils.validate_image_file(path)
        self.assertEqual(utils.validate_image_file(path, ".PNG"), ("image/png", 16, 16))

    def test_uppercase_jpeg_extension(self) -> None:
        path = self.image("image.JPEG", fmt="JPEG")
        self.assertEqual(utils.validate_image_file(path), ("image/jpeg", 16, 16))

    def test_unsupported_content_format_is_rejected(self) -> None:
        path = self.image("image.png", fmt="GIF")
        with self.assertRaisesRegex(ValueError, "не соответствует"):
            utils.validate_image_file(path)

    def test_exactly_five_megabytes_is_accepted(self) -> None:
        path = self.image("image.png")
        with path.open("ab") as file:
            file.truncate(utils.MAX_IMAGE_SIZE_BYTES)
        self.assertEqual(path.stat().st_size, utils.MAX_IMAGE_SIZE_BYTES)
        self.assertEqual(utils.validate_image_file(path), ("image/png", 16, 16))

    def test_exactly_5000_by_5000_is_accepted(self) -> None:
        path = self.image("image.png", (5000, 5000))
        self.assertEqual(utils.validate_image_file(path), ("image/png", 5000, 5000))

    def test_height_over_5000_is_rejected(self) -> None:
        path = self.image("image.png", (16, 5001))
        with self.assertRaisesRegex(ValueError, "5000x5000"):
            utils.validate_image_file(path)

    def test_decompression_bomb_is_rejected(self) -> None:
        path = self.image("image.png")
        with (
            patch.object(Image, "MAX_IMAGE_PIXELS", 1),
            self.assertRaisesRegex(ValueError, "корректным изображением"),
        ):
            utils.validate_image_file(path)

    def test_corrupted_png_is_rejected(self) -> None:
        path = self.image("image.png")
        data = bytearray(path.read_bytes())
        data[29] ^= 1
        path.write_bytes(data)
        with self.assertRaises(ValueError):
            utils.validate_image_file(path)
