"""Offline unit tests for the real image prompt and AI response code."""

import asyncio
import base64
import os
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from openai import APIConnectionError, APIError, BadRequestError
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from PIL import Image

from tests.support import import_application

image_to_text = import_application("app.core.image_to_text")


class _NoNetworkMixin(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        # Fail even if application exception handling would swallow a connection attempt.
        for target, attribute in (
            (socket, "create_connection"),
            (socket, "getaddrinfo"),
            (socket.socket, "connect"),
            (socket.socket, "connect_ex"),
        ):
            patcher = patch.object(
                target, attribute, side_effect=AssertionError("Network access is forbidden")
            )
            guard = patcher.start()
            self.addCleanup(patcher.stop)
            self.addCleanup(guard.assert_not_called)


class AiGenerateTests(_NoNetworkMixin, unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.messages = [{"role": "user", "content": "Опиши картинку"}]
        self.create = AsyncMock(name="chat.completions.create")
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
        )
        client_patcher = patch.object(image_to_text, "client", client)
        client_patcher.start()
        self.addCleanup(client_patcher.stop)
        self.request = httpx.Request("POST", "https://provider.example.invalid/v1/chat/completions")

    @staticmethod
    def completion(content: str | None) -> ChatCompletion:
        return ChatCompletion(
            id="local-completion",
            created=0,
            model="google/gemini-2.5-flash-lite",
            object="chat.completion",
            choices=[
                Choice(
                    index=0,
                    finish_reason="stop",
                    message=ChatCompletionMessage(role="assistant", content=content),
                )
            ],
        )

    def assert_request_parameters(self) -> None:
        self.create.assert_awaited_once_with(
            model="google/gemini-2.5-flash-lite",
            messages=self.messages,
            temperature=0.1,
        )
        arguments = self.create.await_args
        assert arguments is not None
        self.assertIs(arguments.kwargs["messages"], self.messages)

    async def assert_reported_failure(self, error: Exception, prefix: str) -> None:
        self.create.side_effect = error
        with patch("builtins.print") as report:
            result = await image_to_text.ai_generate(self.messages)
        self.assertIsNone(result)
        self.assert_request_parameters()
        report.assert_called_once_with(f"{prefix}: {error}")

    async def test_cleans_markdown_and_whitespace_from_real_completion(self) -> None:
        self.create.return_value = self.completion("###**Кошка**\n\nна\t*окне*")
        with patch("builtins.print") as report:
            result = await image_to_text.ai_generate(self.messages)
        self.assertEqual(result, "Кошка на окне")
        self.assert_request_parameters()
        report.assert_not_called()

    async def test_empty_content_is_an_empty_string_not_an_error(self) -> None:
        self.create.return_value = self.completion("")
        with patch("builtins.print") as report:
            result = await image_to_text.ai_generate(self.messages)
        self.assertEqual(result, "")
        self.assert_request_parameters()
        report.assert_not_called()

    async def test_bad_request_is_reported_without_escaping(self) -> None:
        response = httpx.Response(400, request=self.request)
        error = BadRequestError("invalid image", response=response, body={"error": "invalid image"})
        await self.assert_reported_failure(error, "Ошибка запроса к API")

    async def test_connection_error_is_reported_without_escaping(self) -> None:
        error = APIConnectionError(message="provider unavailable", request=self.request)
        await self.assert_reported_failure(error, "Ошибка подключения к API")

    async def test_other_api_error_is_reported_without_escaping(self) -> None:
        error = APIError("provider failed", self.request, body=None)
        await self.assert_reported_failure(error, "Ошибка API")

    async def test_unexpected_client_error_is_reported_without_escaping(self) -> None:
        await self.assert_reported_failure(ValueError("invalid completion"), "Неожиданная ошибка")

    async def test_none_content_is_reported_as_an_unexpected_error(self) -> None:
        self.create.return_value = self.completion(None)
        with patch("builtins.print") as report:
            result = await image_to_text.ai_generate(self.messages)
        self.assertIsNone(result)
        self.assert_request_parameters()
        report.assert_called_once()
        diagnostic = report.call_args.args[0]
        self.assertTrue(diagnostic.startswith("Неожиданная ошибка: "), diagnostic)
        self.assertIn("NoneType", diagnostic)

    async def test_completion_without_choices_is_reported(self) -> None:
        self.create.return_value = self.completion("unused").model_copy(update={"choices": []})
        with patch("builtins.print") as report:
            result = await image_to_text.ai_generate(self.messages)
        self.assertIsNone(result)
        self.assert_request_parameters()
        report.assert_called_once_with("Неожиданная ошибка: list index out of range")

    async def test_message_without_content_attribute_is_reported(self) -> None:
        self.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace())]
        )
        with patch("builtins.print") as report:
            result = await image_to_text.ai_generate(self.messages)
        self.assertIsNone(result)
        self.assert_request_parameters()
        report.assert_called_once()
        diagnostic = report.call_args.args[0]
        self.assertTrue(diagnostic.startswith("Неожиданная ошибка: "), diagnostic)
        self.assertIn("content", diagnostic)

    async def test_task_cancellation_is_not_swallowed_as_an_api_failure(self) -> None:
        self.create.side_effect = asyncio.CancelledError()
        with patch("builtins.print") as report, self.assertRaises(asyncio.CancelledError):
            await image_to_text.ai_generate(self.messages)
        self.assert_request_parameters()
        report.assert_not_called()


class GeneratePromptTests(_NoNetworkMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        # Explicit test-owned storage never uses the project's pictures/ or system temp.
        temporary_directory = tempfile.TemporaryDirectory(
            prefix="image_to_text_", dir=os.environ.get("TMPDIR")
        )
        self.addCleanup(temporary_directory.cleanup)
        self.directory = Path(temporary_directory.name)

    def assert_image_prompt(self, suffix: str, image_format: str, mime_type: str) -> None:
        path = self.directory / f"изображение{suffix}"
        with Image.new("RGB", (3, 2), color=(12, 34, 56)) as image:
            image.save(path, format=image_format)
        original_bytes = path.read_bytes()

        result = image_to_text.generate_prompt(str(path), mime_type)

        self.assertEqual(len(result), 1)
        self.assertEqual(set(result[0]), {"role", "content"})
        self.assertEqual(result[0]["role"], "user")
        content = result[0]["content"]
        self.assertEqual(len(content), 2)
        expected_prompt = (
            "Опиши коротко содержимое этой картинки на русском языке, чтобы потом по "
            "этому описанию можно было его найти.\n"
            "    Требования:\n"
            "    - Начни сразу с описания содержимого\n"
            '    - Не используй вводные фразы типа "Конечно, вот...", '
            "    - Если есть текст в картинке то писать о нем\n"
            '"Это изображение...", "Перед нами..."\n'
            "    - Описывай только то, что видишь на изображении\n"
            "    - Будь точным и конкретным"
        )
        self.assertEqual(content[0], {"type": "text", "text": expected_prompt})
        self.assertEqual(content[1]["type"], "image_url")
        self.assertEqual(set(content[1]), {"type", "image_url"})
        self.assertEqual(set(content[1]["image_url"]), {"url"})
        header, encoded = content[1]["image_url"]["url"].split(",", 1)
        self.assertEqual(header, f"data:{mime_type};base64")
        self.assertEqual(base64.b64decode(encoded, validate=True), original_bytes)
        self.assertEqual(path.read_bytes(), original_bytes)

    def test_png_file_is_embedded_with_png_mime_type(self) -> None:
        self.assert_image_prompt(".png", "PNG", "image/png")

    def test_jpeg_file_is_embedded_with_jpeg_mime_type(self) -> None:
        self.assert_image_prompt(".jpg", "JPEG", "image/jpeg")

    def test_missing_file_error_propagates_to_the_caller(self) -> None:
        missing = self.directory / "missing.png"
        with self.assertRaises(FileNotFoundError) as caught:
            image_to_text.generate_prompt(str(missing), "image/png")
        self.assertEqual(caught.exception.filename, str(missing))

    def test_encoder_failure_propagates_without_building_a_fake_prompt(self) -> None:
        path = str(self.directory / "denied.png")
        error = PermissionError("cannot read test image")
        with (
            patch.object(image_to_text, "encode_image_to_base64", side_effect=error) as encode,
            self.assertRaises(PermissionError) as caught,
        ):
            image_to_text.generate_prompt(path, "image/png")
        self.assertIs(caught.exception, error)
        encode.assert_called_once_with(path)
