"""Exercise real handlers without Telegram, AI, a database, or local secrets.

Only external-service bindings are replaced. Image validation, prompt generation,
password hashing, Pillow decoding, and filesystem operations use their real code.
All filesystem work takes place in an automatically removed scratch directory.
"""

import asyncio
import base64
import io
import os
import tempfile
import unittest
from contextlib import chdir
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

from PIL import Image

from tests.support import import_application

# Invalid-image checks otherwise trigger a one-off decoder scan inside an
# asyncio debug task. Initialize the real decoders before the async tests run.
Image.init()

handlers = import_application("app.core.handlers")


def image_bytes(image_format="JPEG", size=(4, 3)):
    """Produce genuine, small image fixtures instead of mocking the decoder."""
    with Image.new("RGB", size, color=(20, 70, 120)) as image, io.BytesIO() as output:
        image.save(output, format=image_format)
        return output.getvalue()


class HandlerTestCase(unittest.IsolatedAsyncioTestCase):
    """Provide sandboxed files and strict, local service boundaries."""

    user_id = 123456789
    description = "Кот сидит у окна."
    auth_required = "Сначала авторизуйтесь с помощью /start"
    processing = "🤖 Обрабатываю изображение..."

    def setUp(self):
        scratch = self.enterContext(
            tempfile.TemporaryDirectory(
                prefix="handlers-tests-", dir=os.environ.get("TMPDIR")
            )
        )
        self.root = Path(scratch).resolve()
        # search_images_handler deliberately uses a relative app/pictures path.
        # Changing cwd prevents even its existence checks touching real pictures.
        self.enterContext(chdir(self.root))
        self.pictures = self.root / "app" / "pictures"
        self.enterContext(patch.object(handlers, "PICTURES_DIR", str(self.pictures)))

        self.get_user = self.enterContext(
            patch.object(
                handlers.req,
                "get_user_by_id",
                AsyncMock(return_value=SimpleNamespace(user_id=self.user_id)),
            )
        )
        self.add_user = self.enterContext(
            patch.object(handlers.req, "add_user", AsyncMock())
        )
        self.get_all = self.enterContext(
            patch.object(handlers.req, "get_all_image_descriptions", AsyncMock(return_value=[]))
        )
        self.get_processed = self.enterContext(
            patch.object(handlers.req, "get_processed_image_ids", AsyncMock(return_value=set()))
        )
        self.add_processed = self.enterContext(
            patch.object(handlers.req, "add_processed_image_description", AsyncMock())
        )
        self.create_description = self.enterContext(
            patch.object(handlers.req, "create_image_description", AsyncMock(return_value=True))
        )
        self.manager = SimpleNamespace(
            index_images=AsyncMock(return_value=True),
            get_collection_stats=AsyncMock(return_value={"documents_count": 2}),
            search_images=AsyncMock(return_value=[]),
        )
        self.manager_factory = self.enterContext(
            patch.object(handlers, "LlamaIndexManager", MagicMock(return_value=self.manager))
        )
        self.ai = self.enterContext(
            patch.object(handlers, "ai_generate", AsyncMock(return_value=self.description))
        )
        self.prompt = self.enterContext(
            patch.object(handlers, "generate_prompt", wraps=handlers.generate_prompt)
        )
        self.validate = self.enterContext(
            patch.object(handlers.ut, "validate_image_file", wraps=handlers.ut.validate_image_file)
        )
        self.print_error = self.enterContext(
            patch.object(handlers, "print", MagicMock(), create=True)
        )

        self.payload = image_bytes()
        self.downloaded_paths = []
        self.file = SimpleNamespace(file_path="photos/source.JPG")
        self.bot = SimpleNamespace(
            get_file=AsyncMock(return_value=self.file),
            download=AsyncMock(side_effect=self.download_to_sandbox),
        )
        self.message = SimpleNamespace(
            from_user=SimpleNamespace(id=self.user_id, username="test_user"),
            text="/search кот у окна",
            answer=AsyncMock(),
            answer_photo=AsyncMock(),
            photo=[
                SimpleNamespace(file_id="small-id", file_unique_id="small-unique"),
                SimpleNamespace(file_id="large-id", file_unique_id="large-unique"),
            ],
            bot=self.bot,
        )
        self.state = SimpleNamespace(set_state=AsyncMock(), clear=AsyncMock())

    async def download_to_sandbox(self, file_id, *, destination):
        path = Path(destination)
        self.assertEqual(path.parent, self.pictures)
        self.assertEqual(path.suffix, ".tmp")
        self.assertTrue(path.name.startswith(".large-unique-"))
        self.downloaded_paths.append(path)
        path.write_bytes(self.payload)

    def assert_answers(self, *expected):
        self.assertEqual(
            self.message.answer.await_args_list,
            [call(text) for text in expected],
        )

    def assert_files(self, *expected):
        actual = sorted(
            path.relative_to(self.pictures).as_posix()
            for path in self.pictures.rglob("*")
            if path.is_file()
        )
        self.assertEqual(actual, sorted(expected))
        self.assertFalse(any(name.endswith(".tmp") for name in actual))

    def assert_no_processing(self):
        self.prompt.assert_not_called()
        self.ai.assert_not_awaited()
        self.create_description.assert_not_awaited()
        self.manager_factory.assert_not_called()

    def uploaded_text(self, extension=".jpg"):
        return f"📸 Изображение загружено как large-unique{extension}"

    def success_text(self):
        return f"✅ Изображение успешно обработано!\n\n📝 Описание:\n{self.description}"

    async def assert_successful_upload(self, source_extension=".JPG"):
        self.file.file_path = f"photos/source{source_extension}"
        extension = source_extension.lower()
        mime_type = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}[
            extension
        ]
        destination = self.pictures / f"large-unique{extension}"

        await handlers.photo_upload_handler(self.message)

        self.get_user.assert_awaited_once_with(self.user_id)
        self.bot.get_file.assert_awaited_once_with("large-id")
        self.assertEqual(len(self.downloaded_paths), 1)
        temporary_path = self.downloaded_paths[0]
        self.bot.download.assert_awaited_once_with("large-id", destination=str(temporary_path))
        self.validate.assert_called_once_with(temporary_path, extension)
        self.assertEqual(temporary_path.suffix, ".tmp")
        self.assertFalse(temporary_path.exists())
        self.prompt.assert_called_once_with(str(destination), mime_type)
        self.ai.assert_awaited_once()
        ai_call = self.ai.await_args
        assert ai_call is not None
        generated_prompt = ai_call.args[0]
        self.assertEqual(generated_prompt[0]["role"], "user")
        self.assertEqual(
            generated_prompt[0]["content"][1]["image_url"]["url"],
            f"data:{mime_type};base64,{base64.b64encode(self.payload).decode('ascii')}",
        )
        self.create_description.assert_awaited_once_with(destination.name, self.description)
        self.assert_answers(self.uploaded_text(extension), self.processing, self.success_text())
        self.assertEqual(destination.read_bytes(), self.payload)
        self.assert_files(destination.name)
        self.manager_factory.assert_not_called()
        self.print_error.assert_not_called()

    async def assert_rejected_upload(self, expected_message, *preserved_files):
        await handlers.photo_upload_handler(self.message)
        self.assert_answers(expected_message)
        self.assert_no_processing()
        self.assert_files(*preserved_files)


class RegistrationHandlerTests(HandlerTestCase):
    async def test_start_registered_user_does_not_change_state(self):
        await handlers.start_handler(self.message, self.state)
        self.get_user.assert_awaited_once_with(self.user_id)
        self.assert_answers("Добро пожаловать!")
        self.state.set_state.assert_not_awaited()
        self.state.clear.assert_not_awaited()
        self.add_user.assert_not_awaited()

    async def test_start_unknown_user_requests_password(self):
        self.get_user.return_value = None
        await handlers.start_handler(self.message, self.state)
        self.assert_answers(
            "Добро пожаловать! Для продолжения работы введите пароль для доступа."
        )
        self.state.set_state.assert_awaited_once_with(handlers.RegisterState.waiting_for_password)
        self.state.clear.assert_not_awaited()

    async def test_help_lists_commands_as_html(self):
        await handlers.help_handler(self.message)
        self.message.answer.assert_awaited_once()
        args, kwargs = self.message.answer.await_args
        self.assertEqual(kwargs, {"parse_mode": "HTML"})
        for command in ("/start", "/help", "/search &lt;запрос&gt;", "/sync", "/stats"):
            self.assertIn(command, args[0])
        self.assertIn("После загрузки выполните /sync", args[0])
        self.get_user.assert_not_awaited()
        self.assert_no_processing()

    async def test_correct_password_is_stripped_and_real_hash_is_checked(self):
        password = "test-only-password"
        self.message.text = f" \t{password}\n"
        with patch.object(handlers, "ACCESS_PASSWORD", handlers.ut.hash_password(password)):
            await handlers.password_handler(self.message, self.state)
        self.add_user.assert_awaited_once_with(self.user_id, "test_user")
        self.assert_answers("Авторизация успешна! Теперь у вас полный доступ.")
        self.state.clear.assert_awaited_once_with()
        self.state.set_state.assert_not_awaited()

    async def test_correct_password_without_username_stores_empty_string(self):
        password = "another-test-only-password"
        self.message.text = password
        for username in (None, ""):
            with self.subTest(username=username):
                self.message.from_user.username = username
                self.add_user.reset_mock()
                self.state.clear.reset_mock()
                self.message.answer.reset_mock()
                with patch.object(handlers, "ACCESS_PASSWORD", handlers.ut.hash_password(password)):
                    await handlers.password_handler(self.message, self.state)
                self.add_user.assert_awaited_once_with(self.user_id, "")
                self.state.clear.assert_awaited_once_with()
                self.assert_answers("Авторизация успешна! Теперь у вас полный доступ.")

    async def test_wrong_password_preserves_registration_state(self):
        self.message.text = "wrong-test-password"
        expected_hash = handlers.ut.hash_password("correct-test-password")
        with patch.object(handlers, "ACCESS_PASSWORD", expected_hash):
            await handlers.password_handler(self.message, self.state)
        self.assert_answers("Неверный пароль. Попробуйте еще раз:")
        self.add_user.assert_not_awaited()
        self.state.clear.assert_not_awaited()
        self.state.set_state.assert_not_awaited()

    async def test_registration_error_does_not_clear_state_or_claim_success(self):
        self.message.text = "test-only-password"
        self.add_user.side_effect = RuntimeError("registration failed")
        with (
            patch.object(handlers, "ACCESS_PASSWORD", handlers.ut.hash_password(self.message.text)),
            self.assertRaisesRegex(RuntimeError, "registration failed"),
        ):
            await handlers.password_handler(self.message, self.state)
        self.state.clear.assert_not_awaited()
        self.message.answer.assert_not_awaited()


class SyncHandlerTests(HandlerTestCase):
    async def test_sync_requires_authorization(self):
        self.get_user.return_value = None
        await handlers.sync_images_handler(self.message)
        self.assert_answers(self.auth_required)
        self.get_all.assert_not_awaited()
        self.get_processed.assert_not_awaited()
        self.manager_factory.assert_not_called()

    async def test_sync_empty_database_does_not_construct_index(self):
        await handlers.sync_images_handler(self.message)
        self.assert_answers("🔄 Начинаю синхронизацию...", "✅ Нет новых данных для синхронизации")
        self.get_all.assert_awaited_once_with()
        self.get_processed.assert_awaited_once_with()
        self.manager_factory.assert_not_called()
        self.add_processed.assert_not_awaited()

    async def test_sync_all_duplicates_does_not_reindex(self):
        self.get_all.return_value = [SimpleNamespace(id=1), SimpleNamespace(id=2)]
        self.get_processed.return_value = {1, 2}
        await handlers.sync_images_handler(self.message)
        self.assert_answers("🔄 Начинаю синхронизацию...", "✅ Нет новых данных для синхронизации")
        self.manager_factory.assert_not_called()
        self.add_processed.assert_not_awaited()

    async def test_sync_indexes_only_new_records_then_marks_original_records_processed(self):
        old = SimpleNamespace(id=1, name="old.jpg", description="old")
        first = SimpleNamespace(id=2, name="first.png", description="first description")
        second = SimpleNamespace(id=3, name="second.jpeg", description="second description")
        self.get_all.return_value = [old, first, second]
        self.get_processed.return_value = {1}
        events = MagicMock()
        events.attach_mock(self.manager.index_images, "index")
        events.attach_mock(self.add_processed, "mark")

        await handlers.sync_images_handler(self.message)

        payload = [
            {"id": 2, "name": "first.png", "description": "first description"},
            {"id": 3, "name": "second.jpeg", "description": "second description"},
        ]
        self.manager_factory.assert_called_once_with()
        self.manager.index_images.assert_awaited_once_with(payload)
        self.assertEqual(self.add_processed.await_args_list, [call(first), call(second)])
        self.assertEqual(
            events.mock_calls, [call.index(payload), call.mark(first), call.mark(second)]
        )
        self.assert_answers(
            "🔄 Начинаю синхронизацию...",
            "📥 Найдено 2 новых записей",
            "✅ Синхронизация завершена!\nДобавлено: 2 записей\nВсего в базе: 3 записей",
        )
        self.assert_files()

    async def test_sync_failed_index_does_not_mark_records_processed(self):
        self.get_all.return_value = [SimpleNamespace(id=1, name="cat.jpg", description="cat")]
        self.manager.index_images.return_value = False
        await handlers.sync_images_handler(self.message)
        self.assert_answers(
            "🔄 Начинаю синхронизацию...",
            "📥 Найдено 1 новых записей",
            "❌ Ошибка при индексации в ChromaDB",
        )
        self.add_processed.assert_not_awaited()

    async def test_sync_reports_failures_at_each_service_boundary(self):
        self.get_all.return_value = [SimpleNamespace(id=1, name="cat.jpg", description="cat")]
        for boundary in (
            self.get_all,
            self.get_processed,
            self.manager_factory,
            self.manager.index_images,
            self.add_processed,
        ):
            with self.subTest(boundary=boundary._mock_name):
                self.message.answer.reset_mock()
                boundary.side_effect = RuntimeError("sync failed")
                try:
                    await handlers.sync_images_handler(self.message)
                finally:
                    boundary.side_effect = None
                self.assertEqual(
                    self.message.answer.await_args_list[-1],
                    call("❌ Ошибка синхронизации: sync failed"),
                )
                self.assertEqual(
                    self.message.answer.await_args_list[0], call("🔄 Начинаю синхронизацию...")
                )


class StatsHandlerTests(HandlerTestCase):
    async def test_stats_requires_authorization(self):
        self.get_user.return_value = None
        await handlers.stats_handler(self.message)
        self.assert_answers(self.auth_required)
        self.get_all.assert_not_awaited()
        self.get_processed.assert_not_awaited()
        self.manager_factory.assert_not_called()

    async def test_stats_reports_database_processed_pending_and_chroma_counts(self):
        self.get_all.return_value = [SimpleNamespace(id=index) for index in (1, 2, 3)]
        self.get_processed.return_value = {1}
        self.manager.get_collection_stats.return_value = {"documents_count": 7}
        await handlers.stats_handler(self.message)
        self.assert_answers(
            "📊 Статистика базы данных:\n"
            "• Всего записей в БД: 3\n"
            "• Обработано записей: 1\n"
            "• Новых для обработки: 2\n"
            "• В ChromaDB: 7\n"
        )
        self.manager_factory.assert_called_once_with()
        self.manager.get_collection_stats.assert_awaited_once_with()

    async def test_stats_missing_chroma_count_uses_na(self):
        self.manager.get_collection_stats.return_value = {"error": "collection unavailable"}
        await handlers.stats_handler(self.message)
        self.assert_answers(
            "📊 Статистика базы данных:\n"
            "• Всего записей в БД: 0\n"
            "• Обработано записей: 0\n"
            "• Новых для обработки: 0\n"
            "• В ChromaDB: N/A\n"
        )

    async def test_stats_reports_failures_at_each_service_boundary(self):
        for boundary in (
            self.get_all,
            self.get_processed,
            self.manager_factory,
            self.manager.get_collection_stats,
        ):
            with self.subTest(boundary=boundary._mock_name):
                self.message.answer.reset_mock()
                boundary.side_effect = RuntimeError("statistics failed")
                try:
                    await handlers.stats_handler(self.message)
                finally:
                    boundary.side_effect = None
                self.assert_answers("❌ Ошибка получения статистики: statistics failed")


class SearchHandlerTests(HandlerTestCase):
    async def test_search_requires_authorization_before_parsing_query(self):
        self.get_user.return_value = None
        self.message.text = None
        await handlers.search_images_handler(self.message)
        self.assert_answers(self.auth_required)
        self.manager_factory.assert_not_called()

    async def test_search_missing_or_whitespace_query_requests_input(self):
        for text in ("/search", "/search \t\n", "", " \t "):
            with self.subTest(text=text):
                self.message.text = text
                self.message.answer.reset_mock()
                await handlers.search_images_handler(self.message)
                self.assert_answers("Введите запрос для поиска:\n/search <ваш запрос>")
                self.manager_factory.assert_not_called()

    async def test_search_trims_query_and_requests_five_results(self):
        self.message.text = "/search   кот у окна \n"
        await handlers.search_images_handler(self.message)
        self.manager_factory.assert_called_once_with()
        self.manager.search_images.assert_awaited_once_with("кот у окна", limit=5)
        self.assert_answers('🔍 Ищу: "кот у окна"', "😔 Ничего не найдено")
        self.message.answer_photo.assert_not_awaited()

    async def test_search_sends_existing_files_and_continues_after_missing_file(self):
        self.pictures.mkdir(parents=True)
        first = self.pictures / "first.jpg"
        second = self.pictures / "second.png"
        first.write_bytes(self.payload)
        second.write_bytes(image_bytes("PNG"))
        self.manager.search_images.return_value = [
            {"name": "first.jpg"}, {"name": "missing.png"}, {"name": "second.png"}
        ]

        await handlers.search_images_handler(self.message)

        self.assert_answers(
            '🔍 Ищу: "кот у окна"', "❌ Файл missing.png не найден в папке pictures"
        )
        sent = self.message.answer_photo.await_args_list
        self.assertEqual(len(sent), 2)
        for sent_call, filename in zip(sent, ("first.jpg", "second.png"), strict=True):
            self.assertIsInstance(sent_call.args[0], handlers.FSInputFile)
            self.assertEqual(
                Path(sent_call.args[0].path).resolve(), (self.pictures / filename).resolve()
            )
            self.assertEqual(sent_call.kwargs, {"caption": f"📸 {filename}"})
        self.assertEqual(first.read_bytes(), self.payload)
        self.assert_files("first.jpg", "second.png")

    async def test_search_reports_manager_or_query_errors(self):
        for boundary in (self.manager_factory, self.manager.search_images):
            with self.subTest(boundary=boundary._mock_name):
                self.message.answer.reset_mock()
                boundary.side_effect = RuntimeError("search failed")
                try:
                    await handlers.search_images_handler(self.message)
                finally:
                    boundary.side_effect = None
                self.assert_answers('🔍 Ищу: "кот у окна"', "❌ Ошибка поиска: search failed")

    async def test_search_reports_photo_send_error(self):
        self.pictures.mkdir(parents=True)
        (self.pictures / "cat.jpg").write_bytes(self.payload)
        self.manager.search_images.return_value = [{"name": "cat.jpg"}]
        self.message.answer_photo.side_effect = RuntimeError("photo delivery failed")
        await handlers.search_images_handler(self.message)
        self.assert_answers(
            '🔍 Ищу: "кот у окна"', "❌ Ошибка поиска: photo delivery failed"
        )
        self.assert_files("cat.jpg")


class PhotoUploadHandlerTests(HandlerTestCase):
    async def test_upload_requires_authorization_before_accessing_photo_or_bot(self):
        self.get_user.return_value = None
        self.message.photo = None
        self.message.bot = None
        await self.assert_rejected_upload(self.auth_required)
        self.bot.get_file.assert_not_awaited()
        self.bot.download.assert_not_awaited()
        self.assertFalse(self.pictures.exists())

    async def test_upload_without_photos_does_not_call_telegram(self):
        for photos in (None, []):
            with self.subTest(photos=photos):
                self.message.photo = photos
                self.message.answer.reset_mock()
                await self.assert_rejected_upload("❌ Не удалось получить изображение")
                self.bot.get_file.assert_not_awaited()
                self.bot.download.assert_not_awaited()
                self.assertFalse(self.pictures.exists())

    async def test_upload_without_bot_reports_api_unavailable(self):
        self.message.bot = None
        await self.assert_rejected_upload("❌ Не удалось получить доступ к Telegram API")
        self.bot.get_file.assert_not_awaited()
        self.assertFalse(self.pictures.exists())

    async def test_upload_without_remote_file_path_does_not_create_files(self):
        for file_path in (None, ""):
            with self.subTest(file_path=file_path):
                self.file.file_path = file_path
                self.message.answer.reset_mock()
                await self.assert_rejected_upload("❌ Не удалось получить путь к файлу")
                self.bot.download.assert_not_awaited()
                self.assertFalse(self.pictures.exists())

    async def test_upload_rejects_unsupported_or_missing_extensions_before_download(self):
        for file_path in (
            "photos/source", "photos/.jpg", "photos/file.gif", "photos/file.BMP",
            "photos/file.webp", "photos/file.pdf", "photos/file.jpg.exe",
        ):
            with self.subTest(file_path=file_path):
                self.file.file_path = file_path
                self.message.answer.reset_mock()
                await handlers.photo_upload_handler(self.message)
                self.message.answer.assert_awaited_once()
                text = self.message.answer.await_args.args[0]
                prefix = "❌ Недопустимое расширение файла. Поддерживаемые форматы: "
                self.assertTrue(text.startswith(prefix))
                self.assertEqual(
                    set(text.removeprefix(prefix).split(", ")), handlers.ALLOWED_EXTENSIONS
                )
                self.bot.download.assert_not_awaited()
                self.validate.assert_not_called()
                self.assert_no_processing()
                self.assertFalse(self.pictures.exists())

    async def test_upload_jpeg_tmp_suffix_regression_uses_source_extension_and_detected_mime(self):
        await self.assert_successful_upload(".JPG")

    async def test_upload_jpeg_long_extension_is_supported(self):
        await self.assert_successful_upload(".JPEG")

    async def test_upload_png_tmp_suffix_regression_generates_png_data_url(self):
        self.payload = image_bytes("PNG")
        await self.assert_successful_upload(".PNG")

    async def test_upload_accepts_exact_byte_size_limit(self):
        self.payload += b"\0" * (handlers.ut.MAX_IMAGE_SIZE_BYTES - len(self.payload))
        await self.assert_successful_upload(".jpg")

    async def test_upload_accepts_exact_width_limit(self):
        self.payload = image_bytes("PNG", (handlers.ut.MAX_IMAGE_DIMENSION, 1))
        await self.assert_successful_upload(".png")

    async def test_upload_accepts_exact_height_limit(self):
        self.payload = image_bytes("PNG", (1, handlers.ut.MAX_IMAGE_DIMENSION))
        await self.assert_successful_upload(".png")

    async def test_upload_rejects_oversize_file_and_removes_tmp_before_ai(self):
        self.payload += b"\0" * (handlers.ut.MAX_IMAGE_SIZE_BYTES + 1 - len(self.payload))
        await self.assert_rejected_upload("❌ Размер изображения не должен превышать 5 МБ")
        self.validate.assert_called_once()
        self.assertFalse(self.downloaded_paths[0].exists())
        self.print_error.assert_not_called()

    async def test_upload_rejects_excessive_width(self):
        self.payload = image_bytes("PNG", (handlers.ut.MAX_IMAGE_DIMENSION + 1, 1))
        self.file.file_path = "photos/source.png"
        await self.assert_rejected_upload(
            "❌ Разрешение изображения не должно превышать 5000x5000 пикселей"
        )

    async def test_upload_rejects_excessive_height(self):
        self.payload = image_bytes("PNG", (1, handlers.ut.MAX_IMAGE_DIMENSION + 1))
        self.file.file_path = "photos/source.png"
        await self.assert_rejected_upload(
            "❌ Разрешение изображения не должно превышать 5000x5000 пикселей"
        )

    async def test_upload_rejects_corrupt_empty_or_truncated_content(self):
        for payload in (b"not an image", b"", image_bytes()[:30]):
            with self.subTest(payload_size=len(payload)):
                self.payload = payload
                self.message.answer.reset_mock()
                await self.assert_rejected_upload("❌ Файл не является корректным изображением")
                self.assertFalse(self.downloaded_paths[-1].exists())
        self.print_error.assert_not_called()

    async def test_upload_rejects_mime_mismatch_and_unsupported_image_content(self):
        for extension, image_format in ((".jpg", "PNG"), (".png", "JPEG"), (".jpg", "GIF")):
            with self.subTest(extension=extension, image_format=image_format):
                self.file.file_path = f"photos/source{extension}"
                self.payload = image_bytes(image_format)
                self.message.answer.reset_mock()
                await self.assert_rejected_upload(
                    "❌ Расширение файла не соответствует его содержимому"
                )
                self.assertFalse(self.downloaded_paths[-1].exists())

    async def test_upload_duplicate_preserves_existing_target_and_removes_tmp(self):
        self.pictures.mkdir(parents=True)
        existing = self.pictures / "large-unique.jpg"
        original = b"pre-existing target must not be replaced or removed"
        existing.write_bytes(original)
        await self.assert_rejected_upload(
            "❌ Файл с таким именем уже существует: large-unique.jpg. "
            "Пожалуйста, отправьте другое изображение.",
            existing.name,
        )
        self.validate.assert_called_once()
        self.assertEqual(existing.read_bytes(), original)
        self.assertFalse(self.downloaded_paths[0].exists())

    async def test_upload_validation_error_also_preserves_existing_target(self):
        self.pictures.mkdir(parents=True)
        existing = self.pictures / "large-unique.jpg"
        original = image_bytes()
        existing.write_bytes(original)
        self.payload = b"corrupt incoming upload"
        await self.assert_rejected_upload(
            "❌ Файл не является корректным изображением", existing.name
        )
        self.assertEqual(existing.read_bytes(), original)

    async def test_upload_get_file_error_creates_nothing(self):
        self.bot.get_file.side_effect = RuntimeError("metadata failed")
        await self.assert_rejected_upload("❌ Ошибка при обработке изображения: metadata failed")
        self.bot.download.assert_not_awaited()
        self.assertFalse(self.pictures.exists())
        self.print_error.assert_called_once_with("Ошибка в photo_upload_handler: metadata failed")

    async def test_upload_directory_creation_error_creates_nothing(self):
        with patch.object(handlers.os, "makedirs", side_effect=PermissionError("directory denied")):
            await self.assert_rejected_upload(
                "❌ Ошибка при обработке изображения: directory denied"
            )
        self.bot.download.assert_not_awaited()
        self.assertFalse(self.pictures.exists())

    async def test_upload_temporary_file_creation_error_creates_nothing(self):
        with patch.object(
            handlers.tempfile, "NamedTemporaryFile", side_effect=OSError("temp failed")
        ):
            await self.assert_rejected_upload("❌ Ошибка при обработке изображения: temp failed")
        self.bot.download.assert_not_awaited()

    async def test_upload_download_error_removes_even_empty_tmp(self):
        self.bot.download.side_effect = RuntimeError("download failed")
        await self.assert_rejected_upload("❌ Ошибка при обработке изображения: download failed")
        self.validate.assert_not_called()
        self.print_error.assert_called_once_with("Ошибка в photo_upload_handler: download failed")

    async def test_upload_partial_download_error_preserves_existing_target(self):
        self.pictures.mkdir(parents=True)
        existing = self.pictures / "large-unique.jpg"
        original = b"existing target"
        existing.write_bytes(original)

        async def partial_download(file_id, *, destination):
            self.payload = b"partial bytes"
            await self.download_to_sandbox(file_id, destination=destination)
            raise RuntimeError("interrupted download")

        self.bot.download.side_effect = partial_download
        await self.assert_rejected_upload(
            "❌ Ошибка при обработке изображения: interrupted download", existing.name
        )
        self.assertEqual(existing.read_bytes(), original)
        self.assertFalse(self.downloaded_paths[0].exists())
        self.validate.assert_not_called()

    async def test_upload_rename_error_removes_tmp_without_starting_ai(self):
        with patch.object(Path, "replace", side_effect=PermissionError("rename denied")):
            await self.assert_rejected_upload("❌ Ошибка при обработке изображения: rename denied")
        self.validate.assert_called_once()
        self.assertFalse(self.downloaded_paths[0].exists())

    async def test_upload_prompt_error_removes_promoted_file(self):
        self.prompt.side_effect = OSError("image encoding failed")
        await handlers.photo_upload_handler(self.message)
        self.assert_answers(
            self.uploaded_text(), self.processing,
            "❌ Ошибка при обработке изображения: image encoding failed",
        )
        self.ai.assert_not_awaited()
        self.create_description.assert_not_awaited()
        self.assert_files()

    async def test_upload_empty_ai_description_removes_promoted_file(self):
        for empty_description in (None, ""):
            with self.subTest(description=empty_description):
                self.ai.return_value = empty_description
                self.ai.reset_mock()
                self.message.answer.reset_mock()
                await handlers.photo_upload_handler(self.message)
                self.assert_answers(
                    self.uploaded_text(), self.processing, "❌ Не удалось обработать изображение"
                )
                self.ai.assert_awaited_once()
                self.create_description.assert_not_awaited()
                self.assert_files()
        self.print_error.assert_not_called()

    async def test_upload_ai_exception_removes_promoted_file(self):
        self.ai.side_effect = RuntimeError("AI unavailable")
        await handlers.photo_upload_handler(self.message)
        self.assert_answers(
            self.uploaded_text(), self.processing,
            "❌ Ошибка при обработке изображения: AI unavailable",
        )
        self.create_description.assert_not_awaited()
        self.assert_files()
        self.print_error.assert_called_once_with("Ошибка в photo_upload_handler: AI unavailable")

    async def test_upload_ai_value_error_uses_validation_error_message_and_cleans_file(self):
        self.ai.side_effect = ValueError("invalid AI request")
        await handlers.photo_upload_handler(self.message)
        self.assert_answers(self.uploaded_text(), self.processing, "❌ invalid AI request")
        self.create_description.assert_not_awaited()
        self.print_error.assert_not_called()
        self.assert_files()

    async def test_upload_database_false_removes_promoted_file(self):
        self.create_description.return_value = False
        await handlers.photo_upload_handler(self.message)
        self.assert_answers(
            self.uploaded_text(), self.processing,
            "❌ Не удалось сохранить описание изображения в базу данных",
        )
        self.create_description.assert_awaited_once_with("large-unique.jpg", self.description)
        self.assert_files()

    async def test_upload_database_exception_removes_promoted_file(self):
        self.create_description.side_effect = RuntimeError("database unavailable")
        await handlers.photo_upload_handler(self.message)
        self.assert_answers(
            self.uploaded_text(), self.processing,
            "❌ Ошибка при обработке изображения: database unavailable",
        )
        self.assert_files()

    async def test_upload_failed_acknowledgement_cleans_promoted_file_before_ai(self):
        self.message.answer.side_effect = [RuntimeError("reply failed"), None]
        await handlers.photo_upload_handler(self.message)
        self.assert_answers(
            self.uploaded_text(), "❌ Ошибка при обработке изображения: reply failed"
        )
        self.assert_no_processing()
        self.assert_files()

    async def test_upload_final_reply_error_preserves_successfully_saved_image(self):
        self.message.answer.side_effect = [None, None, RuntimeError("final reply failed"), None]
        await handlers.photo_upload_handler(self.message)
        self.assert_answers(
            self.uploaded_text(), self.processing, self.success_text(),
            "❌ Ошибка при обработке изображения: final reply failed",
        )
        self.create_description.assert_awaited_once_with("large-unique.jpg", self.description)
        self.assertEqual((self.pictures / "large-unique.jpg").read_bytes(), self.payload)
        self.assert_files("large-unique.jpg")

    async def test_upload_download_cancellation_propagates_and_removes_partial_tmp(self):
        async def cancelled_download(file_id, *, destination):
            await self.download_to_sandbox(file_id, destination=destination)
            raise asyncio.CancelledError()

        self.bot.download.side_effect = cancelled_download
        with self.assertRaises(asyncio.CancelledError):
            await handlers.photo_upload_handler(self.message)
        self.message.answer.assert_not_awaited()
        self.assert_no_processing()
        self.validate.assert_not_called()
        self.assertFalse(self.downloaded_paths[0].exists())
        self.assert_files()
        self.print_error.assert_not_called()

    async def test_upload_ai_cancellation_propagates_and_removes_promoted_file(self):
        self.ai.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await handlers.photo_upload_handler(self.message)
        self.assert_answers(self.uploaded_text(), self.processing)
        self.create_description.assert_not_awaited()
        self.assert_files()
        self.print_error.assert_not_called()

    async def test_upload_database_cancellation_propagates_and_removes_promoted_file(self):
        self.create_description.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await handlers.photo_upload_handler(self.message)
        self.assert_answers(self.uploaded_text(), self.processing)
        self.assert_files()
        self.print_error.assert_not_called()

    async def test_upload_final_reply_cancellation_preserves_committed_file(self):
        self.message.answer.side_effect = [None, None, asyncio.CancelledError()]
        with self.assertRaises(asyncio.CancelledError):
            await handlers.photo_upload_handler(self.message)
        self.assert_answers(self.uploaded_text(), self.processing, self.success_text())
        self.create_description.assert_awaited_once_with("large-unique.jpg", self.description)
        self.assertEqual((self.pictures / "large-unique.jpg").read_bytes(), self.payload)
        self.assert_files("large-unique.jpg")

    async def test_upload_missing_tmp_during_cleanup_is_harmless(self):
        async def removed_download(file_id, *, destination):
            await self.download_to_sandbox(file_id, destination=destination)
            Path(destination).unlink()
            raise RuntimeError("download vanished")

        self.bot.download.side_effect = removed_download
        await self.assert_rejected_upload("❌ Ошибка при обработке изображения: download vanished")
        self.assertFalse(self.downloaded_paths[0].exists())

    async def test_upload_tmp_cleanup_oserror_does_not_mask_validation_failure(self):
        self.payload = b"not an image"
        with patch.object(
            Path, "unlink", autospec=True, side_effect=PermissionError("locked")
        ) as unlink:
            await handlers.photo_upload_handler(self.message)
            self.assert_answers("❌ Файл не является корректным изображением")
            self.assert_no_processing()
            unlink.assert_called_once_with(self.downloaded_paths[0], missing_ok=True)
        # The injected OS failure necessarily leaves a file; release it and clean
        # the test-owned artifact with the real filesystem implementation.
        self.assertTrue(self.downloaded_paths[0].exists())
        self.downloaded_paths[0].unlink()
        self.assert_files()
        self.print_error.assert_not_called()

    async def test_upload_owned_file_cleanup_oserror_does_not_mask_empty_ai_result(self):
        self.ai.return_value = None
        destination = self.pictures / "large-unique.jpg"
        with patch.object(
            Path, "unlink", autospec=True, side_effect=PermissionError("locked")
        ) as unlink:
            await handlers.photo_upload_handler(self.message)
            self.assert_answers(
                self.uploaded_text(), self.processing, "❌ Не удалось обработать изображение"
            )
            unlink.assert_called_once_with(destination, missing_ok=True)
        self.assertTrue(destination.exists())
        destination.unlink()
        self.assert_files()
        self.create_description.assert_not_awaited()
        self.print_error.assert_not_called()


if __name__ == "__main__":
    unittest.main()
