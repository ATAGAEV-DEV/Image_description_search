"""Unit tests for persistence behavior and error handling without a database."""

import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import SQLAlchemyError

from tests.support import import_application

requests = import_application("app.data.request")


class RequestTests(unittest.IsolatedAsyncioTestCase):
    """Exercise real queries with a mocked session boundary."""

    def setUp(self) -> None:
        self.session = MagicMock()
        self.session.execute = AsyncMock(return_value=MagicMock())
        self.session.commit = AsyncMock()

        @asynccontextmanager
        async def session_context():
            yield self.session

        self.context = patch.object(requests, "async_session", session_context)
        self.context.start()
        self.addCleanup(self.context.stop)
        self.output = patch("builtins.print")
        self.print_mock = self.output.start()
        self.addCleanup(self.output.stop)

    async def test_get_user_returns_result_and_filters_by_id(self) -> None:
        user = requests.Users(user_id=12, username="reader")
        self.session.execute.return_value.scalar_one_or_none.return_value = user
        self.assertIs(await requests.get_user_by_id(12), user)
        statement = self.session.execute.call_args.args[0]
        self.assertEqual(statement.compile().params, {"user_id_1": 12})

    async def test_get_user_not_found(self) -> None:
        self.session.execute.return_value.scalar_one_or_none.return_value = None
        self.assertIsNone(await requests.get_user_by_id(12))

    async def test_get_user_timeout(self) -> None:
        self.session.execute.side_effect = TimeoutError
        self.assertIsNone(await requests.get_user_by_id(12))
        self.assertIn("Таймаут", self.print_mock.call_args.args[0])

    async def test_get_user_error(self) -> None:
        self.session.execute.side_effect = SQLAlchemyError("unavailable")
        self.assertIsNone(await requests.get_user_by_id(12))
        self.assertIn("Ошибка получения пользователя", self.print_mock.call_args.args[0])

    async def test_add_user_builds_upsert_and_commits(self) -> None:
        await requests.add_user(12, "reader")
        statement = self.session.execute.call_args.args[0]
        compiled = statement.compile(dialect=postgresql.dialect())
        self.assertIn("ON CONFLICT (user_id) DO UPDATE", str(compiled))
        self.assertEqual(compiled.params["user_id"], 12)
        self.assertEqual(compiled.params["username"], "reader")
        self.session.commit.assert_awaited_once_with()

    async def test_add_user_timeout(self) -> None:
        self.session.execute.side_effect = TimeoutError
        await requests.add_user(12, "reader")
        self.session.commit.assert_not_awaited()
        self.assertIn("Таймаут", self.print_mock.call_args.args[0])

    async def test_add_user_commit_error(self) -> None:
        self.session.commit.side_effect = SQLAlchemyError("commit failed")
        await requests.add_user(12, "reader")
        self.assertIn("Ошибка добавления пользователя", self.print_mock.call_args.args[0])

    async def test_get_descriptions_returns_all_scalars(self) -> None:
        records = [requests.ImageDescription(id=1, name="one.png", description="one")]
        self.session.execute.return_value.scalars.return_value.all.return_value = records
        self.assertEqual(await requests.get_all_image_descriptions(), records)

    async def test_get_descriptions_timeout(self) -> None:
        self.session.execute.side_effect = TimeoutError
        self.assertEqual(await requests.get_all_image_descriptions(), [])
        self.assertIn("Таймаут", self.print_mock.call_args.args[0])

    async def test_get_descriptions_error(self) -> None:
        self.session.execute.side_effect = SQLAlchemyError("failed")
        self.assertEqual(await requests.get_all_image_descriptions(), [])
        self.assertIn("Ошибка получения всех описаний", self.print_mock.call_args.args[0])

    async def test_processed_ids_are_deduplicated(self) -> None:
        self.session.execute.return_value = [(1,), (2,), (1,)]
        self.assertEqual(await requests.get_processed_image_ids(), {1, 2})

    async def test_processed_ids_empty(self) -> None:
        self.session.execute.return_value = []
        self.assertEqual(await requests.get_processed_image_ids(), set())

    async def test_processed_ids_timeout(self) -> None:
        self.session.execute.side_effect = TimeoutError
        self.assertEqual(await requests.get_processed_image_ids(), set())
        self.assertIn("Таймаут", self.print_mock.call_args.args[0])

    async def test_processed_ids_error(self) -> None:
        self.session.execute.side_effect = SQLAlchemyError("failed")
        self.assertEqual(await requests.get_processed_image_ids(), set())
        self.assertIn("Ошибка получения обработанных ID", self.print_mock.call_args.args[0])

    async def test_add_processed_description_preserves_fields(self) -> None:
        description = SimpleNamespace(id=3, name="three.png", description="three")
        await requests.add_processed_image_description(description)
        saved = self.session.add.call_args.args[0]
        self.assertIsInstance(saved, requests.ProcessedImageDescriptions)
        self.assertEqual((saved.id, saved.name, saved.description), (3, "three.png", "three"))
        self.session.commit.assert_awaited_once_with()

    async def test_add_processed_timeout(self) -> None:
        self.session.commit.side_effect = TimeoutError
        await requests.add_processed_image_description(
            SimpleNamespace(id=3, name="three.png", description="three")
        )
        self.assertIn("Таймаут", self.print_mock.call_args.args[0])

    async def test_add_processed_error(self) -> None:
        self.session.commit.side_effect = SQLAlchemyError("failed")
        await requests.add_processed_image_description(
            SimpleNamespace(id=3, name="three.png", description="three")
        )
        self.assertIn("Ошибка добавления в обработанные", self.print_mock.call_args.args[0])

    async def test_create_description_commits_new_record(self) -> None:
        self.assertTrue(await requests.create_image_description("one.png", "one"))
        saved = self.session.add.call_args.args[0]
        self.assertIsInstance(saved, requests.ImageDescription)
        self.assertIsNone(saved.id)
        self.assertEqual((saved.name, saved.description), ("one.png", "one"))
        self.session.commit.assert_awaited_once_with()

    async def test_create_description_recovers_after_timeout(self) -> None:
        self.session.commit.side_effect = [TimeoutError(), None]
        with patch.object(requests.asyncio, "sleep", new_callable=AsyncMock) as sleep:
            self.assertTrue(await requests.create_image_description("one.png", "one"))
        self.assertEqual(self.session.commit.await_count, 2)
        sleep.assert_awaited_once_with(0.5)

    async def test_create_description_exhausts_database_retries(self) -> None:
        self.session.commit.side_effect = SQLAlchemyError("failed")
        with patch.object(requests.asyncio, "sleep", new_callable=AsyncMock) as sleep:
            self.assertFalse(await requests.create_image_description("one.png", "one"))
        self.assertEqual(self.session.commit.await_count, 3)
        self.assertEqual([call.args[0] for call in sleep.await_args_list], [0.5, 1.0])
        self.assertIn("после 3 попыток", self.print_mock.call_args.args[0])

    async def test_create_description_exhausts_timeout_retries(self) -> None:
        self.session.commit.side_effect = TimeoutError
        with patch.object(requests.asyncio, "sleep", new_callable=AsyncMock) as sleep:
            self.assertFalse(
                await requests.create_image_description("one.png", "one", max_retries=1)
            )
        sleep.assert_not_awaited()

    async def test_create_description_zero_retries_does_not_write(self) -> None:
        self.assertFalse(await requests.create_image_description("one.png", "one", max_retries=0))
        self.session.add.assert_not_called()
        self.session.commit.assert_not_awaited()
