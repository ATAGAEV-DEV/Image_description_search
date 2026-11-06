"""Opt-in PostgreSQL tests; each test owns and removes one random schema.

Read TEST_DATABASE_URL only from .env.test in the repository root. Never fall
back to DATABASE_URL or load .env. Run this directory separately from unit tests.
"""

import unittest
from unittest.mock import patch
from uuid import uuid4

from sqlalchemy import func, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from sqlalchemy.schema import CreateSchema, DropSchema

from tests.support import import_application, load_test_database_url

TEST_DATABASE_URL = load_test_database_url() or ""

models = import_application("app.data.models")
requests = import_application("app.data.request")


@unittest.skipUnless(TEST_DATABASE_URL, "Set TEST_DATABASE_URL in .env.test to opt in")
class PostgreSQLIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Use real application sessions and queries against an isolated test schema."""

    async def asyncSetUp(self) -> None:
        database_url = TEST_DATABASE_URL
        if make_url(database_url).drivername != "postgresql+asyncpg":
            self.fail("TEST_DATABASE_URL must use postgresql+asyncpg")
        self.schema = "test_image_search_" + uuid4().hex
        self.schema_created = False
        self.database_engine = None
        self.admin_engine = create_async_engine(
            database_url,
            poolclass=NullPool,
            connect_args={"timeout": 10, "command_timeout": 15},
        )
        self.addAsyncCleanup(self.remove_schema)
        async with self.admin_engine.begin() as connection:
            await connection.execute(CreateSchema(self.schema))
        self.schema_created = True
        self.database_engine = create_async_engine(
            database_url,
            poolclass=NullPool,
            connect_args={
                "timeout": 10,
                "command_timeout": 15,
                "server_settings": {"search_path": self.schema},
            },
        )
        self.enterContext(patch.object(models, "engine", self.database_engine))
        self.enterContext(
            patch.object(models, "async_session_maker", async_sessionmaker(self.database_engine))
        )
        # request.py imports this symbol directly; patch its own binding as well.
        self.enterContext(patch.object(requests, "async_session", models.async_session))
        await models.init_models()

    async def remove_schema(self) -> None:
        """Clean even after failed setup or assertions, then verify exact target removal."""
        try:
            if self.database_engine is not None:
                await self.database_engine.dispose()
            if self.schema_created:
                async with self.admin_engine.begin() as connection:
                    await connection.execute(DropSchema(self.schema, cascade=True, if_exists=True))
                async with self.admin_engine.connect() as connection:
                    remaining = await connection.scalar(
                        text("SELECT count(*) FROM pg_namespace WHERE nspname = :schema"),
                        {"schema": self.schema},
                    )
                self.assertEqual(remaining, 0, "Temporary schema cleanup failed")
        finally:
            await self.admin_engine.dispose()

    async def test_init_models_is_idempotent_and_uses_test_schema(self) -> None:
        await models.init_models()
        async with models.async_session() as session:
            self.assertEqual(await session.scalar(text("SELECT current_schema()")), self.schema)
            tables = await session.execute(
                text(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = :schema"
                ),
                {"schema": self.schema},
            )
            self.assertEqual(
                set(tables.scalars()),
                {"users", "image_descriptions", "processed_image_descriptions"},
            )

    async def test_user_upsert_changes_name_without_duplicating_user(self) -> None:
        self.assertIsNone(await requests.get_user_by_id(12345))
        await requests.add_user(12345, "reader")
        user = await requests.get_user_by_id(12345)
        self.assertIsNotNone(user)
        self.assertEqual(user.username, "reader")
        await requests.add_user(12345, "новое_имя")
        updated = await requests.get_user_by_id(12345)
        self.assertIsNotNone(updated)
        self.assertEqual(updated.username, "новое_имя")
        async with models.async_session() as session:
            self.assertEqual(await session.scalar(select(func.count()).select_from(models.Users)), 1)

    async def test_empty_table_generates_ids_and_creation_timestamps(self) -> None:
        self.assertEqual(await requests.get_all_image_descriptions(), [])
        self.assertTrue(await requests.create_image_description("cat.png", "Кот на диване"))
        self.assertTrue(await requests.create_image_description("dog.jpg", "Собака на улице"))
        records = sorted(await requests.get_all_image_descriptions(), key=lambda row: row.id)
        self.assertEqual([row.id for row in records], [1, 2])
        self.assertEqual([row.name for row in records], ["cat.png", "dog.jpg"])
        self.assertTrue(all(row.created_at is not None for row in records))
        self.assertTrue(all(row.created_at.tzinfo is not None for row in records))

    async def test_processed_description_keeps_image_id_and_fields(self) -> None:
        self.assertEqual(await requests.get_processed_image_ids(), set())
        self.assertTrue(await requests.create_image_description("cat.png", "Кот"))
        image = (await requests.get_all_image_descriptions())[0]
        await requests.add_processed_image_description(image)
        self.assertEqual(await requests.get_processed_image_ids(), {image.id})
        async with models.async_session() as session:
            processed = await session.get(models.ProcessedImageDescriptions, image.id)
            self.assertIsNotNone(processed)
            self.assertEqual((processed.name, processed.description), ("cat.png", "Кот"))

    async def test_session_rolls_back_when_context_body_fails(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "abort transaction"):
            async with models.async_session() as session:
                session.add(models.Users(user_id=345, username="uncommitted"))
                await session.flush()
                raise RuntimeError("abort transaction")
        self.assertIsNone(await requests.get_user_by_id(345))

    async def test_duplicate_primary_key_does_not_damage_committed_data(self) -> None:
        await requests.add_user(345, "committed")
        with self.assertRaises(IntegrityError):
            async with models.async_session() as session:
                session.add(models.Users(user_id=345, username="duplicate"))
                await session.commit()
        user = await requests.get_user_by_id(345)
        self.assertIsNotNone(user)
        self.assertEqual(user.username, "committed")
