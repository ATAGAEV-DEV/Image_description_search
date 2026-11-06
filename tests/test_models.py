"""Database configuration and session lifecycle tests without PostgreSQL."""

import os
import runpy
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy.ext.asyncio import AsyncSession

from tests.support import TEST_ENVIRONMENT, import_application

models = import_application("app.data.models")
MODEL_PATH = Path(__file__).resolve().parents[1] / "app" / "data" / "models.py"


class ModelConfigurationTests(unittest.TestCase):
    """Check environment defaults without loading developer credentials."""

    def load_configuration(self, schema: str | None = None) -> tuple[dict, MagicMock]:
        environment = {**TEST_ENVIRONMENT}
        if schema is None:
            environment.pop("DATABASE_SCHEMA")
        else:
            environment["DATABASE_SCHEMA"] = schema
        with (
            patch.dict(os.environ, environment),
            patch("dotenv.load_dotenv", return_value=False),
            patch("sqlalchemy.ext.asyncio.create_async_engine") as create_engine,
            patch("sqlalchemy.ext.asyncio.async_sessionmaker"),
        ):
            if schema is None:
                os.environ.pop("DATABASE_SCHEMA", None)
            configuration = runpy.run_path(str(MODEL_PATH))
        return configuration, create_engine

    def test_missing_schema_defaults_to_public(self) -> None:
        configuration, create_engine = self.load_configuration()
        self.assertEqual(configuration["DATABASE_SCHEMA"], "public")
        self.assertEqual(
            create_engine.call_args.kwargs["connect_args"],
            {"server_settings": {"search_path": "public"}},
        )

    def test_empty_schema_defaults_to_public(self) -> None:
        configuration, create_engine = self.load_configuration("")
        self.assertEqual(configuration["DATABASE_SCHEMA"], "public")
        self.assertEqual(
            create_engine.call_args.kwargs["connect_args"]["server_settings"]["search_path"],
            "public",
        )

    def test_custom_schema_and_database_url_are_passed_to_engine(self) -> None:
        configuration, create_engine = self.load_configuration("test_images")
        self.assertEqual(configuration["DATABASE_SCHEMA"], "test_images")
        create_engine.assert_called_once_with(
            TEST_ENVIRONMENT["DATABASE_URL"],
            connect_args={"server_settings": {"search_path": "test_images"}},
            pool_pre_ping=True,
            pool_recycle=1800,
        )

    def test_database_url_is_required(self) -> None:
        with (
            patch.dict(os.environ, {"DATABASE_URL": ""}),
            patch("dotenv.load_dotenv", return_value=False),
            patch("sqlalchemy.ext.asyncio.create_async_engine") as create_engine,
            self.assertRaisesRegex(RuntimeError, "DATABASE_URL не задан"),
        ):
            runpy.run_path(str(MODEL_PATH))
        create_engine.assert_not_called()

    def test_table_definitions_preserve_constraints(self) -> None:
        self.assertEqual(
            set(models.Base.metadata.tables),
            {"users", "image_descriptions", "processed_image_descriptions"},
        )
        self.assertTrue(models.ImageDescription.__table__.c.id.primary_key)
        self.assertTrue(models.ImageDescription.__table__.c.id.autoincrement)
        self.assertFalse(models.ImageDescription.__table__.c.name.nullable)
        self.assertFalse(models.ImageDescription.__table__.c.description.nullable)
        self.assertFalse(models.Users.__table__.c.user_id.autoincrement)
        self.assertTrue(models.ImageDescription.__table__.c.created_at.type.timezone)


class SessionLifecycleTests(unittest.IsolatedAsyncioTestCase):
    """Ensure normal sessions and initialization use the single configured engine."""

    async def test_session_is_a_real_async_session_bound_to_engine(self) -> None:
        async with models.async_session() as session:
            self.assertIsInstance(session, AsyncSession)
            self.assertIs(session.bind, models.engine)
            self.assertFalse(session.in_transaction())

    async def test_context_closes_session_when_body_raises(self) -> None:
        session = MagicMock(spec=AsyncSession)
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        with (
            patch.object(models, "async_session_maker", return_value=session),
            self.assertRaisesRegex(RuntimeError, "operation failed"),
        ):
            async with models.async_session():
                raise RuntimeError("operation failed")
        session.__aenter__.assert_awaited_once()
        session.__aexit__.assert_awaited_once()
        self.assertIs(session.__aexit__.call_args.args[0], RuntimeError)

    async def test_init_models_creates_tables_on_single_engine(self) -> None:
        connection = MagicMock()
        connection.run_sync = AsyncMock()
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=connection)
        context.__aexit__ = AsyncMock(return_value=False)
        with patch.object(models, "engine") as engine:
            engine.begin.return_value = context
            await models.init_models()
        engine.begin.assert_called_once_with()
        connection.run_sync.assert_awaited_once_with(models.Base.metadata.create_all)
        context.__aexit__.assert_awaited_once()

    async def test_initialization_failure_propagates(self) -> None:
        context = MagicMock()
        context.__aenter__ = AsyncMock(side_effect=RuntimeError("database unavailable"))
        context.__aexit__ = AsyncMock(return_value=False)
        with patch.object(models, "engine") as engine:
            engine.begin.return_value = context
            with self.assertRaisesRegex(RuntimeError, "database unavailable"):
                await models.init_models()
