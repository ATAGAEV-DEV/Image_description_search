"""Exercise startup orchestration without polling Telegram or touching a database."""

import os
import runpy
import unittest
from pathlib import Path
from types import CoroutineType
from unittest.mock import AsyncMock, patch

from tests.support import TEST_ENVIRONMENT, import_application

entrypoint = import_application("main")


class MainTests(unittest.IsolatedAsyncioTestCase):
    """Validate startup order and failure propagation."""

    async def test_initializes_tables_before_starting_polling(self) -> None:
        events = []

        async def initialize() -> None:
            events.append("init_models")

        async def poll(bot: object) -> None:
            events.append("poll")

        with (
            patch.object(entrypoint, "Bot") as bot_type,
            patch.object(entrypoint, "Dispatcher") as dispatcher_type,
            patch.object(entrypoint, "init_models", new=AsyncMock(side_effect=initialize)) as init,
        ):
            dispatcher = dispatcher_type.return_value
            dispatcher.include_router.side_effect = lambda router: events.append("include_router")
            dispatcher.start_polling = AsyncMock(side_effect=poll)
            await entrypoint.main()
        bot_type.assert_called_once_with(token=entrypoint.TG_TOKEN)
        dispatcher.include_router.assert_called_once_with(entrypoint.router)
        init.assert_awaited_once_with()
        dispatcher.start_polling.assert_awaited_once_with(bot_type.return_value)
        self.assertEqual(events, ["include_router", "init_models", "poll"])

    async def test_database_failure_prevents_polling(self) -> None:
        with (
            patch.object(entrypoint, "Bot"),
            patch.object(entrypoint, "Dispatcher") as dispatcher_type,
            patch.object(
                entrypoint,
                "init_models",
                new=AsyncMock(side_effect=RuntimeError("database unavailable")),
            ),
        ):
            dispatcher_type.return_value.start_polling = AsyncMock()
            with self.assertRaisesRegex(RuntimeError, "database unavailable"):
                await entrypoint.main()
        dispatcher_type.return_value.start_polling.assert_not_awaited()


class ScriptEntryPointTests(unittest.TestCase):
    """Cover the __main__ guard while deliberately not running the event loop."""

    def test_script_passes_main_coroutine_to_asyncio_run(self) -> None:
        coroutine_names = []

        def close_without_running(coroutine: CoroutineType) -> None:
            coroutine_names.append(coroutine.cr_code.co_name)
            coroutine.close()

        with (
            patch.dict(os.environ, TEST_ENVIRONMENT),
            patch("dotenv.load_dotenv", return_value=False),
            patch("asyncio.run", side_effect=close_without_running) as run,
        ):
            path = Path(__file__).resolve().parents[1] / "main.py"
            runpy.run_path(str(path), run_name="__main__")
        run.assert_called_once()
        self.assertEqual(coroutine_names, ["main"])
