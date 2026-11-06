"""Import application modules without reading local secrets or using live services."""

import importlib
import os
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from dotenv import dotenv_values

TEST_ENV_FILE = Path(__file__).resolve().parents[1] / ".env.test"

TEST_ENVIRONMENT = {
    "DATABASE_URL": "postgresql+asyncpg://test:test@127.0.0.1:1/unit_tests",
    "DATABASE_SCHEMA": "public",
    "TG_TOKEN": "123456789:unit-test-token",
    "AI_TOKEN": "unit-test-token",
    "AI_PROVIDER_URL": "https://provider.example.invalid/v1",
}


def load_test_database_url() -> str | None:
    """Read only TEST_DATABASE_URL from the repository's .env.test file."""
    return dotenv_values(TEST_ENV_FILE, interpolate=False).get("TEST_DATABASE_URL") or None


def import_application(module_name: str) -> ModuleType:
    """Load real application code with dummy configuration and no .env access."""
    with patch.dict(os.environ, TEST_ENVIRONMENT), patch("dotenv.load_dotenv", return_value=False):
        return importlib.import_module(module_name)
