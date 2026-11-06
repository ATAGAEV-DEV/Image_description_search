"""Test integration configuration using dummy files, never developer secrets."""

import os
import runpy
import tempfile
import unittest
from contextlib import chdir
from pathlib import Path
from unittest.mock import patch

from tests import support


class IntegrationConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = self.enterContext(tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")))
        self.env_file = Path(directory) / ".env.test"
        self.enterContext(patch.object(support, "TEST_ENV_FILE", self.env_file, create=True))

    def test_reads_test_database_url_from_dotenv_test(self) -> None:
        url = "postgresql+asyncpg://test:test@127.0.0.1:1/integration_test"
        self.env_file.write_text(f'TEST_DATABASE_URL="{url}"\n', encoding="utf-8")
        self.assertEqual(support.load_test_database_url(), url)

    def test_postgres_suite_uses_the_dotenv_loader_instead_of_process_secrets(self) -> None:
        url = "postgresql+asyncpg://test:test@127.0.0.1:1/integration_test"
        path = Path(__file__).resolve().parent / "integration" / "test_postgres.py"
        with (
            patch.object(support, "load_test_database_url", return_value=url) as load,
            patch.dict(os.environ, {"TEST_DATABASE_URL": ""}),
        ):
            namespace = runpy.run_path(str(path))
        load.assert_called_once_with()
        self.assertEqual(namespace["TEST_DATABASE_URL"], url)
        self.assertFalse(
            getattr(namespace["PostgreSQLIntegrationTests"], "__unittest_skip__", False)
        )

    def test_missing_file_does_not_fall_back_to_process_or_working_dotenv(self) -> None:
        url = "postgresql+asyncpg://test:test@127.0.0.1:1/working_database"
        (self.env_file.parent / ".env").write_text(f"TEST_DATABASE_URL={url}\n", encoding="utf-8")
        with patch.dict(os.environ, {"TEST_DATABASE_URL": url, "DATABASE_URL": url}):
            self.assertIsNone(support.load_test_database_url())

    def test_database_url_key_is_not_used_as_a_fallback(self) -> None:
        self.env_file.write_text(
            "DATABASE_URL=postgresql+asyncpg://test:test@127.0.0.1:1/working_database\n",
            encoding="utf-8",
        )
        self.assertIsNone(support.load_test_database_url())

    def test_empty_or_valueless_test_url_is_missing(self) -> None:
        for content in ("", "TEST_DATABASE_URL=\n", "TEST_DATABASE_URL\n"):
            with self.subTest(content=content):
                self.env_file.write_text(content, encoding="utf-8")
                self.assertIsNone(support.load_test_database_url())

    def test_test_file_is_used_without_mutating_process_environment(self) -> None:
        url = "postgresql+asyncpg://test:test@127.0.0.1:1/integration_test"
        self.env_file.write_text(f"TEST_DATABASE_URL={url}\n", encoding="utf-8")
        with patch.dict(os.environ, {"TEST_DATABASE_URL": "inherited-value"}):
            self.assertEqual(support.load_test_database_url(), url)
            self.assertEqual(os.environ["TEST_DATABASE_URL"], "inherited-value")

    def test_working_directory_does_not_change_the_file_location(self) -> None:
        url = "postgresql+asyncpg://test:test@127.0.0.1:1/integration_test"
        self.env_file.write_text(f"TEST_DATABASE_URL={url}\n", encoding="utf-8")
        other_directory = self.env_file.parent / "other"
        other_directory.mkdir()
        with chdir(other_directory):
            self.assertEqual(support.load_test_database_url(), url)

    def test_process_variables_are_not_interpolated_into_the_url(self) -> None:
        self.env_file.write_text("TEST_DATABASE_URL=${DATABASE_URL}\n", encoding="utf-8")
        with patch.dict(os.environ, {"DATABASE_URL": "working-database-value"}):
            self.assertEqual(support.load_test_database_url(), "${DATABASE_URL}")

    def test_postgres_suite_skips_when_dotenv_test_has_no_url(self) -> None:
        path = Path(__file__).resolve().parent / "integration" / "test_postgres.py"
        with patch.object(support, "load_test_database_url", return_value=None):
            namespace = runpy.run_path(str(path))
        self.assertEqual(namespace["TEST_DATABASE_URL"], "")
        suite = namespace["PostgreSQLIntegrationTests"]
        self.assertTrue(suite.__unittest_skip__)
        self.assertIn(".env.test", suite.__unittest_skip_why__)
