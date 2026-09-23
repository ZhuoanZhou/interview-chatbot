"""Check both a fresh launch and an update with an old imported storage module."""
import importlib
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
import survey.storage as storage

APP = Path(__file__).resolve().parents[1] / 'streamlit_app.py'


class StartupTests(unittest.TestCase):
    def run_app(self):
        with patch.dict(os.environ, {'SURVEY_PREVIEW':'1'}):
            app = AppTest.from_file(str(APP)).run(timeout=15)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any(b.label == 'Start a new survey' for b in app.button))

    def test_fresh_startup(self):
        self.run_app()

    def test_update_recovers_from_previous_storage_api(self):
        old_class = storage.LocalStore
        del storage.normalize_access
        del storage.new_participant_id
        try:
            self.run_app()
            self.assertTrue(callable(storage.normalize_access))
            self.assertTrue(callable(storage.new_participant_id))
            self.assertIsNot(storage.LocalStore, old_class)
        finally:
            importlib.reload(storage)


if __name__ == '__main__':
    unittest.main()
