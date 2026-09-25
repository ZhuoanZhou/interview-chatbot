"""Check both a fresh launch and an update with an old imported storage module."""
import importlib
import json
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

    def test_demo_uses_native_video_and_component_receives_no_media_payload(self):
        class FakeDriveStore:
            def __init__(self, config):
                pass
        app = AppTest.from_file(str(APP))
        app.secrets.update({key: 'fixture' for key in (
            'GDRIVE_FOLDER_ID', 'GDRIVE_CLIENT_ID', 'GDRIVE_CLIENT_SECRET', 'GDRIVE_REFRESH_TOKEN')})
        record = storage.new_record('fixture', 'P-ABC123')
        record['state'].update(page='demo_video', answers={})
        app.session_state['survey_record'] = record
        app.session_state['survey_token'] = 'P-ABC123'
        with patch.dict(os.environ, {'SURVEY_PREVIEW':'0'}), \
                patch.object(storage, 'DriveStore', FakeDriveStore), \
                patch('survey.media.load_demo', return_value=b'generated fixture bytes'):
            app.run(timeout=15)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.get('video')), 1)
        args = json.loads(app.get('component_instance')[0].proto.json_args)
        self.assertTrue(args['demo_available'])
        self.assertNotIn('demo_url', args)
        self.assertNotIn('demo_data', args)

    def test_demo_preview_does_not_download_or_create_a_player(self):
        app = AppTest.from_file(str(APP))
        record = storage.new_record('fixture', 'P-ABC123')
        record['state']['page'] = 'demo_video'
        app.session_state['survey_record'] = record
        app.session_state['survey_token'] = 'P-ABC123'
        with patch.dict(os.environ, {'SURVEY_PREVIEW':'1'}), \
                patch('survey.media.load_demo') as download:
            app.run(timeout=15)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.get('video')), 0)
        download.assert_not_called()

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
