"""Run the real entry point with an in-memory store and a generated video."""
import json
import hashlib
import os
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import survey.storage as storage


class FixtureStore:
    def __init__(self, config):
        pass

    def demo_bytes(self, file_id):
        return (ROOT / 'tmp/survey-demo-fixture.mp4').read_bytes()

    def save(self, token, packet):
        return storage.make_record(st.session_state.survey_record, packet)


if 'survey_record' not in st.session_state:
    schema = json.loads((ROOT / 'survey/schema.json').read_text(encoding='utf-8'))
    record = storage.new_record(schema['version'], 'P-ABC123')
    record['state'].update(page='demo_video', answers={
        'demo_consent': {'status': 'answered', 'choices': ['Yes']}})
    st.session_state.update(survey_record=record, survey_token='P-ABC123')

fake_secrets = {key: 'fixture' for key in (
    'GDRIVE_FOLDER_ID', 'GDRIVE_CLIENT_ID', 'GDRIVE_CLIENT_SECRET', 'GDRIVE_REFRESH_TOKEN')}
fake_secrets['DEMO_VIDEO_FILE_ID'] = hashlib.sha256((ROOT / 'tmp/survey-demo-fixture.mp4').read_bytes()).hexdigest()
st.sidebar.button('Rerun host')
with patch.dict(os.environ, {'SURVEY_PREVIEW': '0'}), \
        patch.object(storage, 'DriveStore', FixtureStore), \
        patch.object(st, 'secrets', fake_secrets):
    runpy.run_path(str(ROOT / 'streamlit_app.py'), run_name='__main__')
