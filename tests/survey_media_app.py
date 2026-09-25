"""Local media integration harness. Only reads the generated test clip."""
import json
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from survey.media import demo_url, load_demo


class FixtureStore:
    def demo_bytes(self, file_id):
        return (ROOT / 'tmp/survey-demo-fixture.mp4').read_bytes()


st.set_page_config(layout='wide')
st.button('Rerun host')
component = components.declare_component('survey_media_test', path=str(ROOT / 'survey/frontend'))
schema = json.loads((ROOT / 'survey/schema.json').read_text(encoding='utf-8'))
url = demo_url(load_demo('fixture', 'fake-account', FixtureStore()))
component(schema=schema, record={'revision': 0, 'state': {
    'page': 'demo_video', 'status': 'active', 'answers': {}}},
    session_key='media-test', demo_url=url, key='media-test', default=None)
