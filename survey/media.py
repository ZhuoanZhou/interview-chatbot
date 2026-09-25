"""Cache the shared demonstration and serve it through Streamlit's media endpoint."""
import hashlib
import json

import streamlit as st
from streamlit import runtime


def credential_scope(config):
    """Keep cached media separate when the configured Drive account changes."""
    keys = ('GDRIVE_CLIENT_ID', 'GDRIVE_CLIENT_SECRET', 'GDRIVE_REFRESH_TOKEN')
    return hashlib.sha256(json.dumps([config[k] for k in keys]).encode()).hexdigest()


@st.cache_data(show_spinner=False, ttl=3600, max_entries=2)
def load_demo(file_id, scope, _store):
    # Only immutable demo bytes are shared, never a participant's Drive client or
    # answers. Exceptions are not cached, so a later visit can retry a failed load.
    return _store.demo_bytes(file_id)


def demo_url(data):
    # This is the same media manager used by st.video in pinned Streamlit 1.64.
    # Register on EVERY rerun: caching the URL would lose the session reference
    # and allow Streamlit to remove the media while a participant is watching.
    # The browser adds the external app prefix. Python's baseUrlPath does not
    # include reverse-proxy prefixes used by hosted deployments.
    return runtime.get_instance().media_file_mgr.add(
        data, 'video/mp4', 'survey.demo')
