"""Cache the shared demonstration for Streamlit's native video player."""
import hashlib
import json

import streamlit as st


def credential_scope(config):
    """Keep cached media separate when the configured Drive account changes."""
    keys = ('GDRIVE_CLIENT_ID', 'GDRIVE_CLIENT_SECRET', 'GDRIVE_REFRESH_TOKEN')
    return hashlib.sha256(json.dumps([config[k] for k in keys]).encode()).hexdigest()


@st.cache_data(show_spinner=False, ttl=3600, max_entries=2)
def load_demo(file_id, scope, _store):
    # Only immutable demo bytes are shared, never a participant's Drive client or
    # answers. Exceptions are not cached, so a later visit can retry a failed load.
    return _store.demo_bytes(file_id)
