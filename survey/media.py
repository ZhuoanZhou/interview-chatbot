"""Cache the shared demonstration and serve it through Streamlit's media endpoint."""
import hashlib
import json
import threading
import time

from streamlit import runtime

# Same limits as the previous st.cache_data cache: one hour, two videos.
TTL_SECONDS = 3600
MAX_ENTRIES = 2
_lock = threading.Lock()
_entries = {}  # (file_id, scope) -> _Entry


class _Entry:
    def __init__(self):
        self.done = threading.Event()
        self.data = None
        self.error = None
        self.finished_at = None


def credential_scope(config):
    """Keep cached media separate when the configured Drive account changes."""
    keys = ('GDRIVE_CLIENT_ID', 'GDRIVE_CLIENT_SECRET', 'GDRIVE_REFRESH_TOKEN')
    return hashlib.sha256(json.dumps([config[k] for k in keys]).encode()).hexdigest()


def _claim(key):
    """Return (entry, is_new). Caller must hold _lock."""
    entry = _entries.get(key)
    if entry and entry.done.is_set() and entry.data is not None \
            and time.monotonic() - entry.finished_at > TTL_SECONDS:
        del _entries[key]
        entry = None
    if entry:
        return entry, False
    entry = _entries[key] = _Entry()
    while len(_entries) > MAX_ENTRIES:
        del _entries[next(iter(_entries))]  # oldest first
    return entry, True


def _download(key, entry, make_store):
    try:
        entry.data = make_store().demo_bytes(key[0])
    except Exception as error:  # Failures are not cached; a later visit retries.
        entry.error = error
        with _lock:
            if _entries.get(key) is entry:
                del _entries[key]
    finally:
        entry.finished_at = time.monotonic()
        entry.done.set()


def prefetch_demo(file_id, scope, make_store):
    """Start a background download unless the video is cached or already loading.

    make_store must create a separate Drive client: Google API clients are not
    safe to share with the participant's own session thread.
    """
    with _lock:
        entry, new = _claim((file_id, scope))
    if new:
        threading.Thread(target=_download, args=((file_id, scope), entry, make_store),
                         name='survey-demo-prefetch', daemon=True).start()


def load_demo(file_id, scope, store):
    """Return the demonstration bytes, waiting for a background download if one
    is running. Only immutable demo bytes are shared between participants."""
    with _lock:
        entry, new = _claim((file_id, scope))
    if new:
        _download((file_id, scope), entry, lambda: store)
    entry.done.wait()
    if entry.error is not None:
        raise entry.error
    return entry.data


def clear_demo_cache():
    with _lock:
        _entries.clear()


def demo_url(data):
    # This is the same media manager used by st.video in pinned Streamlit 1.64.
    # Register on EVERY rerun: caching the URL would lose the session reference
    # and allow Streamlit to remove the media while a participant is watching.
    # The browser adds the external app prefix. Python's baseUrlPath does not
    # include reverse-proxy prefixes used by hosted deployments.
    return runtime.get_instance().media_file_mgr.add(
        data, 'video/mp4', 'survey.demo')
