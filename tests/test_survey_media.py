"""Demo delivery tests use fabricated bytes; no credentials or network access."""
import threading
import unittest
from unittest.mock import MagicMock, patch

import survey.media as media
from survey.media import clear_demo_cache, credential_scope, demo_url, load_demo, prefetch_demo
from survey.storage import DriveStore


class DemoTests(unittest.TestCase):
    def setUp(self):
        clear_demo_cache()
        self.addCleanup(clear_demo_cache)

    def test_demo_download_is_shared_between_participant_stores(self):
        first = MagicMock()
        first.demo_bytes.return_value = b'shared demonstration'
        second = MagicMock()
        self.assertEqual(load_demo('demo', 'account', first), b'shared demonstration')
        self.assertEqual(load_demo('demo', 'account', second), b'shared demonstration')
        first.demo_bytes.assert_called_once_with('demo')
        second.demo_bytes.assert_not_called()

    def test_changed_video_or_credentials_do_not_reuse_old_bytes(self):
        store = MagicMock()
        store.demo_bytes.side_effect = [b'one', b'two', b'three']
        self.assertEqual(load_demo('one', 'account', store), b'one')
        self.assertEqual(load_demo('two', 'account', store), b'two')
        self.assertEqual(load_demo('two', 'changed-account', store), b'three')

    def test_failed_download_is_retried(self):
        store = MagicMock()
        store.demo_bytes.side_effect = [OSError('unavailable'), b'recovered']
        with self.assertRaises(OSError):
            load_demo('demo', 'account', store)
        self.assertEqual(load_demo('demo', 'account', store), b'recovered')

    def test_background_prefetch_is_reused_on_the_demo_screen(self):
        release = threading.Event()
        background = MagicMock()
        background.demo_bytes.side_effect = lambda file_id: release.wait(5) and b'prefetched'
        prefetch_demo('demo', 'account', lambda: background)
        prefetch_demo('demo', 'account', lambda: background)  # already loading: no second download
        participant = MagicMock()
        result = []
        waiter = threading.Thread(target=lambda: result.append(load_demo('demo', 'account', participant)))
        waiter.start()
        release.set()
        waiter.join(5)
        self.assertEqual(result, [b'prefetched'])
        background.demo_bytes.assert_called_once_with('demo')
        participant.demo_bytes.assert_not_called()

    def test_failed_prefetch_is_retried_on_the_demo_screen(self):
        background = MagicMock()
        background.demo_bytes.side_effect = OSError('unavailable')
        prefetch_demo('demo', 'account', lambda: background)
        for thread in threading.enumerate():
            if thread.name == 'survey-demo-prefetch':
                thread.join(5)
        store = MagicMock()
        store.demo_bytes.return_value = b'recovered'
        self.assertEqual(load_demo('demo', 'account', store), b'recovered')

    def test_cached_demo_expires_after_an_hour(self):
        store = MagicMock()
        store.demo_bytes.side_effect = [b'old', b'new']
        with patch('survey.media.time.monotonic', return_value=1000.0):
            self.assertEqual(load_demo('demo', 'account', store), b'old')
        with patch('survey.media.time.monotonic', return_value=1000.0 + media.TTL_SECONDS - 1):
            self.assertEqual(load_demo('demo', 'account', store), b'old')
        with patch('survey.media.time.monotonic', return_value=1000.0 + media.TTL_SECONDS + 1):
            self.assertEqual(load_demo('demo', 'account', store), b'new')

    def test_scope_changes_with_drive_credentials(self):
        config = dict(GDRIVE_CLIENT_ID='fake-id', GDRIVE_CLIENT_SECRET='fake-secret',
                      GDRIVE_REFRESH_TOKEN='fake-token')
        before = credential_scope(config)
        for key in config:
            self.assertNotEqual(before, credential_scope({**config, key: 'changed'}))
        self.assertNotIn('fake-secret', before)

    def test_media_is_registered_on_each_rerun_without_guessing_external_prefix(self):
        with patch('survey.media.runtime.get_instance') as runtime:
            manager = runtime.return_value.media_file_mgr
            manager.add.return_value = '/media/fixture.mp4'
            self.assertEqual(demo_url(b'video'), '/media/fixture.mp4')
            self.assertEqual(demo_url(b'video'), '/media/fixture.mp4')
            self.assertEqual(manager.add.call_count, 2)
            manager.add.assert_called_with(b'video', 'video/mp4', 'survey.demo')

    def test_chunked_download_assembles_all_bytes_and_retries_transient_errors(self):
        store = object.__new__(DriveStore)
        store.service = MagicMock()
        with patch('googleapiclient.http.MediaIoBaseDownload') as download:
            def make_downloader(buffer, request, chunksize):
                self.assertEqual(chunksize, 8 * 1024 * 1024)
                chunks = iter([(b'first', False), (b'second', True)])
                def next_chunk(num_retries):
                    self.assertEqual(num_retries, 2)
                    data, done = next(chunks)
                    buffer.write(data)
                    return None, done
                return MagicMock(next_chunk=next_chunk)
            download.side_effect = make_downloader
            self.assertEqual(store.demo_bytes('demo-id'), b'firstsecond')
            with self.assertRaises(ValueError):
                store.demo_bytes('../invalid')


if __name__ == '__main__':
    unittest.main()
