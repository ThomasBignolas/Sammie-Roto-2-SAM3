"""Exercise the actual subprocess protocol without model weights or a CUDA device."""
import os
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
from sammie.backends.bridge import WorkerClient


class BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_worker_errors_do_not_desynchronize_protocol_and_process_closes(self):
        client = WorkerClient(python=sys.executable)
        try:
            for command in ('unknown-one', 'unknown-two'):
                with self.assertRaisesRegex(RuntimeError, command):
                    client.call(command)
                self.assertFalse(client.busy)
        finally:
            client.close()
        self.assertIsNotNone(client.process.poll())
        self.assertTrue(all(not thread.is_alive() for thread in client.readers))

    def test_stream_cancel_acknowledges_and_releases_busy_flag_even_on_error(self):
        client = WorkerClient(python=sys.executable)
        try:
            sent = []
            with patch.object(client, '_send', side_effect=sent.append), patch.object(
                    client, '_receive', side_effect=[{'result': {'frame': 0}}, RuntimeError('cancel failed')]):
                stream = client.stream('propagate')
                self.assertEqual(next(stream), {'frame': 0})
                with self.assertRaisesRegex(RuntimeError, 'cancel failed'):
                    stream.close()
                self.assertFalse(client.busy)
                self.assertEqual(sent[-1], {'continue': False})
        finally:
            client.close()


if __name__ == '__main__':
    unittest.main()
