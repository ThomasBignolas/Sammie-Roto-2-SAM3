"""Session isolation, frame alignment and cancellation without a GPU."""
import contextlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from sammie.backends.worker import Sam3Engine


class ConceptEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = Sam3Engine()
        self.engine.resource = 'frames'
        self.engine.sessions = {0: 'person', 1: 'camera'}
        self.calls = []
        def request(r):
            self.calls.append(r)
            return {'is_success': True}
        self.engine.predictor = SimpleNamespace(handle_request=request)

    def test_request_routes_virtual_ids_and_close_preserves_other_session(self):
        fake_torch = SimpleNamespace(inference_mode=contextlib.nullcontext,
                                    autocast=lambda *a, **k: contextlib.nullcontext(), bfloat16=None)
        with patch.dict('sys.modules', {'torch': fake_torch}):
            self.engine.request({'type': 'add_prompt', 'obj_id': 1000005, 'points': [[.5, .5]]})
            self.assertEqual(self.calls[-1]['session_id'], 'camera')
            self.assertEqual(self.calls[-1]['obj_id'], 5)
            self.engine.request({'type': 'close_group', 'group_id': 1})
            self.assertEqual(self.engine.sessions, {0: 'person'})
            self.assertEqual(self.calls[-1], {'type': 'close_session', 'session_id': 'camera'})
            self.engine.request({'type': 'reset_session'})
            self.assertFalse(self.engine.sessions)

    def test_merged_stream_has_one_frame_and_closes_all_on_cancel(self):
        closed = []
        def stream(request, output, group):
            self.assertEqual(request['selected_object_ids'], [5])
            try:
                for frame in range(3):
                    yield {'frame': frame, 'masks': {str(group*1000000+5): f'{group}/{frame}.png'}}
            finally:
                closed.append(group)
        with patch.object(self.engine, '_propagate_group', side_effect=stream), \
                patch.object(self.engine, 'stats', return_value={}):
            iterator = self.engine.propagate({'selected_object_ids': [5, 1000005]}, '.')
            result = next(iterator)
            self.assertEqual(result, {'frame': 0, 'masks': {'5': '0/0.png', '1000005': '1/0.png'}, 'gpu': {}})
            iterator.close()
            self.assertEqual(set(closed), {0, 1})

    def test_misaligned_or_missing_group_frame_fails(self):
        for mismatch in ('offset', 'missing'):
            closed = []
            def stream(request, output, group):
                try:
                    if group and mismatch == 'missing':
                        return
                    yield {'frame': group if mismatch == 'offset' else 0, 'masks': {}}
                finally:
                    closed.append(group)
            with patch.object(self.engine, '_propagate_group', side_effect=stream):
                with self.assertRaisesRegex(RuntimeError, 'different frame'):
                    list(self.engine.propagate({'selected_object_ids': [5, 1000005]}, '.'))
            self.assertEqual(set(closed), {0, 1})


if __name__ == '__main__': unittest.main()
