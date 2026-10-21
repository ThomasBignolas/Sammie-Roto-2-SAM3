import contextlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sammie.backends.compat import configure_multiplex_output
from sammie.backends.worker import Sam3Engine


class MultiplexCompatibilityTests(unittest.TestCase):
    def test_first_tracker_visit_keeps_mask_and_existing_other_objects(self):
        # Reproduce the pinned upstream early return on frames absent from the cache.
        def original(state, frame, refined=None):
            if frame not in state['cached_frame_outputs']:
                return {}
            return dict(state['cached_frame_outputs'][frame], **(refined or {}))
        model = SimpleNamespace(_build_sam2_output=original)
        state = {'cached_frame_outputs': {0: {'a': 'old A', 'b': 'old B'}}}
        configure_multiplex_output(model)
        self.assertEqual(model._build_sam2_output(state, 1, {'a': 'new A'}), {'a': 'new A'})
        self.assertEqual(model._build_sam2_output(state, 0, {'a': 'new A'}),
                         {'a': 'new A', 'b': 'old B'})
        self.assertEqual(model._build_sam2_output(state, 2), {})
        self.assertNotIn(2, state['cached_frame_outputs'])

    def test_forced_pass_does_not_mark_detector_objects_as_user_refined(self):
        existing = {'type': 'refine', 'obj_ids': [9], 'frame_idx': 0}
        state = {'action_history': [existing]}
        def add_action_history(state, action_type, obj_ids, frame_idx):
            state['action_history'].append({'type': action_type, 'obj_ids': obj_ids, 'frame_idx': frame_idx})
        def stream(request):
            self.assertEqual(state['action_history'][-1]['obj_ids'], [1, 2])
            state['action_history'].append({'type': 'propagation_partial', 'obj_ids': [1, 2]})
            yield {'frame': 0}
            yield {'frame': 1}
        engine = Sam3Engine()
        engine.version, engine.session = 'SAM 3.1', 'test'
        engine.sessions = {0: 'test'}
        engine.predictor = SimpleNamespace(
            model=SimpleNamespace(add_action_history=add_action_history),
            _all_inference_states={'test': {'state': state}}, handle_stream_request=stream)
        fake_torch = SimpleNamespace(inference_mode=contextlib.nullcontext,
                                    autocast=lambda *a, **k: contextlib.nullcontext(), bfloat16=None)
        with patch.dict('sys.modules', {'torch': fake_torch}), \
                patch.object(engine, 'output', side_effect=lambda x, _: {'frame': x['frame'], 'masks': {}}), \
                patch.object(engine, 'stats', return_value={}):
            iterator = engine.propagate({'selected_object_ids': [1, 2], 'start_frame_index': 0}, '.')
            self.assertEqual(next(iterator), {'frame': 0, 'masks': {}, 'gpu': {}})
            iterator.close()  # Includes cancellation, not just normal completion.
        self.assertEqual(state['action_history'], [existing, {'type': 'propagation_partial', 'obj_ids': [1, 2]}])


if __name__ == '__main__':
    unittest.main()
