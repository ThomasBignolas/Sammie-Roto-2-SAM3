"""Exercise real official networks with RANDOM weights when gated weights are unavailable.

This validates API plumbing and native kernels only. It does NOT test segmentation
quality, checkpoint loading, or provide a performance benchmark for trained SAM3.
Run with the isolated worker Python: -m scripts.sam3_api_smoke --version sam3
"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import uuid
from sammie.backends.worker import Sam3Engine, configure_runtime


def main():
    configure_runtime()
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', choices=['sam3', 'sam3.1'], default='sam3')
    args = parser.parse_args()
    import cv2
    import numpy as np
    import torch
    from sam3.model_builder import build_sam3_video_model, build_sam3_multiplex_video_predictor
    from sam3.model.sam3_base_predictor import Sam3BasePredictor
    from sammie.backends.compat import configure_attention
    configure_attention()
    root = Path('.test-output').resolve() / ('untrained-' + args.version + '-' + uuid.uuid4().hex[:8])
    frames = root/'frames'
    frames.mkdir(parents=True)
    for i in range(4):
        image = np.zeros((180, 320, 3), np.uint8)
        cv2.rectangle(image, (90+i*4, 30), (190+i*4, 150), (240, 240, 240), -1)
        cv2.imwrite(str(frames/f'{i:05d}.png'), image)
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    engine = Sam3Engine(print)
    engine.version = 'SAM 3' if args.version == 'sam3' else 'SAM 3.1'
    if args.version == 'sam3':
        model = build_sam3_video_model(load_from_HF=False, device='cuda').eval()
        predictor = Sam3BasePredictor()
        predictor.model = model
    else:
        # Explicit test-only bypass of checkpoint download, not used by the application.
        with patch('sam3.model_builder.download_ckpt_from_hf', return_value=None):
            predictor = build_sam3_multiplex_video_predictor(
                use_fa3=False, compile=False, warm_up=False, session_expiration_sec=0)
    engine.predictor = predictor
    state = engine.start(str(frames))
    assert state['frames'] == 4 and (state['width'], state['height']) == (320, 180)
    results = {'version': args.version, 'weights': 'RANDOM / UNTRAINED', 'session': state, 'checks': []}
    for oid, point in [(10000, [.4, .5]), (10001, [.6, .5])]:
        result = engine.request({'type': 'add_prompt', 'frame_index': 1,
                                 'obj_id': oid, 'points': [point], 'point_labels': [1]})
        assert result['frame_index'] == 1
        engine.output(result, str(root/'masks'))
    results['checks'].append('multi-object positive point requests')
    for direction, start, count in [('forward', 1, 2), ('backward', 1, 1)]:
        output = list(engine.propagate({'type': 'propagate_in_video', 'start_frame_index': start,
                                       'max_frame_num_to_track': count, 'propagation_direction': direction,
                                       'selected_object_ids': [10000, 10001], 'force_tracker_propagation': True},
                                      str(root/'masks')))
        expected = [1, 2, 3] if direction == 'forward' else [0]
        assert [x['frame'] for x in output] == expected, [x['frame'] for x in output]
        results['checks'].append(direction + ' propagation')
    engine.request({'type': 'add_prompt', 'frame_index': 2, 'obj_id': 10000,
                    'points': [[.4, .5], [.05, .05]], 'point_labels': [1, 0]})
    list(engine.propagate({'type': 'propagate_in_video', 'start_frame_index': 2,
                          'max_frame_num_to_track': 1, 'propagation_direction': 'forward',
                          'selected_object_ids': [10000, 10001], 'force_tracker_propagation': True}, str(root/'masks')))
    results['checks'].append('intermediate positive/negative correction and restart')
    for query in ({'text': 'person'}, {'bounding_boxes': [[.2, .2, .4, .5]], 'bounding_box_labels': [1]}):
        engine.request({'type': 'reset_session'})
        result = engine.request({'type': 'add_prompt', 'frame_index': 0, **query})
        assert 'out_binary_masks' in result['outputs']
        results['checks'].append('text request' if 'text' in query else 'normalized XYWH box request')
    results['gpu'] = engine.stats()
    (root/'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))
    print('UNTRAINED API DIAGNOSTIC ONLY. Artifacts:', root)


if __name__ == '__main__':
    main()
