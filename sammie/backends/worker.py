"""Isolated official SAM3 runtime. No Qt, SAM2, or MatAnyone imports here."""
import contextlib
import gc
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid


def configure_runtime():
    root = Path(__file__).resolve().parents[2] / '.runtime'
    os.environ.setdefault('TRITON_CACHE_DIR', str(root / 'triton-cache'))
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(root / 'torch-cache'))


class Sam3Engine:
    def __init__(self, emit=lambda message: None):
        self.emit = emit
        self.predictor = None
        self.session = None
        self.version = None

    def load(self, version, checkpoint):
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('Official SAM 3 video requires an NVIDIA CUDA GPU with BF16 support.')
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f'Checkpoint missing: {checkpoint}. Request Meta access and run setup_models.bat.')
        if self.predictor is not None and self.version == version:
            return self.stats()
        self.emit({'status': f'Loading {version} checkpoint on CUDA (BF16 inference)...'})
        from sam3.model_builder import build_sam3_video_predictor, build_sam3_multiplex_video_predictor
        from .compat import configure_attention
        configure_attention()
        torch.cuda.reset_peak_memory_stats()
        if version == 'SAM 3.1':
            self.predictor = build_sam3_multiplex_video_predictor(
                checkpoint_path=checkpoint, use_fa3=False, compile=False,
                warm_up=False, max_num_objects=16, async_loading_frames=True,
                session_expiration_sec=0)
        elif version == 'SAM 3':
            self.predictor = build_sam3_video_predictor(
                checkpoint_path=checkpoint, gpus_to_use=[0], compile=False,
                async_loading_frames=True)
        else:
            raise ValueError(f'Unsupported model: {version}')
        self.version = version
        self.predictor.model.max_num_objects = 16
        return self.stats()

    def start(self, resource):
        import torch
        from .frame_cache import install_bounded_loader
        install_bounded_loader()
        if self.session:
            self.predictor.handle_request({'type': 'close_session', 'session_id': self.session})
        # Pinned upstream's base wrapper passes offload_state_to_cpu to the 3.1 model,
        # whose init_state does not accept it. Use the actual model signature explicitly.
        options = {'resource_path': resource, 'offload_video_to_cpu': True,
                   'offload_state_to_cpu': True, 'async_loading_frames': True}
        params = inspect.signature(self.predictor.model.init_state).parameters
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            state = self.predictor.model.init_state(**{k: v for k, v in options.items() if k in params})
        self.session = str(uuid.uuid4())
        now = time.time()
        self.predictor._all_inference_states[self.session] = {
            'state': state, 'session_id': self.session, 'start_time': now, 'last_use_time': now}
        return {'frames': state['num_frames'], 'width': state['orig_width'],
                'height': state['orig_height'], 'state_offload': 'offload_state_to_cpu' in params}

    def request(self, request):
        import torch
        if not self.session:
            raise RuntimeError('Start a video session first')
        request = dict(request, session_id=self.session)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            return self.predictor.handle_request(request)

    def output(self, response, output_dir):
        import cv2
        import numpy as np
        from .masks import binary_mask
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        outputs = response['outputs']
        ids = outputs['out_obj_ids']
        masks = outputs['out_binary_masks']
        paths = {}
        for oid, mask in zip(ids, masks):
            if hasattr(mask, 'detach'):
                mask = mask.detach().cpu().numpy()
            mask = np.asarray(mask)
            mask = binary_mask(mask, mask.shape[-1], mask.shape[-2])
            path = target / f'{int(oid)}.png'
            if not cv2.imwrite(str(path), mask):
                raise OSError(f'Could not write {path}')
            paths[str(int(oid))] = str(path)
        return {'frame': int(response['frame_index']), 'masks': paths, 'gpu': self.stats()}

    def propagate(self, request, output_dir):
        import torch
        request = dict(request)
        selected = request.pop('selected_object_ids', [])
        if self.version == 'SAM 3.1' and selected:
            # 3.1 lacks SAM3's force_tracker_propagation flag. Its public history
            # method requests partial tracker propagation for all selected objects.
            # This prevents a second directional pass being treated as a cache fetch.
            state = self.predictor._all_inference_states[self.session]['state']
            self.predictor.model.add_action_history(
                state, action_type='refine', obj_ids=selected,
                frame_idx=request.get('start_frame_index'))
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            iterator = self.predictor.handle_stream_request(dict(request, session_id=self.session))
            try:
                for response in iterator:
                    yield self.output(response, output_dir)
            finally:
                iterator.close()

    def stats(self):
        import torch
        free, total = torch.cuda.mem_get_info()
        return {'allocated_mib': torch.cuda.memory_allocated() // 2**20,
                'reserved_mib': torch.cuda.memory_reserved() // 2**20,
                'peak_mib': torch.cuda.max_memory_allocated() // 2**20,
                'free_mib': free // 2**20, 'total_mib': total // 2**20}


def main():
    configure_runtime()
    protocol = sys.stdout
    # Third-party progress/logs cannot contaminate the JSON protocol.
    sys.stdout = sys.stderr
    def emit(message):
        protocol.write(json.dumps(message) + '\n')
        protocol.flush()
    engine = Sam3Engine(emit)
    for line in sys.stdin:
        try:
            data = json.loads(line)
            command = data.pop('command')
            if command == 'load':
                result = engine.load(**data)
            elif command == 'start':
                result = engine.start(**data)
            elif command == 'request':
                response = engine.request(data['request'])
                result = engine.output(response, data['output_dir']) if 'outputs' in response else response
            elif command == 'propagate':
                iterator = engine.propagate(**data)
                try:
                    for response in iterator:
                        emit({'result': response})
                        ack = sys.stdin.readline()
                        if not ack or not json.loads(ack).get('continue'):
                            break
                finally:
                    iterator.close()
                emit({'done': True})
                continue
            else:
                raise ValueError(f'Unknown command: {command}')
            emit({'result': result})
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            emit({'error': f'{type(error).__name__}: {error}'})
            gc.collect()


if __name__ == '__main__':
    main()
