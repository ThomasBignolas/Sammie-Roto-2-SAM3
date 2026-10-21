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
from .concepts import scoped_id, split_id, MAX_OBJECTS


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
        self.sessions = {}
        self.resource = None

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
            from .compat import configure_multiplex_output
            configure_multiplex_output(self.predictor.model)
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
        for group in list(self.sessions):
            self._close_group(group)
        self.resource = resource
        return self._start_group(0)

    def _close_group(self, group):
        session = self.sessions.pop(group, None)
        if session:
            self.predictor.handle_request({'type': 'close_session', 'session_id': session})
        if group == 0:
            self.session = None

    def _start_group(self, group):
        import torch
        from .frame_cache import install_bounded_loader
        install_bounded_loader()
        if self.resource is None or not 0 <= group < MAX_OBJECTS:
            raise ValueError('Start a video before creating a valid concept session.')
        # Pinned upstream's base wrapper passes offload_state_to_cpu to the 3.1 model,
        # whose init_state does not accept it. Use the actual model signature explicitly.
        options = {'resource_path': self.resource, 'offload_video_to_cpu': True,
                   'offload_state_to_cpu': True, 'async_loading_frames': True}
        params = inspect.signature(self.predictor.model.init_state).parameters
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            state = self.predictor.model.init_state(**{k: v for k, v in options.items() if k in params})
        session = str(uuid.uuid4())
        now = time.time()
        self.predictor._all_inference_states[session] = {
            'state': state, 'session_id': session, 'start_time': now, 'last_use_time': now}
        self.sessions[group] = session
        if group == 0:
            self.session = session
        return {'frames': state['num_frames'], 'width': state['orig_width'],
                'height': state['orig_height'], 'state_offload': 'offload_state_to_cpu' in params}

    def request(self, request):
        import torch
        if self.resource is None:
            raise RuntimeError('Start a video session first')
        request = dict(request)
        group = int(request.pop('group_id', 0))
        if request['type'] == 'reset_session':
            for existing in list(self.sessions):
                self._close_group(existing)
            return {'is_success': True}
        if request['type'] == 'close_group':
            self._close_group(group)
            return {'is_success': True}
        if 'obj_id' in request:
            group, request['obj_id'] = split_id(request['obj_id'])
        if group not in self.sessions:
            self._start_group(group)
        request['session_id'] = self.sessions[group]
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            result = self.predictor.handle_request(request)
        return dict(result, group_id=group)

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
            public_id = scoped_id(response.get('group_id', 0), oid)
            path = target / f'{public_id}.png'
            if not cv2.imwrite(str(path), mask):
                raise OSError(f'Could not write {path}')
            paths[str(public_id)] = str(path)
        return {'frame': int(response['frame_index']), 'masks': paths, 'gpu': self.stats()}

    def propagate(self, request, output_dir):
        # One checkpoint, independent native concept sessions. Advance each session
        # once per frame, then emit one aligned set of public object IDs. No union
        # prompts, point approximations, or cross-concept ID suppression are used.
        groups = {}
        for object_id in request.get('selected_object_ids', []):
            group, native = split_id(object_id)
            groups.setdefault(group, []).append(native)
        if not groups:
            groups = {0: []}
        streams = []
        try:
            for group, ids in groups.items():
                if group not in self.sessions:
                    raise RuntimeError('Concept session is missing; replay the saved selection.')
                streams.append(self._propagate_group(dict(request, selected_object_ids=ids), output_dir, group))
            while True:
                results = [next(stream, None) for stream in streams]
                if all(result is None for result in results):
                    return
                if any(result is None for result in results) or len({r['frame'] for r in results}) != 1:
                    raise RuntimeError('Concept trackers returned different frame sequences.')
                masks = {}
                for result in results:
                    masks.update(result['masks'])
                yield {'frame': results[0]['frame'], 'masks': masks, 'gpu': self.stats()}
        finally:
            for stream in reversed(streams):
                stream.close()

    def _propagate_group(self, request, output_dir, group):
        import torch
        session = self.sessions[group]
        request = dict(request)
        selected = request.pop('selected_object_ids', [])
        synthetic_action = None
        if self.version == 'SAM 3.1' and selected:
            # 3.1 lacks SAM3's force_tracker_propagation flag. Its public history
            # method requests partial tracker propagation for all selected objects.
            # This prevents a second directional pass being treated as a cache fetch.
            state = self.predictor._all_inference_states[session]['state']
            self.predictor.model.add_action_history(
                state, action_type='refine', obj_ids=selected,
                frame_idx=request.get('start_frame_index'))
            synthetic_action = state['action_history'][-1]
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            iterator = self.predictor.handle_stream_request(dict(request, session_id=session))
            try:
                for response in iterator:
                    yield self.output(dict(response, group_id=group), output_dir)
            finally:
                try:
                    iterator.close()
                finally:
                    if synthetic_action is not None:
                        # This requests a propagation pass; it is not a user point
                        # refinement. Leaving it in history skips the official first-
                        # correction object extraction and mixes multiplex tensor sizes.
                        state['action_history'][:] = [a for a in state['action_history']
                                                     if a is not synthetic_action]

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
            emit({'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()})
            gc.collect()


if __name__ == '__main__':
    main()
