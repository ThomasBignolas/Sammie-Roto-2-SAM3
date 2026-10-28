"""SAM3/3.1 backend: official requests, stable selected IDs, source-sized masks."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
from sammie import core
from sammie.settings_manager import get_settings_manager
from .bridge import ROOT, WorkerClient
from .masks import binary_mask, normalize_points, normalize_box, pack_mask, match_seeds
from .tracking import TrackingMixin
from .concepts import split_concepts, split_id, MAX_OBJECTS


class Sam3Backend(TrackingMixin, core.CallbackMixin):
    supports_text = True
    supports_boxes = True
    supports_preview = False  # Official API has no reversible preview transaction.

    def __init__(self):
        super().__init__()
        self.loaded_model_name = None
        self.predictor = None
        self.inference_state = None
        self.propagated = self.deduplicated = False
        self.journal = []
        self.id_map = {}
        self.gpu = {}
        self.resource = None
        self.candidates = None
        self.candidate_labels = {}
        self.discovery_misses = []
        self.options = None
        self.transport = Path(core.temp_dir).resolve() / 'sam3_transport'

    def _status(self, text):
        print(text)
        from sammie.background_io import report
        report(text)
        self._notify('status', text=text)

    def load_segmentation_model(self, model=None, parent_window=None):
        self.loaded_model_name = model or get_settings_manager().get_session_setting('sam_model', 'SAM 3')
        filename = 'sam3.1_multiplex.pt' if self.loaded_model_name == 'SAM 3.1' else 'sam3.pt'
        checkpoint = ROOT / 'checkpoints' / filename
        if not checkpoint.is_file():
            raise FileNotFoundError(f'{filename} is missing. Request access at https://huggingface.co/facebook/'
                                    f'{"sam3.1" if self.loaded_model_name == "SAM 3.1" else "sam3"} '
                                    'then run setup_models.bat, or place the checkpoint in checkpoints/.')
        self.predictor = WorkerClient(self._status)
        try:
            self.gpu = self.predictor.call('load', version=self.loaded_model_name, checkpoint=str(checkpoint))
        except Exception:
            self.unload_segmentation_model()
            raise
        self._status(f'{self.loaded_model_name} loaded. GPU peak: {self.gpu["peak_mib"]} MiB')
        return True

    def unload_segmentation_model(self):
        if self.predictor:
            self.predictor.close()
        self.predictor = self.inference_state = None

    def offload_model_to_cpu(self):
        # Exiting the worker releases model AND all GPU session tensors before matting.
        self.unload_segmentation_model()
        self._status('SAM 3 released GPU memory for matting; prompts are retained.')

    def load_model_to_device(self):
        # Lazy restoration avoids reloading a multi-GB checkpoint merely to view/export mattes.
        pass

    def _ensure(self):
        if self.predictor is None:
            journal = copy.deepcopy(self.journal)
            self.load_segmentation_model(self.loaded_model_name)
            self.initialize_predictor()
            self.replay_points(journal)

    def initialize_predictor(self):
        if self.predictor is None or not core.VideoInfo.total_frames:
            return
        settings = get_settings_manager()
        edge = int(settings.get_session_setting('sam3_working_edge', 1920))
        self.options = edge
        self.resource = self._prepare_frames(edge)
        self.inference_state = self.predictor.call('start', resource=str(self.resource))
        if self.inference_state['frames'] != core.VideoInfo.total_frames:
            raise RuntimeError('SAM 3 frame count differs from the source; tracking stopped.')
        self.journal = []
        self.id_map = {}

    def _prepare_frames(self, edge):
        if edge not in (0, 1280, 1920, 2560):
            raise ValueError('Unsupported SAM 3 working resolution')
        source = Path(core.frames_dir).resolve()
        extension = core.get_frame_extension()
        paths = [source / f'{i:05d}.{extension}' for i in range(core.VideoInfo.total_frames)]
        if not paths or any(not path.is_file() for path in paths):
            raise RuntimeError('Source sequence is incomplete; frame indices must be contiguous.')
        if edge == 0 or max(core.VideoInfo.width, core.VideoInfo.height) <= edge:
            return source
        fingerprint = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
        digest = hashlib.sha256(json.dumps([edge, fingerprint]).encode()).hexdigest()[:20]
        destination = Path(core.temp_dir).resolve() / 'sam3_proxy' / digest
        destination.mkdir(parents=True, exist_ok=True)
        ratio = edge / max(core.VideoInfo.width, core.VideoInfo.height)
        size = (round(core.VideoInfo.width * ratio), round(core.VideoInfo.height * ratio))
        missing = [(i, p) for i, p in enumerate(paths) if not (destination/f'{i:05d}.png').is_file()]
        if not missing:
            self._status(f'Reusing {len(paths)} prepared working frames')
            return destination
        from sammie.background_io import run_io
        def prepare(progress, cancel):
            for n, (i, path) in enumerate(missing):
                if cancel(): raise InterruptedError('Working-frame preparation cancelled.')
                image = cv2.imread(str(path))
                if image is None: raise OSError(f'Could not decode {path}')
                image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
                target = destination/f'{i:05d}.png'
                temporary = destination/f'{i:05d}.part.png'
                try:
                    if not cv2.imwrite(str(temporary), image):
                        raise OSError(f'Could not write {target}')
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
                progress(n+1, len(missing), f'Preparing working frames: {n+1}/{len(missing)}')
        run_io(None, 'Preparing SAM working frames', prepare)
        self._status(f'Prepared {len(missing)} working frames; reused {len(paths)-len(missing)}')
        return destination

    def _request(self, request):
        try:
            result = self.predictor.call('request', request=request, output_dir=str(self.transport))
        except Exception:
            # A failed CUDA operation may have partially mutated the tracker. Rebuild
            # from committed prompts on the next edit, and release memory immediately.
            self.unload_segmentation_model()
            raise
        if result and 'gpu' in result:
            self.gpu = result['gpu']
        return result

    def _read_masks(self, result):
        masks = {}
        for oid, path in result.get('masks', {}).items():
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise OSError(f'Worker mask is missing: {path}')
            masks[int(oid)] = mask
        return masks

    def _source_masks(self, masks):
        return {self.id_map[oid]: binary_mask(mask, core.VideoInfo.width, core.VideoInfo.height)
                for oid, mask in masks.items() if oid in self.id_map}

    def discover(self, frame, text=None, box=None):
        self._ensure()
        phrases = split_concepts(text)
        queries = [{'type': 'add_prompt', 'frame_index': frame, 'text': phrase} for phrase in phrases]
        if not queries and box is not None:
            queries = [{'type': 'add_prompt', 'frame_index': frame}]
        if box is not None:
            for query in queries:
                query['bounding_boxes'] = [normalize_box(box, core.VideoInfo.width, core.VideoInfo.height)]
                query['bounding_box_labels'] = [1]
        if not queries:
            raise ValueError('Enter an object description or choose a box')
        occupied = {split_id(oid)[0] for oid in self.id_map}
        free = [group for group in range(MAX_OBJECTS) if group not in occupied]
        if len(queries) > len(free) or len(self.id_map) >= MAX_OBJECTS:
            raise ValueError('Up to 16 selected objects/concept sessions are supported. Remove an object before adding more.')
        self.candidates, self.candidate_labels, self.discovery_misses = {}, {}, []
        # Temporary discovery states are released after each search. Committed
        # trackers and their temporal history remain untouched, including Cancel.
        for index, (query, group) in enumerate(zip(queries, free)):
            query['group_id'] = group
            label = query.get('text', 'Box selection')
            self._status(f'Searching {index+1}/{len(queries)}: {label}')
            result = self._request(query)
            masks = self._read_masks(result)
            self._request({'type': 'close_group', 'group_id': group})
            masks = {oid: mask for oid, mask in masks.items() if np.any(mask)}
            if not masks:
                self.discovery_misses.append(label)
            for number, (oid, mask) in enumerate(masks.items(), 1):
                self.candidates[oid] = (query, mask)
                self.candidate_labels[oid] = f'{label} — instance {number}'
        return {oid: value[1] for oid, value in self.candidates.items()}

    def accept_candidates(self, selected):
        selected = list(dict.fromkeys(map(int, selected)))
        if not self.candidates or not selected:
            self.candidates = None
            return []
        if any(oid not in self.candidates for oid in selected):
            raise ValueError('Selection is stale; search again.')
        if len(selected) + len(self.id_map) > MAX_OBJECTS:
            raise ValueError('Select at most 16 objects for this memory profile')
        records = []
        next_id = max(self.id_map.values(), default=-1) + 1
        for app_id, oid in enumerate(selected, next_id):
            query, mask = self.candidates[oid]
            ys, xs = np.nonzero(mask)
            if not len(xs):
                continue
            records.append({'kind': 'sam3_seed', 'frame': query['frame_index'], 'object_id': app_id,
                            'positive': True,
                            'x': int(xs.mean() * core.VideoInfo.width / mask.shape[1]),
                            'y': int(ys.mean() * core.VideoInfo.height / mask.shape[0]),
                            'query': query, 'seed_mask': pack_mask(mask)})
        old_map = dict(self.id_map)
        try:
            self._restore_seeds(records)
        except Exception:
            self.id_map = old_map
            self.unload_segmentation_model()
            raise
        self.journal.extend(copy.deepcopy(records))
        self.candidates = None
        self.propagated = self.deduplicated = False
        return records

    def _restore_seeds(self, seeds):
        groups = {}
        for seed in seeds:
            key = json.dumps(seed['query'], sort_keys=True)
            groups.setdefault(key, []).append(seed)
        pending = []
        for records in groups.values():
            query = records[0]['query']
            masks = self._read_masks(self._request(query))
            mapping = match_seeds(records, masks)
            for oid in masks:
                if oid not in mapping:
                    self._request({'type': 'remove_object', 'obj_id': oid, 'is_user_action': False,
                                   'frame_index': query['frame_index']})
            self.id_map.update(mapping)
            pending.append((query['frame_index'], {mapping[oid]: binary_mask(masks[oid],
                core.VideoInfo.width, core.VideoInfo.height) for oid in mapping}))
        for frame, masks in pending:
            self.write_masks(frame, masks)

    def segment_image(self, frame_number, object_id, input_points, input_labels):
        self._ensure()
        if object_id not in self.id_map.values():
            if len(self.id_map) >= 16:
                raise ValueError('This memory profile supports up to 16 selected objects')
            self.id_map[10000 + object_id] = object_id
        backend_id = next(oid for oid, app in self.id_map.items() if app == object_id)
        points = normalize_points(input_points, core.VideoInfo.width, core.VideoInfo.height)
        if len(points) != len(input_labels) or any(int(x) not in (0, 1) for x in input_labels):
            raise ValueError('Each point needs a positive (1) or negative (0) label')
        result = self._request({'type': 'add_prompt', 'frame_index': frame_number, 'obj_id': backend_id,
                                'points': points, 'point_labels': list(map(int, input_labels)),
                                'clear_old_points': True})
        masks = self._source_masks(self._read_masks(result))
        if object_id not in masks:
            raise RuntimeError('SAM 3 did not return the requested object; no mask was overwritten.')
        self.write_masks(frame_number, {object_id: masks[object_id]})
        self.journal = [p for p in self.journal if p.get('kind') == 'sam3_seed' or
                        (p['frame'], p['object_id']) != (frame_number, object_id)]
        self.journal.extend({'frame': frame_number, 'object_id': object_id, 'x': float(x), 'y': float(y),
                             'positive': bool(label)} for (x, y), label in zip(input_points, input_labels))
        self.propagated = self.deduplicated = False
        self._notify('segmentation_complete', frame=frame_number, object_id=object_id)

    def preview_point(self, *args):
        return None

    def reset_session(self):
        if self.predictor and self.inference_state:
            self._request({'type': 'reset_session'})
        self.id_map = {}
        self.journal = []
        self.candidates = None

    def replay_points(self, points_list):
        self._ensure()
        if points_list == self.journal:
            return
        self.reset_session()
        self.journal = []
        seeds = [p for p in points_list if p.get('kind') == 'sam3_seed']
        if seeds:
            try:
                self._restore_seeds(seeds)
            except Exception:
                self.journal = copy.deepcopy(points_list)
                self.unload_segmentation_model()
                raise
            self.journal = copy.deepcopy(seeds)
        groups = {}
        for point in points_list:
            if point.get('kind') != 'sam3_seed':
                groups.setdefault((point['frame'], point['object_id']), []).append(point)
        for (frame, oid), points in sorted(groups.items()):
            # Replay each click to preserve official refinement semantics.
            for count in range(1, len(points)+1):
                from sammie.background_io import check_cancelled, report
                check_cancelled()
                report(f'Restoring SAM 3 prompts: frame {frame+1}, object {oid}, click {count}/{len(points)}')
                subset = points[:count]
                self.segment_image(frame, oid, [[p['x'], p['y']] for p in subset],
                                   [int(p['positive']) for p in subset])
        self.journal = copy.deepcopy(points_list)
        self._notify('replay_complete')

    def remove_object(self, object_id):
        remaining = [p for p in self.journal if p['object_id'] != object_id]
        self.replay_points(remaining)

    def iter_masks(self, start_frame_idx, max_frame_num_to_track=None, reverse=False):
        self._ensure()
        if not self.journal:
            raise RuntimeError('Select an object before tracking')
        end = (0 if reverse else core.VideoInfo.total_frames-1) if max_frame_num_to_track is None else (
            start_frame_idx + (-1 if reverse else 1) * max_frame_num_to_track)
        request = {'type': 'propagate_in_video', 'start_frame_index': start_frame_idx,
                   'propagation_direction': 'backward' if reverse else 'forward',
                   'max_frame_num_to_track': max_frame_num_to_track,
                   'selected_object_ids': list(self.id_map),
                   'force_tracker_propagation': True}
        started = time.perf_counter()
        frames = 0
        iterator = self.predictor.stream('propagate', request=request, output_dir=str(self.transport))
        seen = set()
        step = -1 if reverse else 1
        expected = set(range(start_frame_idx - 1 if reverse else start_frame_idx, end + step, step))
        try:
            for result in iterator:
                frame = result['frame']
                if frame not in expected or frame in seen:
                    raise RuntimeError(f'Unexpected frame {frame} returned for the requested range')
                seen.add(frame)
                self.gpu = result['gpu']
                masks = self._source_masks(self._read_masks(result))
                # Missing/occluded objects must erase stale output from earlier propagation.
                for app_id in self.id_map.values():
                    masks.setdefault(app_id, np.zeros((core.VideoInfo.height, core.VideoInfo.width), np.uint8))
                frames += 1
                yield frame, masks
            if seen != expected:
                raise RuntimeError(f'Tracking omitted frames {sorted(expected-seen)[:10]}; output is incomplete.')
        except Exception:
            self.unload_segmentation_model()
            raise
        finally:
            iterator.close()
            seconds = time.perf_counter() - started
            self._status(f'Tracked {frames} frames in {seconds:.2f}s; GPU peak {self.gpu.get("peak_mib", 0)} MiB')

    def clear_tracking(self):
        target = Path(core.mask_dir).resolve()
        if target.parent != Path(core.temp_dir).resolve():
            raise ValueError('Unexpected mask cache location')
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        self.reset_session()
        self.journal = []
        self.propagated = self.deduplicated = False
