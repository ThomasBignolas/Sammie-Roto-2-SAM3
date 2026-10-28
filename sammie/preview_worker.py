"""CPU-only native-resolution preview process, with an explicitly bounded LRU.

Legacy rendering uses module-global session state. A process, rather than a
thread, keeps that state isolated from edits, exports and inference in the UI.
"""
import contextlib
import json
import os
from pathlib import Path
from multiprocessing import shared_memory
import sys


def cache_key(request):
    from .performance import file_stamp
    mode = request['view'].get('view_mode')
    # Matting/AOV/foreground validate their own dependency chains. Never bypass
    # that validation with a display cache whose key only checks the current PNG.
    if mode not in ('None', 'Segmentation-Edit', 'Segmentation-Matte',
                    'Segmentation-Alpha', 'Segmentation-BGcolor', 'ObjectRemoval'):
        return None
    root = Path(request['root'])
    frame = request['frame']
    ids = sorted({int(p['object_id']) for p in request['points']})
    paths = [root/'frames'/f"{frame:05d}.{request['session']['frame_format']}"]
    paths += [root/'masks'/f'{frame:05d}'/f'{i}.png' for i in ids]
    if mode == 'ObjectRemoval':
        paths.append(root/'removal'/f'{frame:05d}.png')
    # Settings and points are snapshots, including unsaved controls/corrections.
    return json.dumps([request['root'], request['generation'], frame,
                       request['session'], request['view'], request['points'],
                       [file_stamp(p) for p in paths]], sort_keys=True)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Preview must not compete for GPU VRAM.
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    if os.name == 'nt':
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    protocol = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import torch
        from . import core, sammie
        from .settings_manager import initialize_settings, SessionSettings
        from .projects import bind_root
        from .performance import ArrayCache, configure_cpu, memory_budget
        manager = initialize_settings()
        core.DeviceManager._device = torch.device('cpu')
        manager.set_app_setting('force_cpu', True)
    cache = ArrayCache(0)
    decoded = ArrayCache(0)
    from .performance import file_stamp
    def load_frame(frame):
        import cv2
        path = Path(core.frames_dir)/f'{frame:05d}.{core.get_frame_extension()}'
        key = file_stamp(path)
        image = decoded.get(key)
        if image is None:
            image = cv2.imread(str(path))
            if image is None: raise OSError(f'Cannot decode preview frame: {path}')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.setflags(write=False)
            decoded.put(key, image)
        return image
    core.load_base_frame = load_frame
    region = None
    previous = None
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if region is not None:
                region.close()
                region.unlink()
                region = None
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    configure_cpu(request['threads'], torch)
                    limit = memory_budget(request['cache_mib'])
                    identity = (request['root'], request['generation'], request['cache_mib'])
                    if previous is None or identity[0] != previous[0] or identity[2] != previous[2] or limit < (cache.limit+decoded.limit)*.75:
                        decoded = ArrayCache(limit//4)
                        cache = ArrayCache(limit-limit//4)
                    elif identity != previous:
                        cache.clear()
                    if identity != previous:
                        previous = identity
                    bind_root(request['root'], manager)
                    manager.session_settings = SessionSettings(**request['session'])
                    for name, value in request['video'].items():
                        setattr(core.VideoInfo, name, value)
                    key = cache_key(request)
                    result = cache.get(key) if key is not None else None
                    hit = result is not None
                    if result is None:
                        result = sammie.update_image(request['frame'], request['view'], request['points'], return_numpy=True)
                        if result is not None:
                            result = np.ascontiguousarray(result)
                            if result.dtype != np.uint8:
                                raise ValueError('Preview renderer must return display uint8 pixels.')
                            if key is not None:
                                cache.put(key, result)
                    response = {'id': request['id'], 'frame': request['frame'], 'hit': hit}
                    if result is not None:
                        region = shared_memory.SharedMemory(create=True, size=result.nbytes)
                        np.ndarray(result.shape, result.dtype, buffer=region.buf)[:] = result
                        response.update(memory=region.name, shape=result.shape, bytes=cache.bytes+decoded.bytes)
                print(json.dumps(response), file=protocol, flush=True)
            except Exception as error:
                print(json.dumps({'id': request['id'], 'error': str(error)}), file=protocol, flush=True)
    finally:
        if region is not None:
            region.close()
            region.unlink()


if __name__ == '__main__':
    main()
