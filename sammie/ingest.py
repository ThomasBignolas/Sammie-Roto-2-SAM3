"""Staged media ingestion with bounded decoded pixels and no GUI/session access."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import uuid
import av
import cv2
from .performance import cpu_budget, memory_budget


def discard(stage):
    stage = Path(stage).resolve()
    if not stage.name.startswith('.sammie-ingest-'):
        raise ValueError('Not an owned import staging directory.')
    if stage.exists():
        shutil.rmtree(stage)


def stage_directory(parent):
    path = Path(parent).resolve()/('.sammie-ingest-'+uuid.uuid4().hex)
    for name in ('frames', 'masks', 'matting'):
        (path/name).mkdir(parents=True)
    return path


def write_frame(path, pixels):
    # Keep OpenCV's original PNG strategy. Forcing compression=1 resets its
    # strategy and was slower/larger on the measured production frames.
    if not cv2.imwrite(str(path), pixels):
        raise OSError(f'Could not write frame: {path}')


def queue_limit(width, height, budget):
    # At most four in-flight writer arrays plus the decoder's current frame.
    return max(1, min(4, budget // max(1, width * height * 3)))


def prepare_video(path, parent, extension='png', threads=0,
                  progress=lambda n,t,s: None, cancel=lambda: False):
    if extension not in ('png', 'jpg'):
        raise ValueError('Working frame format must be PNG or JPG.')
    stage = stage_directory(parent)
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = 'AUTO'
            stream.codec_context.thread_count = min(2, cpu_budget(threads))
            rate = stream.average_rate or stream.guessed_rate
            if rate is None:
                raise ValueError('The source has no usable frame-rate metadata.')
            width, height = stream.width, stream.height
            source_space = int(stream.codec_context.colorspace)
            source_range = int(stream.codec_context.color_range)
            limit = queue_limit(width, height, memory_budget(128))
            timing, pending = [], deque()
            with ThreadPoolExecutor(max_workers=min(2, limit, cpu_budget(threads)), thread_name_prefix='sammie-import') as pool:
                try:
                    for index, frame in enumerate(container.decode(stream)):
                        if cancel(): raise InterruptedError('Import cancelled; previous session unchanged.')
                        # Await the oldest writer before allocating another array.
                        if len(pending) >= limit:
                            pending.popleft().result()
                        if (frame.width, frame.height) != (width, height):
                            raise ValueError(f'Frame {index} changes resolution; import stopped.')
                        pixels = frame.reformat(format='bgr24', src_colorspace=source_space,
                            dst_colorspace=1, src_color_range=source_range, dst_color_range=2).to_ndarray()
                        pending.append(pool.submit(write_frame, stage/'frames'/f'{index:05d}.{extension}', pixels))
                        timing.append(dict(index=index, pts=frame.pts, time_base=str(frame.time_base)))
                        progress(index+1, stream.frames or 0, f'Importing video: {index+1} frames')
                    for future in pending:
                        if cancel(): raise InterruptedError('Import cancelled; previous session unchanged.')
                        future.result()
                finally:
                    for future in pending: future.cancel()
            if not timing:
                raise ValueError('No video frames were decoded.')
            (stage/'source_timing.json').write_text(json.dumps(dict(fps=str(rate), frames=timing)))
            return stage, dict(width=width, height=height, fps=float(rate), fps_rational=str(rate),
                               total_frames=len(timing), color_space=source_space, frame_format=extension)
    except BaseException:
        discard(stage)
        raise


def prepare_images(files, parent, extension='png', threads=0,
                   progress=lambda n,t,s: None, cancel=lambda: False):
    stage = stage_directory(parent)
    try:
        first = cv2.imread(str(files[0]))
        if first is None: raise ValueError(f'Cannot decode image: {files[0]}')
        height, width = first.shape[:2]
        suffix = Path(files[0]).suffix.lower()
        copy = suffix in ('.png', '.jpg', '.jpeg')
        extension = ('jpg' if suffix == '.jpeg' else suffix[1:]) if copy else extension
        limit = queue_limit(width, height, memory_budget(128))
        def convert(index):
            if cancel(): raise InterruptedError('Import cancelled; previous session unchanged.')
            path = files[index]
            pixels = first if index == 0 else cv2.imread(str(path))
            if pixels is None: raise ValueError(f'Cannot decode image: {path}')
            if pixels.shape[:2] != (height, width):
                raise ValueError(f'Image dimensions differ: {path}')
            target = stage/'frames'/f'{index:05d}.{extension}'
            if copy:
                shutil.copy2(path, target)
            else:
                write_frame(target, pixels)
        with ThreadPoolExecutor(max_workers=min(2, limit, cpu_budget(threads)), thread_name_prefix='sammie-import') as pool:
            pending = deque()
            try:
                for index in range(len(files)):
                    if cancel(): raise InterruptedError('Import cancelled; previous session unchanged.')
                    if len(pending) >= limit: pending.popleft().result()
                    pending.append(pool.submit(convert, index))
                    progress(index, len(files), f'Importing images: {index+1}/{len(files)}')
                for future in pending: future.result()
                if cancel(): raise InterruptedError('Import cancelled; previous session unchanged.')
            finally:
                for future in pending: future.cancel()
        return stage, dict(width=width, height=height, fps=24., fps_rational='24', total_frames=len(files),
                           color_space=1, frame_format=extension)
    except BaseException:
        discard(stage)
        raise


def commit(stage, target):
    """Replace only an expendable session after every frame succeeded."""
    from .projects import MARKER
    stage, target = Path(stage).resolve(), Path(target).resolve()
    if (target/MARKER).exists() or stage.parent != target.parent or not stage.name.startswith('.sammie-ingest-'):
        raise ValueError('Unsafe import destination.')
    backup = target.parent/('.sammie-ingest-'+uuid.uuid4().hex)
    if target.exists(): target.rename(backup)
    try:
        stage.rename(target)
    except BaseException:
        if backup.exists(): backup.rename(target)
        raise
    if backup.exists(): discard(backup)


def prepare_dialog(path, target, manager, parent=None, images=False):
    from .background_io import run_io
    from .sammie import detect_image_sequence, get_sequence_start_frame
    from PySide6.QtWidgets import QMessageBox
    extension = manager.get_app_setting('frame_format', 'png')
    threads = manager.get_app_setting('cpu_threads', 0)
    files, base = [path], None
    if images:
        found, sequence, base = detect_image_sequence(str(path))
        if found:
            dialog = QMessageBox(parent)
            dialog.setWindowTitle('Image Sequence Detected')
            dialog.setText(f'This image belongs to a sequence of {len(sequence)} frames.')
            whole = dialog.addButton('Load Sequence', QMessageBox.AcceptRole)
            single = dialog.addButton('Load Single Image', QMessageBox.RejectRole)
            dialog.addButton('Cancel', QMessageBox.RejectRole)
            dialog.exec()
            if dialog.clickedButton() == whole: files = sequence
            elif dialog.clickedButton() != single: return None
    try:
        action = prepare_images if images else prepare_video
        stage, info = run_io(parent, 'Importing images' if images else 'Importing video',
            lambda progress, cancel: action(files if images else path, Path(target).resolve().parent,
                extension, threads, progress, cancel))
        info['sequence_start_frame'] = get_sequence_start_frame(files) if images else 0
        if images and len(files) > 1 and base:
            source = Path(files[0])
            info['video_file_path'] = str(source.with_name(base.rstrip('_-.')+source.suffix))
        return stage, info
    except InterruptedError:
        return None


def apply_info(info, manager):
    from . import core
    for key in ('width', 'height', 'fps', 'fps_rational', 'total_frames', 'color_space'):
        setattr(core.VideoInfo, key, info[key])
    for key in ('frame_format', 'sequence_start_frame', 'video_file_path'):
        if key in info: manager.set_session_setting(key, info[key])
    manager.set_session_setting('video_fps_rational', info['fps_rational'])
