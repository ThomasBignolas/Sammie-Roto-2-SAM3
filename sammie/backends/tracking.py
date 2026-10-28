"""Shared tracking controls; frame indices are always zero based and inclusive."""
from pathlib import Path
import cv2
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QProgressDialog, QApplication
from sammie import core
from sammie.settings_manager import get_settings_manager

class TrackingMixin:
    def write_masks(self, frame, masks):
        target = Path(core.mask_dir) / f"{frame:05d}"
        target.mkdir(parents=True, exist_ok=True)
        for object_id, mask in masks.items():
            if mask.shape != (core.VideoInfo.height, core.VideoInfo.width):
                raise ValueError(f"Mask dimensions {mask.shape} do not match the source")
            if not cv2.imwrite(str(target / f"{object_id}.png"), mask):
                raise OSError(f"Could not write mask for frame {frame}, object {object_id}")

    def _propagate(self, parent_window, start_frame_idx, max_frame_num_to_track, reverse=False,
                    show_progress=True):
        """Track in a worker; only queued progress updates touch the timeline."""
        from sammie.background_io import run_compute
        frequency = get_settings_manager().get_app_setting('display_update_frequency', 5)
        available = start_frame_idx + 1 if reverse else core.VideoInfo.total_frames - start_frame_idx
        total = min(available, max_frame_num_to_track + 1) if max_frame_num_to_track is not None else available
        state = {'last': None, 'cancelled': False}
        def track(progress, cancel):
            iterator = self.iter_masks(start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_frame_num_to_track, reverse=reverse)
            try:
                for frame, masks in iterator:
                    if cancel():
                        state['cancelled'] = True
                        break
                    self.write_masks(frame, masks)
                    state['last'] = frame
                    progress(abs(frame-start_frame_idx)+1, total, f'Tracking frame {frame} - {"backward" if reverse else "forward"}')
            finally:
                iterator.close()
        def update(done, count, message):
            frame = state['last']
            if parent_window is not None and frame is not None and (not show_progress or frame % frequency == 0):
                parent_window.frame_slider.setValue(frame)
        try:
            run_compute(parent_window, 'Tracking', track, update)
        except InterruptedError:
            state['cancelled'] = True
        if parent_window is not None and state['last'] is not None:
            parent_window.frame_slider.setValue(state['last'])
        return state['last'], state['cancelled']

    def track_objects(self, parent_window):
        """Track all objects across the full in/out point range (or the entire video)."""
        frame_count = core.VideoInfo.total_frames
        settings_mgr = get_settings_manager()
        in_point = settings_mgr.get_session_setting("in_point", None)
        out_point = settings_mgr.get_session_setting("out_point", None)
        if in_point is None:
            in_point = 0
        frames_to_track = None
        total_frames = frame_count
        if out_point is not None:
            frames_to_track = out_point - in_point
            total_frames = frames_to_track + 1

        last_frame_idx, cancelled = self._propagate(
            parent_window, start_frame_idx=in_point, max_frame_num_to_track=frames_to_track, reverse=False)

        if not cancelled:
            self.propagated = (total_frames == frame_count)
            print("Tracking completed")
            return 1
        else:
            self.propagated = False
            print("Tracking cancelled")
            return 0

    def track_forward(self, parent_window, current_frame):
        """Track all objects forward from current_frame to the out point (or end of video)."""
        settings_mgr = get_settings_manager()
        out_point = settings_mgr.get_session_setting("out_point", None)
        last_frame = out_point if out_point is not None else core.VideoInfo.total_frames - 1
        if current_frame >= last_frame:
            print("Already at the last frame")
            return 1

        max_frame_num_to_track = max(last_frame - current_frame, 0)

        last_frame_idx, cancelled = self._propagate(
            parent_window, start_frame_idx=current_frame, max_frame_num_to_track=max_frame_num_to_track,
            reverse=False)

        if cancelled:
            print("Forward tracking cancelled")
            return 0
        print(f"Forward tracking completed up to frame {last_frame_idx}")
        return 1

    def track_backward(self, parent_window, current_frame):
        """Track all objects backward from current_frame to the in point (or start of video)."""
        settings_mgr = get_settings_manager()
        in_point = settings_mgr.get_session_setting("in_point", None)
        if in_point is None:
            in_point = 0
        if current_frame <= in_point:
            print("Already at the first frame")
            return 1

        max_frame_num_to_track = max(current_frame - in_point, 0)

        last_frame_idx, cancelled = self._propagate(
            parent_window, start_frame_idx=current_frame, max_frame_num_to_track=max_frame_num_to_track,
            reverse=True)

        if cancelled:
            print("Backward tracking cancelled")
            return 0
        print(f"Backward tracking completed back to frame {last_frame_idx}")
        return 1

    def track_one_frame_forward(self, parent_window, current_frame):
        """Track all objects one frame forward from current_frame. Returns the new frame index."""
        last_frame = core.VideoInfo.total_frames - 1
        if current_frame >= last_frame:
            print("Already at the last frame")
            return current_frame

        last_frame_idx, _ = self._propagate(
            parent_window, start_frame_idx=current_frame, max_frame_num_to_track=1,
            reverse=False, show_progress=False)

        return last_frame_idx if last_frame_idx is not None else current_frame

    def track_one_frame_backward(self, parent_window, current_frame):
        """Track all objects one frame backward from current_frame. Returns the new frame index."""
        if current_frame <= 0:
            print("Already at the first frame")
            return current_frame

        last_frame_idx, _ = self._propagate(
            parent_window, start_frame_idx=current_frame, max_frame_num_to_track=1,
            reverse=True, show_progress=False)

        return last_frame_idx if last_frame_idx is not None else current_frame

