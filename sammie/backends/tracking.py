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
        """Core propagation loop shared by all tracking functions.

        Args:
            parent_window: Used for the progress dialog and to nudge the frame slider as we go.
            start_frame_idx: Frame to start propagating from.
            max_frame_num_to_track: How many additional frames to propagate beyond the start
                frame, or None to propagate to the end (or beginning, if reverse=True) of the video.
            reverse: If True, propagate backward toward frame 0 instead of forward.
            show_progress: If False, skips the progress dialog - intended for single-frame steps
                where a modal dialog would just be visual noise.

        Returns:
            (last_frame_idx, cancelled) - last_frame_idx is the last frame actually processed
            (None if nothing was processed), cancelled is True if the user hit Cancel.
        """
        settings_mgr = get_settings_manager()
        display_update_frequency = settings_mgr.get_app_setting("display_update_frequency", 5)
        total_frames = (max_frame_num_to_track + 1) if max_frame_num_to_track is not None else core.VideoInfo.total_frames

        progress_dialog = None
        if show_progress:
            progress_dialog = QProgressDialog("Tracking...", "Cancel", 0, 100, parent_window)
            progress_dialog.setWindowTitle("Progress")
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setAutoClose(True)
            progress_dialog.show()

        last_frame_idx = None
        cancelled = False

        iterator = self.iter_masks(
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_frame_num_to_track, reverse=reverse)
        try:
            for out_frame_idx, masks in iterator:
                self.write_masks(out_frame_idx, masks)

                last_frame_idx = out_frame_idx

                if progress_dialog is not None:
                    frames_processed = abs(out_frame_idx - start_frame_idx) + 1
                    progress_dialog.setValue(int(frames_processed * 100 / total_frames))

                # Update display at the specified frequency (always update for quick, dialog-less steps)
                if not show_progress or out_frame_idx % display_update_frequency == 0:
                    try:
                        parent_window.frame_slider.setValue(out_frame_idx)
                    except Exception as e:
                        print(f"Error updating display: {e}")

                QApplication.processEvents()
                if progress_dialog is not None and progress_dialog.wasCanceled():
                    cancelled = True
                    break

        finally:
            iterator.close()
            if progress_dialog is not None:
                progress_dialog.close()

        if progress_dialog is not None:
            if cancelled:
                progress_dialog.close()
            else:
                progress_dialog.setValue(100)

        # Always leave the slider showing the last frame actually processed
        if last_frame_idx is not None:
            try:
                parent_window.frame_slider.setValue(last_frame_idx)
                QApplication.processEvents()
            except Exception as e:
                print(f"Error updating display: {e}")

        return last_frame_idx, cancelled

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

