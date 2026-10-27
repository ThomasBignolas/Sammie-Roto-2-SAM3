import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import shutil
import uuid
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from sammie.mask_contacts import combine_masks


def touching_pair(size=1080, gap=4):
    a = np.zeros((size, size), np.uint8)
    b = a.copy()
    a[size//6:5*size//6, size//6:size//2-gap//2] = 255
    b[size//6:5*size//6, size//2+(gap+1)//2:5*size//6] = 255
    return a, b


class ContactTests(unittest.TestCase):
    def test_disabled_is_exact_union_and_inputs_are_immutable(self):
        a, b = touching_pair()
        before = (a.copy(), b.copy())
        np.testing.assert_array_equal(combine_masks([a, b]), np.maximum(a, b))
        combine_masks([a, b], 8)
        np.testing.assert_array_equal(a, before[0])
        np.testing.assert_array_equal(b, before[1])

    def test_interior_contact_closed_but_exterior_and_opening_preserved(self):
        a, b = touching_pair()
        fixed = combine_masks([a, b], 8)
        self.assertEqual(fixed[540, 540], 255)
        self.assertEqual(fixed[180, 540], 0)  # open end, near exterior
        self.assertEqual(fixed[100, 540], 0)
        self.assertEqual(fixed[540, 170], 0)  # no grow of outer edge
        self.assertTrue(np.all(fixed[np.maximum(a, b)>0] == 255))

    def test_wide_gap_not_filled(self):
        masks = touching_pair(gap=40)
        np.testing.assert_array_equal(combine_masks(masks, 8), np.maximum(*masks))

    def test_single_object_hole_is_not_contact(self):
        a, b = touching_pair(gap=40)
        a[450:454, 300:304] = 0
        fixed = combine_masks([a, b], 8)
        self.assertFalse(fixed[450:454, 300:304].any())
        np.testing.assert_array_equal(combine_masks([a], 8), a)

    def test_empty_full_and_absent_object(self):
        empty = np.zeros((200, 300), np.uint8)
        full = np.full_like(empty, 255)
        self.assertIsNone(combine_masks([], 8))
        for masks, expected in [([empty, empty], empty), ([full, full], full), ([full, empty], full)]:
            np.testing.assert_array_equal(combine_masks(masks, 8), expected)
        a, _ = touching_pair()
        a[450:454, 300:304] = 0
        np.testing.assert_array_equal(combine_masks([a, np.zeros_like(a)], 8), a)

    def test_resolution_scaling_and_frame_border(self):
        for scale in (1, 2):
            a, b = touching_pair(1080*scale, 4*scale)
            a[-300*scale:, :540*scale-2*scale] = 255
            b[-300*scale:, 540*scale+2*scale:] = 255
            fixed = combine_masks([a, b], 8)
            self.assertEqual(fixed[540*scale, 540*scale], 255)
            self.assertEqual(fixed[-1, 540*scale], 255)
            self.assertEqual(fixed.shape, a.shape)
            self.assertEqual(fixed.dtype, np.uint8)

    def test_single_object_thin_gap_and_hair_unchanged(self):
        a, b = touching_pair(gap=40)
        a[200:800, 300:303] = 0
        a[100:180, 350] = 255
        np.testing.assert_array_equal(combine_masks([a, b], 8), np.maximum(a, b))

    def test_permutation_invariant(self):
        a, b = touching_pair()
        c = np.zeros_like(a); c[10:50, 10:50] = 255
        np.testing.assert_array_equal(combine_masks([a,b,c],8), combine_masks([c,b,a],8))

    def test_dimension_errors(self):
        with self.assertRaises(ValueError):
            combine_masks([np.zeros((2,3),np.uint8), np.zeros((3,2),np.uint8)], 8)

    def test_enclosed_remnant_of_repaired_contact_is_completed(self):
        a, b = touching_pair(gap=4)
        a[450:600, 535:538] = 0
        b[450:600, 542:545] = 0
        with patch('sammie.mask_contacts._complete_contact_remnants'):
            before = combine_masks([a,b],8)
        after = combine_masks([a,b],8)
        self.assertEqual(before[520,540],0)
        self.assertEqual(after[520,540],255)
        self.assertTrue(np.all(after >= before))
        # A pre-existing hole unrelated to contact correction is preserved.
        a[450:454,300:304] = 0
        self.assertFalse(combine_masks([a,b],8)[450:454,300:304].any())

    def test_broad_remnant_and_open_background_are_preserved(self):
        a,b=touching_pair(gap=4)
        a[450:600,520:538]=0; b[450:600,542:560]=0
        out=combine_masks([a,b],8)
        self.assertEqual(out[520,540],0)
        self.assertEqual(out[180,540],0)

    def test_intermittent_narrow_remnants_do_not_reappear(self):
        for extra in (0,1,2,3,2,1,0,3):
            a,b=touching_pair(gap=4)
            if extra:
                a[450:600,538-extra:538]=0
                b[450:600,542:542+extra]=0
            out=combine_masks([a,b],8)
            self.assertTrue((out[470:580,535:545]==255).all())


class ContactIntegrationTests(unittest.TestCase):
    def setUp(self):
        from sammie import core
        from sammie.settings_manager import initialize_settings
        Path('.test-output').mkdir(exist_ok=True)
        self.root = Path('.test-output').resolve()/('contacts-'+uuid.uuid4().hex)
        self.root.mkdir()
        self.settings = initialize_settings(str(self.root))
        self.settings.set_session_setting('contact_seam_width', 8)
        self.patches = [patch.object(core, 'mask_dir', str(self.root/'masks')),
                        patch.object(core, 'frames_dir', str(self.root/'frames')),
                        patch.object(core, 'matting_dir', str(self.root/'matting')),
                        patch.object(core.VideoInfo, 'width', 1080),
                        patch.object(core.VideoInfo, 'height', 1080)]
        for p in self.patches: p.start()
        self.masks = touching_pair()
        for folder in ('masks', 'matting'):
            (self.root/folder/'00000').mkdir(parents=True)
            for oid, mask in enumerate(self.masks):
                cv2.imwrite(str(self.root/folder/'00000'/f'{oid}.png'), mask)
        (self.root/'frames').mkdir()
        cv2.imwrite(str(self.root/'frames/00000.png'), np.zeros((1080,1080,3),np.uint8))

    def tearDown(self):
        for p in reversed(self.patches): p.stop()
        assert self.root.parent == Path('.test-output').resolve() and self.root.name.startswith('contacts-')
        shutil.rmtree(self.root)

    def test_display_and_combined_loader_agree_and_filter_preserves_object(self):
        from sammie import core
        points = [{'object_id':0}, {'object_id':1}]
        display = core.load_masks_for_frame(0, points, return_combined=True)
        guide, count = core.load_combined_mask(0, [0,1])
        self.assertEqual(count, 2)
        self.assertEqual(display[540,540], 255)
        np.testing.assert_array_equal(display, guide)
        np.testing.assert_array_equal(core.load_masks_for_frame(0, points, True, 0), self.masks[0])
        np.testing.assert_array_equal(core.load_combined_mask(0, [0,0])[0], self.masks[0])
        np.testing.assert_array_equal(core.load_combined_mask(0, [0,1], core.matting_dir)[0],
                                      np.maximum(*self.masks))
        for oid, mask in enumerate(self.masks):
            np.testing.assert_array_equal(cv2.imread(str(self.root/'masks/00000'/f'{oid}.png'),0),mask)

    def test_videomama_actual_input_uses_repaired_guide(self):
        from sammie.matting import VideoMaMaManager
        manager = VideoMaMaManager()
        self.settings.set_session_setting('matany_res', 1080)
        frames, masks, valid = manager._load_batch(0, 1, 0, 'png', combine_ids=[0,1])
        self.assertTrue(valid)
        self.assertEqual(masks[0][540,540], 255)
        _, single, _ = manager._load_batch(0, 1, 0, 'png')
        self.assertEqual(single[0][540,540], 0)
        self.settings.set_session_setting('contact_seam_width', 0)
        _, disabled, _ = manager._load_batch(0, 1, 0, 'png', combine_ids=[0,1])
        np.testing.assert_array_equal(disabled[0], np.maximum(*self.masks))

    def test_matanyone_guide_also_uses_repair(self):
        from sammie.matting import MatAnyManager
        tensor, size = MatAnyManager()._load_mask_for_matting(0, 0, 'cpu', combine_ids=[0,1])
        self.assertEqual(size, (1080,1080))
        self.assertEqual(float(tensor[540,540]), 255)

    def test_ui_control_and_project_setting_roundtrip(self):
        from PySide6.QtWidgets import QApplication
        from sammie_main import SegmentationTab
        app = QApplication.instance() or QApplication([])
        widget = SegmentationTab()
        self.assertEqual(widget.contact_seam_width_slider.value(), 8)
        widget.contact_seam_width_slider.setValue(6)
        self.assertEqual(self.settings.get_session_setting('contact_seam_width'), 6)
        self.assertTrue(self.settings.save_session_settings())
        self.settings.set_session_setting('contact_seam_width', 0)
        self.assertTrue(self.settings.load_session_settings())
        self.assertEqual(self.settings.get_session_setting('contact_seam_width'), 6)
        widget.close()


if __name__ == '__main__':
    unittest.main()
