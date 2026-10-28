import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from dataclasses import asdict
from fractions import Fraction
import json
from pathlib import Path
import shutil
import threading
import time
import unittest
import uuid
from unittest.mock import patch, Mock
import av
import cv2
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QProgressDialog
from sammie import core, projects, sammie
from sammie.performance import ArrayCache, cpu_budget
from sammie.settings_manager import initialize_settings


class PerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.root = Path('.test-output').resolve()/('performance-unit-'+uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.source = self.root/'source'
        (self.source/'frames').mkdir(parents=True)
        self.manager = initialize_settings(str(self.source))
        self.manager.app_settings_file = str(self.root/'app.conf')
        self.manager.app_settings.auto_load_model = False
        self.manager.app_settings.preview_cache_mib = 16
        self.previous = {k:getattr(core,k) for k in ('temp_dir','frames_dir','mask_dir','backup_dir','matting_dir','removal_dir')}
        self.video = {k:getattr(core.VideoInfo,k) for k in ('width','height','total_frames','fps','color_space')}
        projects.bind_root(self.source, self.manager)
        core.VideoInfo.width, core.VideoInfo.height, core.VideoInfo.total_frames = 96, 64, 2
        for frame in range(2):
            cv2.imwrite(str(self.source/'frames'/f'{frame:05d}.png'), np.full((64,96,3), 60+frame*80, np.uint8))
            folder = self.source/'masks'/f'{frame:05d}'; folder.mkdir(parents=True)
            cv2.imwrite(str(folder/'0.png'), np.full((64,96),255,np.uint8))
        self.points = [dict(frame=0, object_id=0, x=40, y=30, positive=True)]

    def tearDown(self):
        for k,v in self.previous.items(): setattr(core,k,v)
        for k,v in self.video.items(): setattr(core.VideoInfo,k,v)
        assert self.root.parent == Path('.test-output').resolve() and self.root.name.startswith('performance-unit-')
        shutil.rmtree(self.root)

    def test_byte_cache_eviction_replacement_and_oversize(self):
        c=ArrayCache(10)
        c.put('a',np.zeros(6,np.uint8)); c.put('b',np.zeros(4,np.uint8))
        c.get('a'); c.put('c',np.zeros(4,np.uint8))
        self.assertIsNone(c.get('b')); self.assertEqual(c.bytes,10)
        c.put('a',np.zeros(2,np.uint8)); self.assertEqual(c.bytes,6)
        c.put('large',np.zeros(12,np.uint8)); self.assertEqual(c.bytes,6)
        c.clear(); self.assertEqual(c.bytes,0)
        with patch('os.cpu_count',return_value=14): self.assertEqual(cpu_budget(),4)

    def test_overlay_pixels_match_original_including_overlaps(self):
        rng=np.random.default_rng(3)
        image=rng.integers(0,256,(64,96,3),dtype=np.uint8)
        masks={0:rng.integers(0,2,(64,96),dtype=np.uint8)*255,3:rng.integers(0,2,(64,96),dtype=np.uint8)*255}
        colors=np.zeros_like(image); active=np.zeros(image.shape[:2],bool)
        for oid,mask in masks.items():
            active |= mask>0
            colors[mask>0]=core.PALETTE[oid]
        expected=image.copy()
        expected[active]=cv2.addWeighted(image[active],.5,colors[active],.5,0)
        np.testing.assert_array_equal(sammie.draw_masks(image,masks),expected)

    def test_checkerboard_pixels_match_original(self):
        rgb=np.full((64,96,3),127,np.uint8); alpha=np.arange(64*96,dtype=np.uint8).reshape(64,96)
        grid=(np.arange(64)[:,None]//8+np.arange(96)[None,:]//8)%2
        bg=np.repeat(np.where(grid[...,None],160,104).astype(np.uint8),3,axis=2)
        a=alpha.astype(np.float32)/255
        np.testing.assert_array_equal(sammie._composite_bgcolor(rgb,alpha,(0,0,0),'checkerboard'),cv2.blendLinear(rgb,bg,a,1-a))

    def test_import_failure_and_cancel_leave_previous_frames(self):
        from sammie.ingest import prepare_images
        broken=self.root/'bad.png';broken.write_bytes(b'broken')
        old=(self.source/'frames/00000.png').read_bytes()
        for files,cancel in [([self.source/'frames/00000.png',broken],lambda:False),([self.source/'frames/00000.png'],lambda:True)]:
            with self.assertRaises((ValueError,InterruptedError)):
                prepare_images(files,self.root,cancel=cancel)
            self.assertFalse(list(self.root.glob('.sammie-ingest-*')))
            self.assertEqual((self.source/'frames/00000.png').read_bytes(),old)
        self.assertFalse(any(t.name.startswith('sammie-import') for t in threading.enumerate()))

    def test_jpeg_normalization_dimensions_and_bounded_queue(self):
        from sammie.ingest import prepare_images,discard,queue_limit
        a=self.root/'image.jpeg';cv2.imwrite(str(a),np.full((64,96,3),120,np.uint8))
        stage,info=prepare_images([a],self.root)
        self.assertEqual(info['frame_format'],'jpg')
        self.assertEqual((stage/'frames/00000.jpg').read_bytes(),a.read_bytes())
        discard(stage)
        self.assertLessEqual(queue_limit(4608,3164,128*1024**2)*4608*3164*3,128*1024**2)
        self.assertEqual(queue_limit(4608,3164,0),1)

    def test_image_sequence_does_not_mix_another_prefix(self):
        for name in ('shot0001.png','shot0002.png','shot_extra0003.png'):
            cv2.imwrite(str(self.root/name),np.zeros((8,8,3),np.uint8))
        found,files,_=sammie.detect_image_sequence(str(self.root/'shot0001.png'))
        self.assertTrue(found)
        self.assertEqual([Path(p).name for p in files],['shot0001.png','shot0002.png'])

    def test_memory_preflight_explains_large_buffer_requirements(self):
        from sammie.performance import require_host_memory
        with patch('sammie.performance.available_memory',return_value=1024**3):
            require_host_memory(128*1024**2,'Test')
            with self.assertRaisesRegex(MemoryError,'Shorten the In/Out range'):
                require_host_memory(1024**3,'Test')

    def test_drawn_points_never_change_source_pixels(self):
        image=np.zeros((64,96,3),np.uint8);image.setflags(write=False)
        result=sammie.draw_points(image,0,self.points)
        self.assertTrue(np.any(result));self.assertFalse(np.any(image))

    def test_video_direct_bgr_is_pixel_identical_with_rate_and_timestamps(self):
        from sammie.ingest import prepare_video,discard
        path=self.root/'test.mkv'
        with av.open(str(path),'w') as out:
            stream=out.add_stream('ffv1',rate=Fraction(24000,1001));stream.width=96;stream.height=64;stream.pix_fmt='yuv420p'
            for i in range(3):
                rgb=np.random.default_rng(i).integers(0,256,(64,96,3),dtype=np.uint8)
                for p in stream.encode(av.VideoFrame.from_ndarray(rgb,format='rgb24')):out.mux(p)
            for p in stream.encode():out.mux(p)
        stage,info=prepare_video(path,self.root)
        with av.open(str(path)) as c:
            st=c.streams.video[0]
            for i,f in enumerate(c.decode(st)):
                old=f.reformat(format='rgb24',src_colorspace=int(st.codec_context.colorspace),dst_colorspace=1,
                    src_color_range=int(st.codec_context.color_range),dst_color_range=2).to_ndarray()
                np.testing.assert_array_equal(cv2.imread(str(stage/'frames'/f'{i:05d}.png')),cv2.cvtColor(old,cv2.COLOR_RGB2BGR))
        self.assertEqual(info['fps_rational'],'24000/1001')
        self.assertEqual(len(json.loads((stage/'source_timing.json').read_text())['frames']),3)
        discard(stage)

    def test_background_io_ticks_main_thread_and_cleans_up_on_error(self):
        from sammie.background_io import run_io, report
        ticks=[];timer=QTimer();timer.timeout.connect(lambda:ticks.append(time.monotonic()));timer.start(10)
        thread=[]
        def action(progress,cancel):
            thread.append(threading.get_ident());report('Opening test data')
            time.sleep(.18)
            raise ValueError('expected failure')
        try:
            with self.assertRaisesRegex(ValueError,'expected failure'):run_io(None,'Test operation',action)
        finally:timer.stop()
        self.assertNotEqual(thread[0],threading.get_ident());self.assertGreater(len(ticks),5)

    def test_background_cancel_waits_for_cleanup_without_blocking_events(self):
        from sammie.background_io import run_io
        cleaned=[]
        def cancel():
            for w in self.app.topLevelWidgets():
                if isinstance(w,QProgressDialog) and w.windowTitle()=='Cancel test': w.canceled.emit()
        QTimer.singleShot(40,cancel)
        def action(progress, cancelled):
            deadline=time.monotonic()+3
            try:
                while not cancelled() and time.monotonic()<deadline:time.sleep(.01)
                if cancelled():raise InterruptedError('cancelled')
                raise AssertionError('cancel not received')
            finally:cleaned.append(True)
        with self.assertRaises(InterruptedError):run_io(None,'Cancel test',action)
        self.assertEqual(cleaned,[True])

    def test_working_copy_preserves_sam_proxies_but_portable_does_not(self):
        import zipfile
        proxy=self.source/'sam3_proxy/cache';proxy.mkdir(parents=True)
        (proxy/'00000.png').write_bytes(b'prepared')
        descriptor=projects.create_project(self.source,self.root/'copy.sammieproj')
        data,_=projects.read_project(descriptor)
        self.assertEqual((data/'sam3_proxy/cache/00000.png').read_bytes(),b'prepared')
        archive=self.root/'copy.sammie';projects.pack_project(data,archive)
        with zipfile.ZipFile(archive) as z:self.assertFalse(any(n.startswith('sam3_proxy/') for n in z.namelist()))

    def test_render_cache_keys_reject_replaced_masks_and_derived_views(self):
        from sammie.preview_worker import cache_key
        request=dict(root=str(self.source),frame=0,points=self.points,session=asdict(self.manager.session_settings),
                     view=dict(view_mode='Segmentation-Matte'),generation=0)
        before=cache_key(request)
        cv2.imwrite(str(self.source/'masks/00000/0.png'),np.zeros((64,96),np.uint8))
        self.assertNotEqual(cache_key(request),before)
        request['view']['view_mode']='Matting-Matte'
        self.assertIsNone(cache_key(request))

    def test_sam2_cache_preserves_exact_float32_preprocessing(self):
        import torch
        from sam2.utils.misc import _load_img_as_tensor
        from sammie.backends.sam2_frames import BoundedSam2Frames
        paths=[str(self.source/'frames'/f'{i:05d}.png') for i in (0,1)]
        mean=torch.tensor([.485,.456,.406])[:,None,None]
        std=torch.tensor([.229,.224,.225])[:,None,None]
        cache=BoundedSam2Frames(paths,128,True,mean,std,torch.device('cpu'))
        for i in (0,1,0):
            expected,h,w=_load_img_as_tensor(paths[i],128);expected-=mean;expected/=std
            torch.testing.assert_close(cache[i],expected,rtol=0,atol=0)
            self.assertEqual(cache[i].dtype,torch.float32)
        self.assertLessEqual(len(cache.cache),4)

    def test_progress_adapter_runs_without_constructing_widgets(self):
        from sammie.background_io import run_io,progress_dialog
        def action(progress,cancel):
            dialog=progress_dialog('Decode', 'Cancel',0,0,None)
            self.assertNotIsInstance(dialog,QProgressDialog)
            dialog.setMinimumDuration(0);dialog.setAutoReset(False);dialog.setRange(0,3)
            dialog.setLabelText('Inference');dialog.setValue(1)
            self.assertFalse(dialog.wasCanceled());dialog.close()
        run_io(None,'Adapter test',action)

    def test_model_load_is_background_and_failure_unloads(self):
        import torch
        import sammie_main
        with patch.object(sammie_main,'initialize_settings',return_value=self.manager), \
             patch.object(sammie_main.MainWindow,'_deferred_init'), \
             patch.object(sammie_main.MainWindow,'_setup_console_redirect'), \
             patch.object(core.DeviceManager,'_device',torch.device('cpu')), \
             patch('sammie.model_downloader.ensure_models',return_value=True):
            window=sammie_main.MainWindow()
            ticks=[];timer=QTimer();timer.timeout.connect(lambda:ticks.append(1));timer.start(10)
            def fail(*args,**kwargs):
                self.assertNotEqual(threading.current_thread(),threading.main_thread())
                time.sleep(.18)
                raise ValueError('checkpoint failed')
            try:
                with patch.object(window.sam_manager,'load_segmentation_model',side_effect=fail), \
                     patch.object(window.sam_manager,'unload_segmentation_model') as unload:
                    with self.assertRaisesRegex(ValueError,'checkpoint failed'):window._load_model_responsive('Base')
                    unload.assert_called()
                self.assertGreater(len(ticks),5)
                self.assertFalse(window._compute_busy)
            finally:timer.stop();window.close()

    def test_tracking_worker_preserves_reverse_order_and_cancel_cleanup(self):
        import torch
        from sammie.backends.tracking import TrackingMixin
        class Backend(TrackingMixin):
            def __init__(self): self.closed=False;self.written=[]
            def iter_masks(self,**kwargs):
                try:
                    for frame in (1,0):
                        time.sleep(.02)
                        yield frame,{0:np.ones((64,96),np.uint8)*255}
                finally:self.closed=True
            def write_masks(self,frame,masks): self.written.append(frame)
        backend=Backend()
        with patch.object(core.DeviceManager,'_device',torch.device('cpu')):
            self.assertEqual(backend._propagate(None,1,1,reverse=True),(0,False))
        self.assertEqual(backend.written,[1,0]);self.assertTrue(backend.closed)

    def test_roi_postprocessing_empty_and_cancel(self):
        from sammie.background_io import run_io
        with patch.object(core,'apply_mask_postprocessing',return_value=np.zeros((64,96),np.uint8)):
            self.assertIsNone(core.compute_mask_bounding_box(range(2),[0]))
        with patch('sammie.background_io.check_cancelled',side_effect=InterruptedError('cancel')):
            with self.assertRaises(InterruptedError):core.compute_mask_bounding_box(range(2),[0])
        observed=[]
        with patch('sammie.background_io.report',side_effect=lambda *args:observed.append(args)):
            run_io(None,'Finding test ROI',lambda progress,cancel:core.compute_mask_bounding_box(range(2),[0]))
        self.assertIn('scanning frame 0',observed[0][0])

    def test_single_image_export_renders_in_worker_and_preserves_rgba(self):
        import torch
        from PIL import Image
        from sammie.export_image_dialog import ImageExportDialog
        dialog=ImageExportDialog(frame_number=1)
        rgba=np.full((64,96,4),97,np.uint8);rgba[...,3]=128
        def render(*args,**kwargs):
            self.assertNotEqual(threading.current_thread(),threading.main_thread())
            self.assertEqual(args[0],1)
            return rgba
        try:
            with patch.object(core.DeviceManager,'_device',torch.device('cpu')),patch.object(sammie,'update_image',side_effect=render):
                dialog._export_image(str(self.root/'export.png'))
            with Image.open(self.root/'export.png') as saved:np.testing.assert_array_equal(np.asarray(saved),rgba)
        finally:dialog.close()

    def test_matting_refinement_handoff_keeps_sam_offloaded_and_restores_on_failure(self):
        import torch
        import sammie_main
        from sammie.background_io import run_io
        from types import SimpleNamespace
        with patch.object(sammie_main,'initialize_settings',return_value=self.manager), \
             patch.object(sammie_main.MainWindow,'_deferred_init'), \
             patch.object(sammie_main.MainWindow,'_setup_console_redirect'), \
             patch.object(core.DeviceManager,'_device',torch.device('cpu')):
            window=sammie_main.MainWindow()
            window.point_manager.points=list(self.points)
            self.manager.set_session_setting('matany_model','VideoMaMa')
            self.manager.set_session_setting('vitmatte_enabled',True)
            events=[]
            def event(name,value=True):
                def action(*args,**kwargs):
                    if name != 'vitmatte':self.assertNotEqual(threading.current_thread(),threading.main_thread())
                    events.append(name)
                    return value
                return action
            mat=Mock(propagated=False)
            mat.load_matting_model.side_effect=event('load')
            mat.run_matting.side_effect=event('matting')
            mat.unload_matting_model.side_effect=event('unload')
            try:
                with patch('sammie.model_downloader.ensure_models',return_value=True), \
                     patch('sammie.background_io.run_compute',side_effect=run_io), \
                     patch.object(core.DeviceManager,'get_device',return_value=SimpleNamespace(type='cuda')), \
                     patch.object(sammie_main.matting,'create_matting_manager',return_value=mat), \
                     patch.object(window.sam_manager,'offload_model_to_cpu',side_effect=event('offload')), \
                     patch.object(window.sam_manager,'load_model_to_device',side_effect=event('restore')), \
                     patch.object(window,'run_vitmatte',side_effect=event('vitmatte')), \
                     patch.object(window,'_update_current_frame_display'), \
                     patch.object(sammie_main,'show_message_dialog'):
                    window.run_matting()
                    self.assertEqual(events,['offload','load','matting','unload','vitmatte','restore'])
                    events.clear();mat.run_matting.side_effect=RuntimeError('model failed')
                    window.run_matting()
                    self.assertEqual(events,['offload','load','unload','restore'])
            finally:window.close()

    @unittest.skipUnless(os.name=='nt','Windows venv process-tree regression')
    def test_worker_stop_terminates_owned_launcher_and_children(self):
        import subprocess,sys
        try:import psutil
        except ImportError:self.skipTest('psutil needed to verify child process exit')
        from sammie.processes import stop_worker_tree
        record=self.root/'children.json'
        program="import subprocess,sys,json,os,time; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); Path(sys.argv[1]).write_text(json.dumps([os.getpid(),p.pid])); time.sleep(60)"
        child=subprocess.Popen([sys.executable,'-c',program,str(record)],creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            deadline=time.monotonic()+10
            while not record.exists() and time.monotonic()<deadline:time.sleep(.02)
            self.assertTrue(record.exists())
            pids=json.loads(record.read_text())
            self.assertTrue(all(psutil.pid_exists(pid) for pid in pids))
            stop_worker_tree(child.pid)
            child.wait(timeout=10)
            deadline=time.monotonic()+5
            while any(psutil.pid_exists(pid) for pid in pids) and time.monotonic()<deadline:time.sleep(.02)
            self.assertFalse(any(psutil.pid_exists(pid) for pid in pids))
        finally:
            if child.poll() is None:stop_worker_tree(child.pid)

    def test_real_preview_process_latest_seek_and_pixel_equivalence(self):
        from sammie.preview import PreviewController
        controller=PreviewController();results=[];errors=[];ticks=[]
        controller.ready.connect(lambda frame,pixmap:results.append((frame,pixmap)))
        controller.failed.connect(errors.append)
        timer=QTimer();timer.timeout.connect(lambda:ticks.append(1));timer.start(10)
        def wait(count):
            deadline=time.monotonic()+45
            while len(results)<count and not errors and time.monotonic()<deadline:
                self.app.processEvents();time.sleep(.002)
            self.assertFalse(errors,errors);self.assertEqual(len(results),count)
        try:
            controller.request(0,dict(view_mode='None'),self.points,self.manager)
            controller.request(1,dict(view_mode='None'),self.points,self.manager)
            wait(1);self.assertEqual(results[0][0],1);self.assertGreater(len(ticks),5)
            self.assertEqual(results[0][1].toImage().pixelColor(10,10).red(),140)
            controller.request(0,dict(view_mode='Segmentation-Matte',antialias=False),self.points,self.manager)
            wait(2);self.assertEqual(results[-1][1].toImage().pixelColor(10,10).red(),255)
            cv2.imwrite(str(self.source/'masks/00000/0.png'),np.zeros((64,96),np.uint8))
            controller.request(0,dict(view_mode='Segmentation-Matte',antialias=False),self.points,self.manager)
            wait(3);self.assertEqual(results[-1][1].toImage().pixelColor(10,10).red(),0)
        finally:
            timer.stop();controller.close()
        self.assertFalse(controller.busy)


if __name__=='__main__':unittest.main()
