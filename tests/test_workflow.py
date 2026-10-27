import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from dataclasses import asdict
from fractions import Fraction
import json
from pathlib import Path
import shutil
import time
import unittest
from unittest.mock import patch
import uuid
import zipfile
import cv2
import numpy as np
import OpenEXR
from PySide6.QtWidgets import QApplication,QDialog
from PySide6.QtCore import QTimer
from sammie import projects,core
from sammie.settings_manager import SessionSettings
from sammie.workflow.jobs import make_job,QueueStore,RangeTransaction,recover,signature,frame_range
from sammie.workflow.worker import execute,output_signature


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
    def setUp(self):
        self.root=Path('.test-output').resolve()/('workflow-unit-'+uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.source=self.root/'session'
        (self.source/'frames').mkdir(parents=True)
        (self.source/'matting').mkdir()
        self.session=asdict(SessionSettings(video_width=96,video_height=64,total_frames=3,
                    video_fps=24000/1001,video_fps_rational='24000/1001',sequence_start_frame=1001,
                    matany_model='VideoMaMa',vitmatte_enabled=False))
        self.points=[dict(frame=0,object_id=0,x=30,y=20,positive=True)]
        for frame in range(3):
            cv2.imwrite(str(self.source/'frames'/f'{frame:05d}.png'),np.full((64,96,3),127,np.uint8))
            p=self.source/'matting'/f'{frame:05d}';p.mkdir()
            cv2.imwrite(str(p/'0.png'),np.full((64,96),128,np.uint8))
        projects.save_metadata(self.source,self.session,self.points)
        self.descriptor=projects.create_project(self.source,self.root/'shot.sammieproj')
        self.data,_=projects.read_project(self.descriptor)
    def tearDown(self):
        assert self.root.parent==Path('.test-output').resolve() and self.root.name.startswith('workflow-unit-')
        shutil.rmtree(self.root)

    def test_incremental_save_does_not_scan_or_rewrite_media(self):
        p=self.data/'frames/00000.png'; before=(p.read_bytes(),p.stat().st_mtime_ns)
        with patch('sammie.projects.session_files',side_effect=AssertionError('must not scan media')):
            projects.save_metadata(self.data,dict(self.session,current_frame=2),self.points,self.descriptor)
        self.assertEqual((p.read_bytes(),p.stat().st_mtime_ns),before)
        self.assertEqual(json.loads((self.data/'session_settings.conf').read_text())['current_frame'],2)

    def test_workspace_layout_and_project_portability(self):
        workspace=projects.set_workspace(self.root/'school')
        self.assertTrue(all((workspace/n).is_dir() for n in ('scenes','footage','exports','nuke')))
        target=self.root/'moved';target.mkdir()
        shutil.copy2(self.descriptor,target/self.descriptor.name)
        shutil.copytree(self.data,target/self.data.name)
        root,_=projects.read_project(target/self.descriptor.name)
        self.assertTrue((root/'frames/00000.png').is_file())

    def test_lock_excludes_second_writer_and_releases(self):
        with projects.ProjectLock(self.data):
            with self.assertRaises(RuntimeError):projects.ProjectLock(self.data).acquire()
        with projects.ProjectLock(self.data):pass

    def test_save_as_cancel_and_existing_name_never_overwrite(self):
        with self.assertRaises(FileExistsError):projects.create_project(self.source,self.descriptor)
        with self.assertRaises(InterruptedError):
            projects.create_project(self.source,self.root/'cancel.sammieproj',cancel=lambda:True)
        self.assertFalse((self.root/'cancel.sammieproj').exists())
        self.assertFalse(list(self.root.glob('.sammie-create-*')))

    def test_portable_pack_round_trip_and_cancel_keeps_previous_archive(self):
        archive=self.root/'packed.sammie';archive.write_bytes(b'previous')
        with self.assertRaises(InterruptedError):projects.pack_project(self.data,archive,cancel=lambda:True)
        self.assertEqual(archive.read_bytes(),b'previous')
        projects.pack_project(self.data,archive)
        restored=self.root/'restored'
        projects.unpack_project(archive,restored)
        self.assertEqual((restored/'frames/00000.png').read_bytes(),(self.data/'frames/00000.png').read_bytes())
        self.assertFalse((restored/projects.MARKER).exists())
        with self.assertRaises(ValueError):projects.unpack_project(archive,self.data)

    def test_legacy_archive_rejects_traversal_without_removing_session(self):
        archive=self.root/'bad.sammie'
        with zipfile.ZipFile(archive,'w') as z:z.writestr('../escape.txt','bad')
        with self.assertRaises(ValueError):projects.unpack_project(archive,self.source)
        self.assertTrue((self.source/'frames/00000.png').exists())
        self.assertFalse((self.root/'escape.txt').exists())

    def test_range_plan_and_full_shot_geometry(self):
        job=make_job(self.descriptor,['vitmatte','depth'],1,1,1)
        self.assertEqual(job['requested'],[1,1]);self.assertEqual(job['effective'],[0,2])
        self.assertEqual(frame_range(job,'depth'),(0,2))
        with self.assertRaises(ValueError):make_job(self.descriptor,['matting'],2,1)

    def test_range_rollback_and_recovery_preserve_outside_outputs(self):
        path=self.data/'matting/00001/0.png'; old=path.read_bytes()
        outside=self.data/'matting/00000/0.png'; untouched=outside.stat().st_mtime_ns
        with self.assertRaises(RuntimeError):
            with RangeTransaction(self.data,'matting',1,1):
                path.write_bytes(b'partial');raise RuntimeError('cancel')
        self.assertEqual(path.read_bytes(),old);self.assertEqual(outside.stat().st_mtime_ns,untouched)
        transaction=RangeTransaction(self.data,'matting',1,1);transaction.__enter__()
        path.write_bytes(b'killed worker')
        recover(self.data)
        self.assertEqual(path.read_bytes(),old)

    def test_completed_stage_reuse_checks_both_inputs_and_outputs(self):
        job=make_job(self.descriptor,['matting'],1,1,0); calls=[]
        def runner(job,stage,progress):
            calls.append(stage)
            cv2.imwrite(str(self.data/'matting/00001/0.png'),np.full((64,96),150,np.uint8))
        result=execute(job,'matting',runner)
        job['completed']['matting']=result['receipt']
        self.assertTrue(execute(job,'matting',runner)['cached'])
        (self.data/'matting/00001/0.png').unlink()
        self.assertFalse(execute(job,'matting',runner)['cached'])
        self.assertEqual(len(calls),2)
        job['session']['matany_res']=256
        self.assertFalse(execute(job,'matting',runner)['cached'])

    def test_queue_persists_interrupted_status_and_receipts(self):
        store=QueueStore(self.root/'queue.json');job=make_job(self.descriptor,['nuke'])
        store.add(job);job.update(status='running',completed={'matting':{'input':'a','output':'b'}});store.save()
        restored=QueueStore(store.path)
        self.assertEqual(restored.jobs[0]['status'],'interrupted')
        restored.resume();self.assertEqual(restored.jobs[0]['status'],'pending')
        self.assertEqual(restored.jobs[0]['completed'],job['completed'])

    def test_changed_project_refuses_old_queued_configuration(self):
        job=make_job(self.descriptor,['matting'])
        projects.save_metadata(self.data,dict(self.session,matany_res=512),self.points)
        with self.assertRaisesRegex(ValueError,'changed after queueing'):
            execute(job,'matting',lambda *args:self.fail('must not run stale settings'))

    def test_failed_unpack_and_failed_rollback_keep_last_good_session(self):
        archive=self.root/'packed.sammie';projects.pack_project(self.data,archive)
        rename=Path.rename
        def fail(path,target):
            if path.name.startswith(('.sammie-unpack-','.sammie-old-')):
                raise OSError('simulated disk failure')
            return rename(path,target)
        with patch.object(Path,'rename',fail),self.assertRaises(OSError):
            projects.unpack_project(archive,self.source)
        backups=list(self.root.glob('.sammie-old-*'))
        self.assertEqual(len(backups),1)
        self.assertEqual((backups[0]/'frames/00000.png').read_bytes(),(self.data/'frames/00000.png').read_bytes())

    def test_nuke_interrupted_staging_recovery_keeps_completed_delivery(self):
        from sammie.workflow.nuke import deliver
        job=make_job(self.descriptor,['nuke']);job['delivery_parent']=str(self.root/'delivery')
        completed=deliver(job)
        token=uuid.uuid4().hex;stage=self.root/'delivery'/('.sammie-delivery-'+token)
        stage.mkdir();(stage/'partial.exr').write_bytes(b'incomplete')
        projects.atomic_json(stage/'.sammie-delivery-owner.json',dict(project=str(self.data)))
        projects.atomic_json(self.data/'.workflow/delivery-staging.json',dict(parent=str(stage.parent),token=token))
        recover(self.data)
        self.assertFalse(stage.exists())
        self.assertTrue(Path(completed['script']).exists())

    def test_mainwindow_fast_save_and_media_detach_keep_saved_frames(self):
        import sammie_main
        from sammie.settings_manager import initialize_settings
        manager=initialize_settings(str(self.source));manager.app_settings_file=str(self.root/'app.conf')
        overrides={k:str(self.source/v) for k,v in [('frames_dir','frames'),('mask_dir','masks'),
                   ('matting_dir','matting'),('backup_dir','masks_backup'),('removal_dir','removal')]}
        overrides['temp_dir']=str(self.source)
        with patch.multiple(core,**overrides),patch.multiple(core.VideoInfo,total_frames=3,width=96,height=64),\
             patch.object(sammie_main,'initialize_settings',return_value=manager),\
             patch.object(sammie_main.MainWindow,'_deferred_init'),patch.object(sammie_main.MainWindow,'_setup_console_redirect'):
            window=sammie_main.MainWindow()
            try:
                window.point_manager.points=list(self.points)
                descriptor=self.root/'ui-shot.sammieproj'
                with patch('sammie.project_ui.QFileDialog.getSaveFileName',return_value=(str(descriptor),'')):
                    self.assertTrue(window.projects.save())
                data,_=projects.read_project(descriptor)
                saved=(data/'frames/00000.png').read_bytes()
                self.assertEqual(Path(core.frames_dir),data/'frames')
                with patch('sammie.projects.session_files',side_effect=AssertionError('save scanned media')):
                    self.assertTrue(window.projects.save())
                self.assertEqual(json.loads((data/'points.json').read_text()),self.points)
                bad=self.root/'invalid.sammie'
                with zipfile.ZipFile(bad,'w') as archive:archive.writestr('../escape.txt','bad')
                with self.assertRaises(ValueError):window.projects.open(bad)
                self.assertEqual(window.projects.descriptor,descriptor)
                self.assertEqual(Path(core.temp_dir),data)
                window.projects.detach()
                self.assertEqual(Path(core.temp_dir),self.source)
                self.assertEqual((data/'frames/00000.png').read_bytes(),saved)
                with projects.ProjectLock(data):pass
            finally:window.close()

    def test_queue_stop_kills_worker_and_restores_interrupted_range(self):
        from sammie.workflow.ui import QueueDialog
        from PySide6.QtCore import QProcess
        store=QueueStore(self.root/'cancel-queue.json')
        job=make_job(self.descriptor,['matting'],1,1,0);store.add(job)
        output=self.data/'matting/00001/0.png';old=output.read_bytes()
        script=self.root/'slow_worker.py'
        script.write_text('from sammie.workflow.jobs import RangeTransaction\nfrom pathlib import Path\nimport time\n'
                          +f'with RangeTransaction({str(self.data)!r}, "matting", 1, 1):\n'
                          +f'    Path({str(output)!r}).write_bytes(b"partial")\n    time.sleep(90)\n')
        class SlowProcess(QProcess):
            def start(self,program,args):super().start(program,['-u',str(script)])
        dialog=QueueDialog(None,store);cancelled=[]
        def poll():
            if not cancelled and output.read_bytes()==b'partial':
                cancelled.append(True);dialog.cancel()
            if not dialog.running:dialog.accept()
        timer=QTimer();timer.timeout.connect(poll);timer.start(50)
        timeout=QTimer();timeout.setSingleShot(True);timeout.timeout.connect(dialog.cancel);timeout.start(30000)
        try:
            with patch('sammie.workflow.ui.QProcess',SlowProcess):
                QTimer.singleShot(0,dialog.begin);dialog.exec()
        finally:timer.stop();timeout.stop()
        self.assertTrue(cancelled)
        self.assertEqual(job['status'],'cancelled')
        self.assertEqual(output.read_bytes(),old)
        self.assertFalse((self.data/'.workflow/transaction').exists())

    def test_update_handoff_waits_for_app_and_requests_restart(self):
        from sammie.school_updates import start_update
        script=self.root/'scripts/update_school.ps1';script.parent.mkdir();script.write_text('# fixture')
        with patch('sammie.school_updates.subprocess.Popen') as launch:
            log,_=start_update(self.root,pid=4321)
        command=launch.call_args.args[0]
        self.assertEqual(command[command.index('-WaitForPid')+1],'4321')
        self.assertIn('-Restart',command)
        self.assertTrue(log.is_file())

    def test_nuke_delivery_has_float_alpha_linear_rgb_and_original_numbering(self):
        from sammie.workflow.nuke import deliver
        job=make_job(self.descriptor,['nuke'],0,2,0);job['delivery_parent']=str(self.root/'delivery')
        result=deliver(job)
        with OpenEXR.File(str(Path(result['directory'])/'frames/sammie.1001.exr'),separate_channels=True) as f:
            self.assertEqual(f.header()['framesPerSecond'],Fraction(24000,1001))
            self.assertAlmostEqual(float(f.channels()['A'].pixels[0,0]),128/255,places=6)
            self.assertTrue(.20<float(f.channels()['R'].pixels[0,0])<.23)
        text=Path(result['script']).read_text()
        self.assertIn('first_frame 1001',text);self.assertIn('last_frame 1003',text)
        self.assertNotIn(str(self.root),text);self.assertIn('raw true',text)

    def test_real_queue_process_completes_two_cpu_deliveries(self):
        from sammie.workflow.ui import QueueDialog
        store=QueueStore(self.root/'queue.json')
        for _ in range(2):
            job=make_job(self.descriptor,['nuke']);job['delivery_parent']=str(self.root/'deliveries');store.add(job)
        dialog=QueueDialog(None,store);dialog.auto_close=True
        poll=QTimer();poll.timeout.connect(lambda: dialog.accept() if not dialog.running else None);poll.start(100)
        timeout=QTimer();timeout.setSingleShot(True);timeout.timeout.connect(dialog.cancel);timeout.start(60000)
        QTimer.singleShot(0,dialog.begin)
        try:dialog.exec()
        finally:timeout.stop();poll.stop()
        self.assertEqual([j['status'] for j in store.jobs],['complete','complete'],str(store.jobs))
        for job in store.jobs:self.assertTrue(Path(job['delivery']['script']).is_file())

    def test_update_release_validation_and_installed_manifest(self):
        from sammie.school_updates import installed_version,latest_release
        projects.atomic_json(self.root/'SCHOOL_BUILD.json',dict(commit='a'*40))
        self.assertEqual(installed_version(self.root),('a'*40,False))
        import io
        value=dict(draft=False,prerelease=False,html_url='https://example.com/releases/bad')
        with self.assertRaises(ValueError):latest_release(lambda *args,**kwargs:io.BytesIO(json.dumps(value).encode()))


if __name__=='__main__':unittest.main()
