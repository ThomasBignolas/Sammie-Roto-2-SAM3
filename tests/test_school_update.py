import hashlib
import json
from pathlib import Path
import shutil
import unittest
import uuid
import zipfile
from scripts.update_school import apply_archive,validate_archive,target_path


class SchoolUpdateTests(unittest.TestCase):
    def setUp(self):
        self.root=Path('.test-output').resolve()/('school-update-unit-'+uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.app=self.root/'installed';self.app.mkdir()
        self.base={'install_school.bat':b'installer','scripts/install_school.ps1':b'installer',
                   'run_sam3.bat':b'launcher','sammie/app.py':b'old','obsolete.py':b'old'}
        self.manifest=self.make_manifest(self.base,'a'*40)
        for name,data in self.base.items():
            p=self.app/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        (self.app/'SCHOOL_BUILD.json').write_text(json.dumps(self.manifest))
        for name in ('checkpoints/sam3.pt','temp/session_settings.conf','temp/source_exr/00000.exr',
                     'shot.sammie','sammie_settings.conf','.venv/keep.txt'):
            p=self.app/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'precious')

    def tearDown(self):
        assert self.root.parent==Path('.test-output').resolve() and self.root.name.startswith('school-update-unit-')
        shutil.rmtree(self.root)

    def make_manifest(self,files,commit):
        return {'commit':commit,'sha256':{n:hashlib.sha256(b).hexdigest() for n,b in files.items()}}

    def archive(self,files=None):
        files=files or dict(self.base,**{'sammie/app.py':b'new','new.py':b'new'})
        files={n:b for n,b in files.items() if n!='obsolete.py'}
        archive=self.root/'release.zip'
        with zipfile.ZipFile(archive,'w') as z:
            for name,data in files.items(): z.writestr('Sammie-Roto-School/'+name,data)
            z.writestr('Sammie-Roto-School/SCHOOL_BUILD.json',json.dumps(self.make_manifest(files,'b'*40)))
        return archive,hashlib.sha256(archive.read_bytes()).hexdigest()

    def test_updates_only_packaged_files_preserves_data_and_is_repeatable(self):
        archive,sha=self.archive()
        report=apply_archive(self.app,archive,sha)
        self.assertEqual((self.app/'sammie/app.py').read_bytes(),b'new')
        self.assertFalse((self.app/'obsolete.py').exists())
        self.assertEqual((Path(report['backup'])/'sammie/app.py').read_bytes(),b'old')
        for name in ('checkpoints/sam3.pt','temp/source_exr/00000.exr','shot.sammie','sammie_settings.conf','.venv/keep.txt'):
            self.assertEqual((self.app/name).read_bytes(),b'precious')
        self.assertEqual(apply_archive(self.app,archive,sha)['changed'],0)

    def test_refuses_local_source_edits_and_untracked_collisions_before_changes(self):
        archive,sha=self.archive()
        for name in ('sammie/app.py','new.py'):
            p=self.app/name;p.write_bytes(b'local-change')
            with self.assertRaisesRegex(ValueError,'Local source edit'):
                apply_archive(self.app,archive,sha)
            self.assertTrue((self.app/'obsolete.py').exists())
            if name in self.base: p.write_bytes(self.base[name])
            else: p.unlink()

    def test_failure_rolls_back_completed_replacements(self):
        archive,sha=self.archive()
        import os
        calls=[]
        def broken(source,target):
            calls.append(str(target))
            if len(calls)==2: raise OSError('locked file')
            os.replace(source,target)
        with self.assertRaisesRegex(RuntimeError,'restored'):
            apply_archive(self.app,archive,sha,replace=broken)
        for name,data in self.base.items(): self.assertEqual((self.app/name).read_bytes(),data)
        self.assertFalse((self.app/'new.py').exists())
        self.assertEqual(json.loads((self.app/'SCHOOL_BUILD.json').read_text()),self.manifest)

    def test_checksum_failure_and_protected_paths_cannot_touch_models(self):
        archive,sha=self.archive()
        with self.assertRaisesRegex(ValueError,'checksum mismatch'):
            apply_archive(self.app,archive,'0'*64)
        for name in ('../outside.py','.git/config','.venv/keep.txt','checkpoints/sam3.pt',
                     'temp/source_exr/00000.exr','shot.sammie','C:/outside.py','sammie_settings.conf'):
            with self.assertRaises(ValueError): target_path(self.app,name)
        archive,sha=self.archive(dict(self.base,**{'checkpoints/sam3.pt':b'bad'}))
        with self.assertRaises(ValueError): validate_archive(archive,sha)
        self.assertEqual((self.app/'checkpoints/sam3.pt').read_bytes(),b'precious')

    def test_rejects_archive_manifest_mismatch_and_windows_case_collision(self):
        archive,sha=self.archive()
        with zipfile.ZipFile(archive,'a') as z: z.writestr('Sammie-Roto-School/extra.py','not in manifest')
        with self.assertRaisesRegex(ValueError,'manifest disagree'):
            validate_archive(archive,hashlib.sha256(archive.read_bytes()).hexdigest())
        archive,sha=self.archive(dict(self.base,**{'RUN_SAM3.BAT':b'collision'}))
        with self.assertRaisesRegex(ValueError,'duplicate'):
            validate_archive(archive,sha)


if __name__=='__main__': unittest.main()
