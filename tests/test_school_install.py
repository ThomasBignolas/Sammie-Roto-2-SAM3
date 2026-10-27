"""Installer integrity, resumable downloads and source-package boundaries."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest.mock import Mock, patch
import uuid
import zipfile

from sammie.model_downloader import DownloadSpec
from scripts.setup_school_models import download
from scripts.package_school import allowed, build, PREFIX, REQUIRED


class SchoolInstallTests(unittest.TestCase):
    def setUp(self):
        self.root = Path('.test-output').resolve() / ('school-unit-' + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.payload = b'official-test-model'
        self.spec = DownloadSpec('https://huggingface.co/example/model/resolve/pinned/model.pt',
                                 hashlib.md5(self.payload).hexdigest(), 'checkpoints')

    def tearDown(self):
        assert self.root.parent == Path('.test-output').resolve() and self.root.name.startswith('school-unit-')
        shutil.rmtree(self.root)

    def response(self, payload=None, status=200, headers=None):
        response = Mock(status_code=status, headers=headers or {})
        response.iter_content.return_value = [self.payload if payload is None else payload]
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        return response

    def test_download_then_verified_reuse_without_network(self):
        get = Mock(return_value=self.response())
        path = download(self.spec, self.root, get, progress=lambda _: None)
        self.assertEqual(path.read_bytes(), self.payload)
        download(self.spec, self.root, get, progress=lambda _: None)
        self.assertEqual(get.call_count, 1)
        self.assertFalse(path.with_suffix('.pt.part').exists())

    def test_resume_valid_range_and_handle_server_ignoring_range(self):
        for resume in (True, False):
            part = self.root / 'checkpoints/model.pt.part'
            part.parent.mkdir(exist_ok=True)
            part.write_bytes(self.payload[:4])
            response = self.response(self.payload[4:] if resume else self.payload,
                                     206 if resume else 200, {'Content-Range': f'bytes 4-{len(self.payload)-1}/{len(self.payload)}'})
            get = Mock(return_value=response)
            path = download(self.spec, self.root, get, progress=lambda _: None)
            self.assertEqual(path.read_bytes(), self.payload)
            self.assertEqual(get.call_args.kwargs['headers']['Range'], 'bytes=4-')
            path.unlink()

    def test_checksum_failure_keeps_old_file_and_rejects_corrupt_download(self):
        destination = self.root / 'checkpoints/model.pt'
        destination.parent.mkdir()
        destination.write_bytes(b'existing-file')
        with self.assertRaisesRegex(RuntimeError, 'Checksum mismatch'):
            download(self.spec, self.root, Mock(return_value=self.response(b'corrupt')), progress=lambda _: None)
        self.assertEqual(destination.read_bytes(), b'existing-file')
        self.assertFalse(destination.with_suffix('.pt.part').exists())

    def test_http_error_never_installs_error_page(self):
        with self.assertRaisesRegex(RuntimeError, 'HTTP 403'):
            download(self.spec, self.root, Mock(return_value=self.response(b'Forbidden', 403)), progress=lambda _: None)
        self.assertFalse((self.root / self.spec.final_path).exists())

    def test_invalid_range_preserves_partial(self):
        part = self.root / 'checkpoints/model.pt.part'
        part.parent.mkdir()
        part.write_bytes(b'part')
        with self.assertRaisesRegex(RuntimeError, 'Invalid resume'):
            download(self.spec, self.root, Mock(return_value=self.response(status=206, headers={'Content-Range': 'bytes 0-5/6'})), progress=lambda _: None)
        self.assertEqual(part.read_bytes(), b'part')

    def test_cancellation_leaves_partial_and_no_destination(self):
        response = self.response()
        def interrupted():
            yield self.payload[:4]
            raise KeyboardInterrupt()
        response.iter_content.return_value = interrupted()
        with self.assertRaises(KeyboardInterrupt):
            download(self.spec, self.root, Mock(return_value=response), progress=lambda _: None)
        self.assertEqual((self.root / 'checkpoints/model.pt.part').read_bytes(), self.payload[:4])
        self.assertFalse((self.root / self.spec.final_path).exists())

    def test_credentials_only_sent_to_huggingface(self):
        for host in ('huggingface.co', 'github.com', 'huggingface.co.other.example'):
            spec = DownloadSpec(f'https://{host}/model.pt', self.spec.md5, host)
            get = Mock(return_value=self.response())
            download(spec, self.root, get, token='test-only-placeholder', progress=lambda _: None)
            self.assertEqual('Authorization' in get.call_args.kwargs['headers'], host == 'huggingface.co')

    def test_package_excludes_credentials_media_and_includes_configs(self):
        for name in ['.env', '.env.school', '.git/config', '.runtime/token', 'checkpoints/.cache/a.json',
                     'AlphaTest.nk', 'checkpoints/sam3.pt', 'shot.sammie', 'token', 'stored_tokens',
                     'shot.sammieproj','shot.sammie-data/frames/00000.png','sammie.workspace.json','.sammie/queue.json']:
            self.assertFalse(allowed(name), name)
        for name in REQUIRED:
            self.assertTrue(allowed(name), name)
        self.assertTrue(allowed('checkpoints/propainterx/memfof/config.json'))

    def test_exact_revision_package_has_manifest_and_repeatable_hash(self):
        original = io.BytesIO()
        with zipfile.ZipFile(original, 'w') as archive:
            for name in (*REQUIRED, 'AlphaTest.nk', '.env', '.runtime/private-data.json'):
                archive.writestr(name, 'fixture')
        with patch('scripts.package_school.subprocess.check_output', side_effect=['a'*40, original.getvalue()]):
            path, digest, count = build(self.root, output=self.root/'first.zip')
        with patch('scripts.package_school.subprocess.check_output', side_effect=['a'*40, original.getvalue()]):
            _, second_digest, _ = build(self.root, output=self.root/'second.zip')
        self.assertEqual(digest, second_digest)
        self.assertEqual(count, len(REQUIRED))
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read(PREFIX + 'SCHOOL_BUILD.json'))
            self.assertEqual(manifest['commit'], 'a'*40)
            self.assertEqual(set(manifest['sha256']), set(REQUIRED))
            for name, sha in manifest['sha256'].items():
                self.assertEqual(hashlib.sha256(archive.read(PREFIX+name)).hexdigest(), sha)

    @unittest.skipUnless(os.name == 'nt', 'Windows installer')
    def test_plan_needs_no_login_and_does_not_install(self):
        script = Path('scripts/install_school.ps1').resolve()
        result = subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(script),
                                 '-Plan', '-Models', 'OnDemand', '-NoLogin'], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('ViTMatte', result.stdout)
        self.assertNotIn('Download authorized', result.stdout)
        self.assertNotIn('Taille du sexe', result.stdout)

    def bootstrap(self, entries=None, bad_checksum=False):
        fixture = self.root / 'release.zip'
        with zipfile.ZipFile(fixture, 'w') as archive:
            if entries is None:
                entries = {'Sammie-Roto-School/scripts/install_school.ps1':
                           "param($Models,[switch]$NoLogin,[switch]$NoShortcut,[switch]$SkipSamCheckpoints)\n"
                           "Set-Content -LiteralPath (Join-Path $PSScriptRoot '../called.txt') -Value $Models\n"}
            for name, value in entries.items():
                archive.writestr(name, value)
        digest = '0'*64 if bad_checksum else hashlib.sha256(fixture.read_bytes()).hexdigest()
        quote = lambda p: "'" + str(p).replace("'", "''") + "'"
        wrapper = self.root / 'mock-bootstrap.ps1'
        wrapper.write_text(
            "$ErrorActionPreference='Stop'\n"
            f"$Fixture={quote(fixture)}\n"
            "function Invoke-WebRequest {\n"
            "param([switch]$UseBasicParsing,[string]$Uri,[string]$OutFile)\n"
            "if ($OutFile) { Copy-Item -LiteralPath $Fixture -Destination $OutFile }\n"
            f"else {{ [PSCustomObject]@{{Content='{digest}  Sammie-Roto-School.zip'}} }}\n"
            "}\n"
            f"& {quote(Path('scripts/bootstrap_school.ps1').resolve())} -Destination {quote(self.root/'installed')} -Models Essential -NoLogin -NoShortcut\n"
            "exit $LASTEXITCODE\n",
            encoding='utf-8')
        return subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(wrapper)],
                              capture_output=True, text=True, timeout=30)

    @unittest.skipUnless(os.name == 'nt', 'Windows bootstrap')
    def test_bootstrap_download_extract_and_rerun_without_overwrite(self):
        result = self.bootstrap()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        app = self.root/'installed/Sammie-Roto-School'
        self.assertEqual((app/'called.txt').read_text().strip(), 'Essential')
        (app/'personal.txt').write_text('preserved')
        # Reuse the same fixture bytes/checksum to check the resume path.
        wrapper = self.root/'mock-bootstrap.ps1'
        result = subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(wrapper)],
                                capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('Reusing', result.stdout)
        self.assertEqual((app/'personal.txt').read_text(), 'preserved')

    @unittest.skipUnless(os.name == 'nt', 'Windows bootstrap')
    def test_bootstrap_rejects_bad_hash_and_traversal(self):
        result = self.bootstrap(bad_checksum=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('checksum mismatch', result.stdout)
        self.assertFalse((self.root/'installed/Sammie-Roto-School').exists())
        result = self.bootstrap({'Sammie-Roto-School/../escape.txt': 'bad'})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Unsafe path', result.stdout)
        self.assertFalse((self.root/'installed/escape.txt').exists())

    @unittest.skipUnless(os.name == 'nt', 'Windows bootstrap')
    def test_bootstrap_refuses_existing_unrelated_installation(self):
        app = self.root/'installed/Sammie-Roto-School'
        app.mkdir(parents=True)
        (app/'personal.txt').write_text('preserved')
        result = self.bootstrap()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('No files overwritten', result.stdout)
        self.assertEqual((app/'personal.txt').read_text(), 'preserved')

    @unittest.skipUnless(os.name == 'nt', 'Windows bootstrap')
    def test_bootstrap_propagates_installer_failure(self):
        result = self.bootstrap({'Sammie-Roto-School/scripts/install_school.ps1': 'exit 23\n'})
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('Installation checks passed', result.stdout)


if __name__ == '__main__':
    unittest.main()
