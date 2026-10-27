"""Verified school-source updates; models, projects and settings are never replaced.

This is standalone standard-library code so the online updater also works with
the first school release, which did not contain an updater.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import urllib.request
import uuid
import zipfile

BASE = 'https://github.com/ThomasBignolas/Sammie-Roto-2-SAM3/releases/latest/download/'
PREFIX = 'Sammie-Roto-School/'
PROTECTED = {'.git', '.venv', '.runtime', '.uv', 'temp', '.test-output', '.cache', '.agents', '.codex'}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def target_path(root, name):
    p = PurePosixPath(name)
    if ('\\' in name or ':' in name or p.is_absolute() or '..' in p.parts or
        not p.parts or PROTECTED.intersection(p.parts) or
        p.name.startswith('.env') or p.name in ('sammie_settings.conf','sammie_settings.json','token','stored_tokens','.git-credentials') or
        p.suffix.lower() in ('.sammie','.nk','.exr') or
        (p.suffix.lower() in ('.pt','.pth','.safetensors','.bin') and name != 'checkpoints/1x_binary_mask_smooth.pth') or
        (p.parts[0] == 'checkpoints' and p.suffix != '.json' and name != 'checkpoints/1x_binary_mask_smooth.pth')):
        raise ValueError(f'Protected or unsafe update path: {name}')
    result = root.joinpath(*p.parts)
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Update path escapes the installation: {name}')
    return result


def validate_archive(archive, expected):
    if sha(archive) != expected:
        raise ValueError('Release checksum mismatch. No installed file was changed; retry the download.')
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(set(n.casefold() for n in names)) != len(names):
            raise ValueError('Release archive contains duplicate paths.')
        for item in z.infolist():
            if not item.filename.startswith(PREFIX) or stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError('Unexpected root or symlink in release archive.')
            target_path(Path.cwd(),item.filename[len(PREFIX):])
        if sum(item.file_size for item in z.infolist()) > 512*2**20:
            raise ValueError('Source package exceeds 512 MiB; refusing a model/media bundle.')
        manifest = json.loads(z.read(PREFIX+'SCHOOL_BUILD.json'))
        if not re.fullmatch('[0-9a-f]{40}',manifest.get('commit','')):
            raise ValueError('Release manifest has no valid commit identity.')
        if set(names) != {PREFIX+n for n in manifest['sha256']} | {PREFIX+'SCHOOL_BUILD.json'}:
            raise ValueError('Archive and release manifest disagree.')
        for name, digest in manifest['sha256'].items():
            if hashlib.sha256(z.read(PREFIX+name)).hexdigest() != digest:
                raise ValueError(f'Invalid file hash in release: {name}')
        for name in ('install_school.bat','scripts/install_school.ps1','run_sam3.bat'):
            if name not in manifest['sha256']:
                raise ValueError(f'Release is missing {name}.')
        return manifest


def apply_archive(root, archive, expected, replace=os.replace):
    root, archive = Path(root).resolve(), Path(archive).resolve()
    new = validate_archive(archive, expected)
    previous = root/'SCHOOL_BUILD.json'
    if not previous.is_file():
        raise ValueError('No school package manifest here. Use a Git checkout or install the school ZIP first.')
    old = json.loads(previous.read_text(encoding='utf-8'))
    changes = []
    # Validate every destination and reject artist modifications before writing.
    for name in sorted(set(new['sha256']) | set(old['sha256'])):
        path = target_path(root,name)
        current = sha(path) if path.is_file() else None
        wanted = new['sha256'].get(name)
        if current == wanted:
            continue
        if current is not None and current != old['sha256'].get(name):
            raise ValueError(f'Local source edit/collision: {name}. Keep your changes; update a separate installation or resolve this file first.')
        changes.append(name)
    if not changes and old.get('commit') == new['commit']:
        return dict(commit=new['commit'],changed=0,backup=None)
    changes.append('SCHOOL_BUILD.json')
    storage = root/'.runtime'/'update-backups'/uuid.uuid4().hex
    if not storage.resolve().is_relative_to(root):
        raise ValueError('Update backup directory escapes the installation.')
    backup, staged = storage/'previous', storage/'new'
    backup.mkdir(parents=True)
    staged.mkdir()
    had_file, applied = set(), []
    with zipfile.ZipFile(archive) as z:
        for name in changes:
            target = target_path(root,name)
            if target.is_file():
                dest = backup/name
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(target,dest)
                had_file.add(name)
            if name in new['sha256'] or name == 'SCHOOL_BUILD.json':
                dest = staged/name
                dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(z.read(PREFIX+name))
    try:
        for name in changes:
            target = target_path(root,name)
            target.parent.mkdir(parents=True,exist_ok=True)
            source = staged/name
            if source.exists():
                replace(source,target)
            else:
                target.unlink(missing_ok=True)  # Only unchanged, formerly packaged source.
            applied.append(name)
    except BaseException:
        failures = []
        for name in reversed(applied):
            target = target_path(root,name)
            try:
                if name in had_file:
                    shutil.copy2(backup/name,target)
                else:
                    target.unlink(missing_ok=True)
            except OSError:
                failures.append(name)
        if failures:
            raise RuntimeError(f'Update failed; rollback could not restore {failures}. Close applications and restore those files from {backup}.') from None
        raise RuntimeError(f'Update failed; previous source files restored. Backup: {backup}') from None
    marker = root/'.runtime/school-package.sha256'
    marker.write_text(expected+'\n',encoding='ascii')
    return dict(commit=new['commit'],changed=len(changes)-1,backup=str(backup))


def update_git(root, commit):
    def git(*args):
        return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
    remote = git('remote','get-url','origin').removesuffix('.git').rstrip('/')
    if remote not in ('https://github.com/ThomasBignolas/Sammie-Roto-2-SAM3',
                       'git@github.com:ThomasBignolas/Sammie-Roto-2-SAM3'):
        raise ValueError('Origin is not the school repository; no Git update attempted.')
    if git('status','--porcelain','--untracked-files=no'):
        raise ValueError('Commit or preserve your local source changes before updating this Git checkout.')
    current = git('rev-parse','HEAD')
    if current != commit:
        git('fetch','origin',commit)
        # Never reset, switch branches, overwrite local commits, or go backwards.
        result = subprocess.run(['git','-C',str(root),'merge-base','--is-ancestor',current,commit])
        if result.returncode:
            raise ValueError('This checkout has local/newer commits. No reset was attempted; reconcile with the release manually.')
        git('merge','--ff-only',commit)
    return dict(commit=commit,changed=int(current!=commit),backup='Git history')


def fetch(url, destination=None):
    with urllib.request.urlopen(url,timeout=90) as response:
        if destination is None:
            return response.read(4096).decode('ascii')
        with destination.open('wb') as output:
            shutil.copyfileobj(response,output,1024*1024)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--archive',type=Path,help='IT/testing: already downloaded release ZIP')
    parser.add_argument('--sha256',help='Required with --archive')
    parser.add_argument('--plan',action='store_true')
    args=parser.parse_args()
    root=args.root.resolve()
    if not (root/'run_sam3.bat').is_file():
        raise SystemExit('Choose the installed Sammie-Roto-School folder containing run_sam3.bat.')
    if args.plan:
        print(f'Update {root} from the latest published school release. Models, projects and settings are preserved.')
        return
    archive=args.archive
    if archive:
        if not args.sha256: raise SystemExit('--archive requires --sha256')
        expected=args.sha256
    else:
        expected=fetch(BASE+'Sammie-Roto-School.zip.sha256').split()[0]
        if not re.fullmatch('[0-9a-f]{64}',expected): raise SystemExit('Invalid release checksum.')
        folder=root/'.runtime/update-downloads'
        if not folder.resolve().is_relative_to(root): raise SystemExit('Unsafe update download folder.')
        folder.mkdir(parents=True,exist_ok=True)
        archive=folder/(expected+'.zip')
        if not archive.is_file() or sha(archive)!=expected:
            print('Downloading the latest source package...',flush=True)
            fetch(BASE+'Sammie-Roto-School.zip',archive)
    manifest=validate_archive(archive,expected)
    report=update_git(root,manifest['commit']) if (root/'.git').exists() else apply_archive(root,archive,expected)
    print(json.dumps(report,indent=2),flush=True)
    print('Source update complete. The launcher will now reconcile dependencies; model weights stay cached.',flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        raise SystemExit(f'Update stopped: {error}') from None
