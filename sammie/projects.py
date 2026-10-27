"""Working projects keep media in place; portable archives are explicit deliveries."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import stat
import uuid
import zipfile

MARKER = '.sammie-workspace.json'
EXCLUDED = {'.sammie.lock', MARKER, '.workflow', 'sam3_transport', 'sam3_proxy', '__pycache__'}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as output:
            json.dump(value, output, indent=2)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class ProjectLock:
    """OS lock is released even after a crash; no stale PID guessing."""
    def __init__(self, root):
        self.path = Path(root)/'.sammie.lock'
        self.file = None

    def acquire(self):
        if self.file:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        f = self.path.open('a+b')
        if not f.tell():
            f.write(b'0'); f.flush()
        f.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            f.close()
            raise RuntimeError('This working project is open or processing in another Sammie instance.') from None
        self.file = f
        return self

    def release(self):
        if self.file:
            self.file.close()
            self.file = None

    def __enter__(self): return self.acquire()
    def __exit__(self, *args): self.release()


def read_project(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('format') != 'sammie-working-project' or data.get('version') != 1:
        raise ValueError('Not a supported Sammie working project.')
    relative = Path(data['data'])
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name in ('.', '..'):
        raise ValueError('Invalid working-project data folder.')
    root = (path.parent/relative).resolve()
    if root.parent != path.parent or not (root/MARKER).is_file() or not (root/'frames').is_dir():
        raise ValueError('Project data folder is missing. Move the .sammieproj file together with its .sammie-data folder.')
    if json.loads((root/MARKER).read_text())['id'] != data['id']:
        raise ValueError('Project descriptor and data folder do not match.')
    return root, data


def session_files(root):
    root = Path(root).resolve()
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED)
        for name in sorted(files):
            path = Path(base)/name
            if name in EXCLUDED or name.endswith(('.tmp','.part')):
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError(f'Project contains an external link: {path}')
            yield path, path.relative_to(root)


def create_project(source, descriptor, progress=lambda n,t,s: None, cancel=lambda: False):
    source, descriptor = Path(source).resolve(), Path(descriptor).resolve()
    if descriptor.suffix.lower() != '.sammieproj':
        descriptor = descriptor.with_suffix('.sammieproj')
    destination = descriptor.with_suffix('.sammie-data')
    if descriptor.exists() or destination.exists():
        raise FileExistsError('Choose a new project name; an existing project is never overwritten by Save As.')
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('Save the new project outside the current session folder.')
    descriptor.parent.mkdir(parents=True, exist_ok=True)
    stage = descriptor.parent/('.sammie-create-'+uuid.uuid4().hex)
    stage.mkdir()
    try:
        files = list(session_files(source))
        for i,(path, relative) in enumerate(files):
            if cancel(): raise InterruptedError('Project creation cancelled.')
            target = stage/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            progress(i+1,len(files),f'Preparing working project: {relative}')
        if cancel(): raise InterruptedError('Project creation cancelled.')
        identity = uuid.uuid4().hex
        atomic_json(stage/MARKER, {'id':identity})
        stage.rename(destination)
        data = dict(format='sammie-working-project', version=1, id=identity, data=destination.name)
        try:
            atomic_json(descriptor,data)
        except BaseException:
            destination.rename(stage)
            raise
        return descriptor
    finally:
        if stage.exists():
            assert stage.parent == descriptor.parent and stage.name.startswith('.sammie-create-')
            shutil.rmtree(stage)


def save_metadata(root, settings, points, descriptor=None):
    """Never enumerate, hash or copy frame/cache files during a normal save."""
    root = Path(root)
    atomic_json(root/'points.json',points)
    atomic_json(root/'session_settings.conf',settings)
    if descriptor:
        actual, data = read_project(descriptor)
        if actual != root.resolve(): raise ValueError('Active project path changed.')
        data['saved_utc'] = datetime.now(timezone.utc).isoformat()
        atomic_json(descriptor,data)


def pack_project(root, archive, progress=lambda n,t,s: None, cancel=lambda: False):
    root, archive = Path(root).resolve(), Path(archive).resolve()
    if archive.is_relative_to(root):
        raise ValueError('Save the portable archive outside the working data folder.')
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(archive.name+'.'+uuid.uuid4().hex+'.part')
    try:
        files=list(session_files(root))
        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_STORED,allowZip64=True) as z:
            for i,(path,relative) in enumerate(files):
                if cancel(): raise InterruptedError('Packing cancelled; the previous archive is unchanged.')
                z.write(path,relative.as_posix())
                progress(i+1,len(files),f'Packing {relative}')
        if cancel(): raise InterruptedError('Packing cancelled; the previous archive is unchanged.')
        os.replace(temporary,archive)
    finally:
        temporary.unlink(missing_ok=True)


def unpack_project(archive, target, progress=lambda n,t,s: None, cancel=lambda: False):
    """Validate and stage legacy archives before replacing an expendable session."""
    target = Path(target).resolve()
    if (target/MARKER).exists():
        raise ValueError('Cannot unpack over a saved working project.')
    stage = target.parent/('.sammie-unpack-'+uuid.uuid4().hex)
    stage.mkdir(parents=True)
    backup = target.parent/('.sammie-old-'+uuid.uuid4().hex)
    installed = False
    try:
        with zipfile.ZipFile(archive) as z:
            names=set()
            for info in z.infolist():
                name=info.filename.replace('\\','/')
                relative=Path(name)
                dest=stage/relative
                if (relative.is_absolute() or '..' in relative.parts or ':' in name or
                    not dest.resolve().is_relative_to(stage) or stat.S_ISLNK(info.external_attr >> 16) or
                    name.casefold() in names):
                    raise ValueError('Unsafe or duplicate path in portable project.')
                names.add(name.casefold())
            for i,info in enumerate(z.infolist()):
                if cancel(): raise InterruptedError('Opening cancelled.')
                name=info.filename.replace('\\','/')
                if any(p in EXCLUDED for p in Path(name).parts): continue
                destination=stage/name
                if info.is_dir(): destination.mkdir(parents=True,exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(info) as incoming,destination.open('wb') as outgoing:
                        shutil.copyfileobj(incoming,outgoing,1024*1024)
                progress(i+1,len(names),f'Opening {name}')
        if not (stage/'frames').is_dir(): raise ValueError('Project contains no frames.')
        if cancel(): raise InterruptedError('Opening cancelled.')
        if target.exists(): target.rename(backup)
        try:
            stage.rename(target)
            installed = True
        except BaseException:
            if backup.exists(): backup.rename(target)
            raise
    finally:
        # A failed rollback must leave the last good session available for recovery.
        for folder in ((stage,backup) if installed else (stage,)):
            if folder.exists():
                assert folder.parent == target.parent and folder.name.startswith(('.sammie-unpack-','.sammie-old-'))
                shutil.rmtree(folder)


def bind_root(root, manager):
    """One explicit root for UI, managers, exports and model workers."""
    from sammie import core
    root = str(Path(root).resolve())
    core.temp_dir = root
    for attr,folder in [('frames_dir','frames'),('mask_dir','masks'),('backup_dir','masks_backup'),
                        ('matting_dir','matting'),('removal_dir','removal')]:
        setattr(core,attr,str(Path(root)/folder))
    manager.temp_dir = root
    manager.session_settings_file = str(Path(root)/'session_settings.conf')
    manager.points_file = str(Path(root)/'points.json')


def set_workspace(path):
    root=Path(path).resolve()
    root.mkdir(parents=True,exist_ok=True)
    config=root/'sammie.workspace.json'
    if config.exists():
        data=json.loads(config.read_text())
        if data.get('format')!='sammie-workspace': raise ValueError('Unknown workspace configuration.')
    for name in ('scenes','footage','exports','nuke'):
        (root/name).mkdir(exist_ok=True)
    if not config.exists():
        atomic_json(config,dict(format='sammie-workspace',version=1,
                    folders=dict(scenes='scenes',footage='footage',exports='exports',nuke='nuke')))
    return root
