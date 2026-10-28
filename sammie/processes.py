"""Stop only an owned worker tree, including Windows venv launcher children."""
import os
import subprocess


def stop_worker_tree(pid, wait=True):
    if not isinstance(pid, int) or pid <= 0 or pid == os.getpid():
        raise ValueError('Expected the PID of an owned child worker.')
    if os.name == 'nt':
        args = ['taskkill', '/PID', str(pid), '/T', '/F']
        options = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=subprocess.CREATE_NO_WINDOW)
        if wait:
            subprocess.run(args, timeout=10, **options)
        else:
            subprocess.Popen(args, **options)
    else:
        import signal
        try: os.kill(pid, signal.SIGTERM)
        except ProcessLookupError: pass
