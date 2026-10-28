"""Small, explicit resource budgets; no changes to model precision or resolution."""
from collections import OrderedDict
import os


def available_memory():
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        if os.name == 'nt':
            import ctypes
            class MemoryStatus(ctypes.Structure):
                _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in
                    ('total', 'available', 'page_total', 'page_available', 'virtual_total', 'virtual_available', 'extended')]
            state = MemoryStatus()
            state.length = ctypes.sizeof(state)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
                return state.available
        return 2 * 1024**3  # Conservative fallback, not a measurement.


def cpu_budget(requested=0):
    available = max(1, (os.cpu_count() or 2) - 2)
    return min(available, max(1, int(requested))) if requested else min(4, available)


def configure_cpu(requested=0, torch_module=None):
    import cv2
    count = cpu_budget(requested)
    cv2.setNumThreads(count)
    if torch_module is not None:
        torch_module.set_num_threads(count)
    return count


def memory_budget(mib):
    # Caches/queued decoded frames may consume at most 1/8 of current free RAM.
    return min(max(0, int(mib)) * 1024**2, available_memory() // 8)


def require_host_memory(estimated_bytes, operation):
    available = available_memory()
    if estimated_bytes > available * .6:
        raise MemoryError(f'{operation} needs approximately {estimated_bytes/1024**3:.1f} GiB for its frame buffers; '
                          f'only {available/1024**3:.1f} GiB RAM is currently available. '
                          'Shorten the In/Out range, lower processing resolution, or close other applications.')


class ArrayCache:
    """LRU bounded by bytes, including zero-budget and oversize entries."""
    def __init__(self, limit):
        self.limit = limit
        self.bytes = 0
        self.items = OrderedDict()

    def get(self, key):
        if key not in self.items:
            return None
        self.items.move_to_end(key)
        return self.items[key]

    def put(self, key, value):
        old = self.items.pop(key, None)
        if old is not None:
            self.bytes -= old.nbytes
        if value.nbytes > self.limit:
            return
        while self.items and self.bytes + value.nbytes > self.limit:
            self.bytes -= self.items.popitem(last=False)[1].nbytes
        self.items[key] = value
        self.bytes += value.nbytes

    def clear(self):
        self.items.clear()
        self.bytes = 0


def file_stamp(path):
    try:
        s = path.stat()
        return str(path), s.st_size, s.st_mtime_ns, s.st_ctime_ns
    except FileNotFoundError:
        return str(path), None
