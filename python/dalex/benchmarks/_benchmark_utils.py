"""Benchmark measurement helpers independent of the DALEX implementation."""
import threading
import hashlib
from pathlib import Path

import numpy as np
import psutil


def source_identity(package):
    """Fingerprint imported Python sources, including archived/editable installs."""
    root = Path(package.__file__).resolve().parent
    files = sorted(root.rglob('*.py'))
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.relative_to(root).as_posix().encode())
        digest.update(b'\0')
        digest.update(file.read_bytes().replace(b'\r\n', b'\n'))
        digest.update(b'\0')
    return {'python_modules': len(files), 'python_source_sha256': digest.hexdigest(),
            'normalization': 'sorted relative paths and source bytes, CRLF normalized to LF'}


class RSS:
    def __enter__(self):
        self.process = psutil.Process()
        self.baseline = self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        while not self.stop.wait(.002):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __exit__(self, *args):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()


def summary(values):
    return {'median': float(np.median(values)), 'q25': float(np.quantile(values, .25)),
            'q75': float(np.quantile(values, .75)), 'min': min(values), 'max': max(values)}
