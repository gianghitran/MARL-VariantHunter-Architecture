"""
logmaking.py
============
Minimal logging utility used by train_CTI.py and dataloader.py.
Original TAGAPT code uses make_print_to_file() to redirect stdout to a log file.
This is a self-contained recreation of that utility.
"""

import sys
import os
from datetime import datetime


class _Tee:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, log_file):
        self._file = log_file
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def __getattr__(self, attr):
        return getattr(self._stdout, attr)


def make_print_to_file(path="./logs", prefix=""):
    """
    Redirect print() output to both the console and a timestamped log file.

    Args:
        path   : Directory where log files are saved (created if missing).
        prefix : Optional prefix for the log filename.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}log_{timestamp}.txt" if prefix else f"log_{timestamp}.txt"
    log_path = os.path.join(path, filename)

    log_file = open(log_path, "a", encoding="utf-8")
    sys.stdout = _Tee(log_file)
    print(f"[logmaking] Logging to: {log_path}")
    return log_path
