"""End a helper process when the process that started it dies.

The web UI cancels a job by terminating its inference worker, and a terminated process takes
none of its own children with it: the retrain subprocess the worker started kept training on
the GPU with nobody reading its output.  A child that watches its parent closes that gap no
matter how the parent went away -- cancel, crash, or the launcher window being closed.
"""
from __future__ import annotations

import os
import sys
import threading
import time


def exit_when_parent_dies(parent: int, poll_s: float = 1.0) -> None:
    """Start a daemon thread that hard-exits this process once ``parent`` is gone.

    The parent passes its own pid on the command line rather than this reading ``os.getppid()``
    at startup: a parent that dies during the child's slow imports has already re-parented it,
    and the watchdog would then guard the wrong process forever.
    """
    if sys.platform == "win32":
        target = _wait_windows(parent)
    else:
        def target() -> None:
            # An orphan is re-parented (to init or a subreaper), so its parent pid changes.
            while os.getppid() == parent:
                time.sleep(poll_s)
            _exit()
    threading.Thread(target=target, name="parent-watchdog", daemon=True).start()


def _wait_windows(parent: int):
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    synchronize, infinite = 0x00100000, 0xFFFFFFFF
    # Opened now, not inside the thread: the handle pins this exact process, so a later pid
    # reuse cannot make a dead parent look alive.
    handle = kernel32.OpenProcess(synchronize, False, parent)
    if not handle:
        _exit()

    def target() -> None:
        kernel32.WaitForSingleObject(handle, infinite)
        _exit()

    return target


def _exit() -> None:
    print("[ERR ] Parent process exited -- stopping.", flush=True)
    os._exit(1)
