"""AutoHealth package."""

from __future__ import annotations

import os


def _disable_joblib_multiprocessing_if_blocked() -> None:
    # Avoid joblib warning when POSIX semaphores are blocked by the environment.
    if "JOBLIB_MULTIPROCESSING" in os.environ:
        return
    try:
        from _multiprocessing import SemLock

        name = f"/joblib-check-{os.getpid()}"
        sem = SemLock(0, 0, 1, name=name, unlink=True)
        del sem
    except PermissionError:
        os.environ["JOBLIB_MULTIPROCESSING"] = "0"
    except Exception:
        # Fall back to joblib's own detection if we cannot probe safely.
        pass


_disable_joblib_multiprocessing_if_blocked()
