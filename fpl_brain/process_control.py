"""Owned child-process trees for production execution (R2A).

The production planning runner is **in-process**: ``freeze_predictions.py``,
``final_operational_refresh_gw04.py`` and ``run_four_gw_decision.py`` launch no
child processes, so process ownership exists to make that guarantee durable —
if a helper ever spawns a worker, it must be registered here so cancellation can
reap it.

Ownership contract
------------------
* Every child is registered by PID with the command line that produced it, so
  the audit trail is ``pid -> argv`` and nothing else is ever signalled.
* On Windows an owned **Job Object** is created with
  ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``; children are assigned to it, so
  closing the job (or a hard terminate) kills the whole tree even if a worker
  re-parents itself.
* ``taskkill /PID <pid> /T /F`` is used as the escalation path.  It is always
  addressed to a registered root PID, never to "all python processes", so
  unrelated interpreters are never touched.
* On POSIX the child is started in its own session (``start_new_session=True``)
  and the whole process group is signalled.

Termination is staged: cooperative ``terminate()`` first, a short grace period,
then a forced tree kill.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field

LOGGER = logging.getLogger(__name__)

IS_WINDOWS = os.name == "nt"

# Windows creation flags / job-object constants.
CREATE_NEW_PROCESS_GROUP = 0x00000200
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100
_SYNCHRONIZE = 0x00100000
_STILL_ACTIVE = 259
_WAIT_TIMEOUT = 258

_KERNEL32 = None


def _kernel32():
    """Lazily load kernel32 with explicit prototypes.

    Without ``argtypes``/``restype`` ctypes defaults handles to 32-bit ``int``
    and silently truncates them on 64-bit Windows, which makes every subsequent
    call fail.  Prototypes are set once here.
    """

    global _KERNEL32
    if not IS_WINDOWS:
        return None
    if _KERNEL32 is not None:
        return _KERNEL32
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    k32.SetInformationJobObject.restype = wintypes.BOOL
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.AssignProcessToJobObject.restype = wintypes.BOOL
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.CloseHandle.restype = wintypes.BOOL
    k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k32.GetExitCodeProcess.restype = wintypes.BOOL
    k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k32.WaitForSingleObject.restype = wintypes.DWORD
    _KERNEL32 = k32
    return k32


@dataclass
class OwnedChild:
    """One registered child process and the command that created it."""

    pid: int
    argv: tuple[str, ...]
    popen: subprocess.Popen | None = None
    registered_at: str | None = None
    terminated_at: str | None = None
    exit_status: str | None = None

    def as_dict(self) -> dict:
        return {
            "pid": int(self.pid),
            "argv": list(self.argv),
            "registered_at": self.registered_at,
            "terminated_at": self.terminated_at,
            "exit_status": self.exit_status,
        }


class _JobObject:
    """A Windows Job Object with kill-on-close semantics (no-op elsewhere)."""

    def __init__(self) -> None:
        self._handle = None
        if not IS_WINDOWS:
            return
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = _kernel32()

            class _IO_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("ReadOperationCount", ctypes.c_ulonglong),
                    ("WriteOperationCount", ctypes.c_ulonglong),
                    ("OtherOperationCount", ctypes.c_ulonglong),
                    ("ReadTransferCount", ctypes.c_ulonglong),
                    ("WriteTransferCount", ctypes.c_ulonglong),
                    ("OtherTransferCount", ctypes.c_ulonglong),
                ]

            class _BASIC_LIMIT(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
                    ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD),
                ]

            class _EXTENDED_LIMIT(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", _BASIC_LIMIT),
                    ("IoInfo", _IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t),
                ]

            self._ctypes = ctypes
            self._kernel32 = kernel32
            self._wintypes = wintypes
            self._EXTENDED_LIMIT = _EXTENDED_LIMIT
            handle = kernel32.CreateJobObjectW(None, None)
            if not handle:
                raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
            info = _EXTENDED_LIMIT()
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            ok = kernel32.SetInformationJobObject(
                handle,
                _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )
            if not ok:
                kernel32.CloseHandle(handle)
                raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")
            self._handle = handle
        except Exception as exc:  # pragma: no cover - platform dependent
            LOGGER.warning("Job object unavailable; falling back to taskkill only: %s", exc)
            self._handle = None

    @property
    def available(self) -> bool:
        return self._handle is not None

    def assign(self, pid: int) -> bool:
        if self._handle is None:
            return False
        try:
            ctypes = self._ctypes
            kernel32 = self._kernel32
            handle = kernel32.OpenProcess(_PROCESS_TERMINATE | _PROCESS_SET_QUOTA, False, int(pid))
            if not handle:
                return False
            try:
                return bool(kernel32.AssignProcessToJobObject(self._handle, handle))
            finally:
                kernel32.CloseHandle(handle)
        except Exception as exc:  # pragma: no cover - platform dependent
            LOGGER.warning("Could not assign pid %s to job object: %s", pid, exc)
            return False

    def close(self) -> None:
        if self._handle is None:
            return
        try:
            self._kernel32.CloseHandle(self._handle)
        finally:
            self._handle = None


class OwnedProcessTree:
    """Registry + terminator for the children owned by one production run."""

    def __init__(self, label: str = "run") -> None:
        self.label = label
        self._children: dict[int, OwnedChild] = {}
        self._job = _JobObject()

    # -- registration ------------------------------------------------------
    def register(self, pid: int, argv: tuple[str, ...] | list[str], *, popen=None, clock=None) -> OwnedChild:
        """Record a child as owned.  Only registered PIDs are ever signalled."""

        pid = int(pid)
        previous = self._children.get(pid)
        # ExecutionController records a spawned child in its durable audit after
        # OwnedProcessTree.spawn() has already registered the Popen handle.  A
        # second registration must not discard that handle: POSIX reaping relies
        # on the owner retaining it, and replacing it would leave only a PID
        # whose zombie state cannot be observed accurately.
        retained_popen = popen if popen is not None else (previous.popen if previous else None)
        child = OwnedChild(
            pid=pid,
            argv=tuple(str(a) for a in argv),
            popen=retained_popen,
            registered_at=(previous.registered_at if previous else (clock or _utc_now)()),
        )
        self._children[pid] = child
        self._job.assign(pid)
        LOGGER.info("registered owned child pid=%s argv=%s", pid, list(child.argv))
        return child

    def spawn(self, argv: list[str], **kwargs) -> subprocess.Popen:
        """Launch a child, own it, and return the handle."""

        flags = 0
        if IS_WINDOWS:
            flags = CREATE_NEW_PROCESS_GROUP
        else:
            kwargs.setdefault("start_new_session", True)
        popen = subprocess.Popen(argv, creationflags=flags, **kwargs)
        self.register(popen.pid, argv, popen=popen)
        return popen

    def owned(self) -> list[OwnedChild]:
        return [self._children[pid] for pid in sorted(self._children)]

    def audit_list(self) -> list[dict]:
        """The auditable ``pid -> command`` list for this run."""

        return [child.as_dict() for child in self.owned()]

    # -- liveness ----------------------------------------------------------
    @staticmethod
    def is_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        if IS_WINDOWS:
            from ctypes import wintypes

            kernel32 = _kernel32()
            handle = kernel32.OpenProcess(_SYNCHRONIZE, False, int(pid))
            if not handle:
                return False
            try:
                # WAIT_TIMEOUT (258) means the process object is still unsignalled.
                return kernel32.WaitForSingleObject(handle, 0) == _WAIT_TIMEOUT
            finally:
                kernel32.CloseHandle(handle)
        try:
            os.kill(int(pid), 0)
        except (OSError, ProcessLookupError):
            return False
        return True

    @staticmethod
    def _popen_alive(popen: subprocess.Popen) -> bool:
        """Use the owned handle when available so POSIX poll() reaps zombies."""

        return popen.poll() is None

    def _child_alive(self, child: OwnedChild) -> bool:
        if child.popen is not None:
            return self._popen_alive(child.popen)
        return self.is_alive(child.pid)

    # -- termination -------------------------------------------------------
    def request_stop(self) -> None:
        """Cooperative phase: ask owned children to stop without force."""

        for child in self.owned():
            if child.popen is not None and child.popen.poll() is None:
                try:
                    child.popen.terminate()
                    LOGGER.info("requested cooperative stop pid=%s", child.pid)
                except Exception as exc:  # pragma: no cover - race dependent
                    LOGGER.debug("cooperative stop failed pid=%s: %s", child.pid, exc)
            elif not IS_WINDOWS and self._child_alive(child):
                try:
                    os.kill(child.pid, signal.SIGTERM)
                except OSError as exc:  # pragma: no cover - race dependent
                    LOGGER.debug("SIGTERM failed pid=%s: %s", child.pid, exc)

    def terminate_all(self, grace_seconds: float = 5.0, *, clock=None, sleep=None) -> list[dict]:
        """Cooperative stop, grace period, then forced tree kill.

        Returns the final audit list.  Never signals a PID that was not
        registered with this tree.
        """

        _clock = clock or time.monotonic
        _sleep = sleep or time.sleep
        now = _utc_now

        if not self._children:
            return self.audit_list()

        self.request_stop()

        deadline = _clock() + max(0.0, float(grace_seconds))
        while _clock() < deadline:
            if not any(self._child_alive(child) for child in self.owned()):
                break
            _sleep(0.05)

        for child in self.owned():
            if not self._child_alive(child):
                child.terminated_at = child.terminated_at or now()
                child.exit_status = child.exit_status or "exited"
                continue
            self._force_kill(child)
            self._reap_posix(child)
            child.terminated_at = now()
            child.exit_status = "terminated"

        return self.audit_list()

    def _force_kill(self, child: OwnedChild) -> None:
        if IS_WINDOWS:
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(child.pid), "/T", "/F"],
                    capture_output=True,
                    timeout=15,
                    check=False,
                )
                LOGGER.info("forced tree kill pid=%s", child.pid)
                return
            except Exception as exc:  # pragma: no cover - platform dependent
                LOGGER.warning("taskkill failed pid=%s: %s", child.pid, exc)
        else:
            try:
                os.killpg(os.getpgid(child.pid), signal.SIGKILL)
                return
            except (OSError, ProcessLookupError) as exc:  # pragma: no cover
                LOGGER.debug("killpg failed pid=%s: %s", child.pid, exc)
        if child.popen is not None:
            try:
                child.popen.kill()
            except Exception as exc:  # pragma: no cover - race dependent
                LOGGER.debug("kill failed pid=%s: %s", child.pid, exc)

    def _reap_posix(self, child: OwnedChild, timeout: float = 1.0) -> None:
        """Reap an owned Popen child after forced termination, without hanging."""

        if IS_WINDOWS or child.popen is None:
            return
        try:
            child.popen.wait(timeout=max(0.0, float(timeout)))
        except subprocess.TimeoutExpired:  # pragma: no cover - hostile child
            LOGGER.warning("bounded reap timed out for owned pid=%s", child.pid)
        except (ChildProcessError, OSError) as exc:  # pragma: no cover - race dependent
            LOGGER.debug("reap failed pid=%s: %s", child.pid, exc)

    def close(self) -> None:
        """Close the job object; kill-on-close reaps anything still running."""

        self._job.close()


def _utc_now() -> str:
    from .utils import utc_now

    return utc_now()


def spawn_sleep_child(seconds: float = 60.0) -> list[str]:
    """A harmless, deterministic child command used by tests and drills."""

    return [sys.executable, "-c", f"import time; time.sleep({seconds!r})"]


__all__ = [
    "IS_WINDOWS",
    "OwnedChild",
    "OwnedProcessTree",
    "spawn_sleep_child",
]
