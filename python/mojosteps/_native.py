# SPDX-License-Identifier: BSD-3-Clause
"""Validated NumPy boundary to the standalone C ABI. No forecast algorithms here."""
from __future__ import annotations
import ctypes
import operator
import os
from pathlib import Path
import sys
import threading
import numpy as np

_CALL_LOCK = threading.Lock()


def _integer(value, name, low, high):
    try:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError
        value = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be an integer") from None
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def library_path(backend):
    override = os.environ.get(f"MOJOSTEPS_{backend.upper()}_LIBRARY")
    if override:
        return Path(override).expanduser().resolve()
    suffix = "dylib" if sys.platform == "darwin" else "so"
    name = f"libmojosteps_{backend}.{suffix}"
    for root in (Path.cwd(), Path(__file__).resolve().parents[2]):
        candidate = root / ".build" / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{name} not found. In a source checkout run python scripts/build.py "
        f"--backend {backend}, or pass library=... / set MOJOSTEPS_{backend.upper()}_LIBRARY."
    )


class Forecast:
    """Load CPU or Apple Metal native core. Calls are serialized across instances.

    See docs/API.md for fixed configuration, units, bounds and C ABI lifetime rules.
    """
    def __init__(self, backend="cpu", *, library=None):
        if backend not in ("cpu", "gpu"):
            raise ValueError("backend must be cpu or gpu")
        self.backend = backend
        self.path = Path(library).expanduser().resolve() if library else library_path(backend)
        self._library = ctypes.CDLL(str(self.path))
        self._pointer = ctypes.POINTER(ctypes.c_double)
        name = "steps_forecast_native" if backend == "cpu" else "steps_forecast_gpu_all"
        self._entry = getattr(self._library, name)
        self._entry.argtypes = [self._pointer] * 3 + [ctypes.c_ssize_t] * 2 + [
            ctypes.c_uint32, ctypes.c_ssize_t, ctypes.c_int32]
        self._entry.restype = ctypes.c_int32

    def forecast(self, history, velocity, leads, *, members=96, workers=1, seed=20260817):
        """Return float64 [member, lead, y, x] dBZ; NaN marks advected-out pixels."""
        members = _integer(members, "members", 1, 96)
        leads = _integer(leads, "leads", 1, 14)
        workers = _integer(workers, "workers", 1, sys.maxsize)
        seed = _integer(seed, "seed", 0, 0xFFFFFFFF)
        if np.shape(history) != (3, 192, 192) or np.shape(velocity) != (2, 192, 192):
            raise ValueError("requires history (3,192,192), velocity (2,192,192)")
        motion = np.asarray(velocity)
        if motion.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("velocity must have float32 or float64 dtype")
        velocity32 = int(motion.dtype == np.float32)
        if np.asarray(history).dtype.kind not in "biuf":
            raise ValueError("history must contain real numeric values")
        source = np.ascontiguousarray(history, dtype=np.float64)
        motion = np.ascontiguousarray(motion, dtype=np.float64)
        if not np.all(np.isfinite(source)) or not np.all(np.isfinite(motion)):
            raise ValueError("history and velocity must be finite")
        output = np.empty((members, leads, 192, 192), dtype=np.float64)
        address = lambda a: a.ctypes.data_as(self._pointer)
        with _CALL_LOCK:
            status = self._entry(address(source), address(motion), address(output),
                                 members, leads, seed, workers, velocity32)
        if status:
            reason = {1: "invalid options", 2: "nonfinite or degenerate model",
                      3: "GPU/runtime failure"}.get(status, "unknown failure")
            raise RuntimeError(f"native forecast: {reason} (status {status})")
        return output
