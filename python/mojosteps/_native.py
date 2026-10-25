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
from .config import ForecastConfig

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

    See docs/API.md for configuration, units, bounds and C ABI lifetime rules.
    """

    def __init__(self, backend="cpu", *, library=None, config=None):
        if backend not in ("cpu", "gpu"):
            raise ValueError("backend must be cpu or gpu")
        self._config = ForecastConfig() if config is None else config
        if not isinstance(self.config, ForecastConfig):
            raise TypeError("config must be ForecastConfig")
        self.config.validate_backend(backend)
        self.backend = backend
        self.path = (
            Path(library).expanduser().resolve() if library else library_path(backend)
        )
        self._library = ctypes.CDLL(str(self.path))
        self._pointer = ctypes.POINTER(ctypes.c_double)
        name = f"steps_forecast_{backend}_v2"
        try:
            self._entry = getattr(self._library, name)
        except AttributeError:
            raise ValueError(
                "native library lacks runtime configuration ABI; rebuild with scripts/build.py"
            ) from None
        self._entry.argtypes = (
            [self._pointer] * 3
            + [ctypes.c_ssize_t] * 3
            + [ctypes.c_uint32, ctypes.c_ssize_t, ctypes.c_int32]
            + [ctypes.c_double] * 4
        )
        self._entry.restype = ctypes.c_int32

    @property
    def config(self):
        return self._config

    def forecast(
        self,
        history,
        velocity,
        leads,
        *,
        members=96,
        workers=1,
        seed=20260817,
        config=None,
    ):
        """Return float64 [member, lead, y, x] dBZ; NaN marks advected-out pixels."""
        c = self.config if config is None else config
        if not isinstance(c, ForecastConfig):
            raise TypeError("config must be ForecastConfig")
        c.validate_backend(self.backend)
        members = _integer(members, "members", 1, c.max_members)
        leads = _integer(leads, "leads", 1, c.max_leads)
        workers = _integer(workers, "workers", 1, sys.maxsize)
        seed = _integer(seed, "seed", 0, 0xFFFFFFFF)
        n = c.size
        if np.shape(history) != (3, n, n) or np.shape(velocity) != (2, n, n):
            raise ValueError(f"requires history (3,{n},{n}), velocity (2,{n},{n})")
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
        output = np.empty((members, leads, n, n), dtype=np.float64)
        address = lambda a: a.ctypes.data_as(self._pointer)
        with _CALL_LOCK:
            status = self._entry(
                address(source),
                address(motion),
                address(output),
                n,
                members,
                leads,
                seed,
                workers,
                velocity32,
                c.kmperpixel,
                c.timestep,
                c.precip_thr,
                c.norain_thr,
            )
        if status:
            reason = {
                1: "invalid options",
                2: "nonfinite or degenerate model",
                3: "GPU/runtime failure",
            }.get(status, "unknown failure")
            raise RuntimeError(f"native forecast: {reason} (status {status})")
        return output
