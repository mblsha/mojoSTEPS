# SPDX-License-Identifier: BSD-3-Clause
"""Runtime native configuration; geometry and physical units travel together."""

from dataclasses import dataclass
import math
import numbers


@dataclass(frozen=True)
class ForecastConfig:
    size: int = 192
    kmperpixel: float = 0.5
    timestep: float = 5.0
    precip_thr: float = 10.0
    norain_thr: float = -10.0
    max_members: int = 96
    max_leads: int = 14

    def __post_init__(self):
        for key, low, high in (
            ("size", 32, 1024),
            ("max_members", 1, 4096),
            ("max_leads", 1, 288),
        ):
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Integral)
                or not low <= value <= high
            ):
                raise ValueError(f"{key} must be an integer in {low}..{high}")
            object.__setattr__(self, key, int(value))
        for key in ("kmperpixel", "timestep", "precip_thr", "norain_thr"):
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Real)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{key} must be finite")
            object.__setattr__(self, key, float(value))
        if not 0.01 <= self.kmperpixel <= 20 or not 0.1 <= self.timestep <= 60:
            raise ValueError("kmperpixel must be 0.01..20 and timestep 0.1..60 minutes")
        if not -100 <= self.norain_thr < self.precip_thr <= 100:
            raise ValueError("require -100 <= norain_thr < precip_thr <= 100 dBZ")
        self.validate_backend("cpu")

    def validate_backend(self, backend):
        if self.size not in (64, 128, 192, 256, 512):
            raise ValueError("supported square sizes are 64, 128, 192, 256 and 512")
