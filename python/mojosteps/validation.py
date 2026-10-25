# SPDX-License-Identifier: BSD-3-Clause
"""Optional pySTEPS reference and a deliberately explicit numerical contract."""
import contextlib
import io
import numpy as np
from .config import ForecastConfig


def reference_forecast(history, velocity, leads, *, members=96, workers=1, seed=20260817, config=None):
    config=config or ForecastConfig()
    # Persistence is mojoSTEPS' documented sparse-input policy, outside pySTEPS.
    if np.count_nonzero(history >= config.precip_thr) < 16:
        return np.tile(history[-1], (members, leads, 1, 1))
    with contextlib.redirect_stdout(io.StringIO()):
        from pysteps.nowcasts.steps import forecast
        return forecast(history, velocity, leads, n_ens_members=members,
                        n_cascade_levels=6, precip_thr=config.precip_thr, norain_thr=config.norain_thr,
                        kmperpixel=config.kmperpixel, timestep=config.timestep, num_workers=workers,
                        seed=seed, vel_pert_method="bps")


def compare(reference, actual):
    if reference.shape != actual.shape:
        raise ValueError("forecast shapes differ")
    finite_r, finite_a = np.isfinite(reference), np.isfinite(actual)
    both = finite_r & finite_a
    errors = np.abs(reference[both] - actual[both])
    r32, a32 = reference.astype(np.float32), actual.astype(np.float32)
    states = {str(t): int(np.count_nonzero((r32 >= t) != (a32 >= t)))
              for t in (10., 15., 20.)}
    finite_exact = bool(np.array_equal(finite_r, finite_a))
    max_error = float(errors.max()) if errors.size else 0.0
    return dict(shape=list(reference.shape), values=int(reference.size),
                finite_masks_identical=finite_exact,
                max_absolute_error_dbz=max_error,
                mean_absolute_error_dbz=float(errors.mean()) if errors.size else 0.0,
                float32_arrays_identical=bool(np.array_equal(r32, a32, equal_nan=True)),
                threshold_state_differences=states,
                accepted=finite_exact and max_error <= 1e-5 and not any(states.values()))
