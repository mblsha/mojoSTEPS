# SPDX-License-Identifier: BSD-3-Clause
"""Reproducible moving rain cells generated without any observational data."""
import numpy as np
from scipy.ndimage import gaussian_filter, shift


def rain_cells(seed=2468, velocity_dtype="float32"):
    y, x = np.mgrid[:192, :192]
    rng = np.random.RandomState(seed)
    rain = 41 * np.exp(-((x - 103) ** 2 / 700 + (y - 88) ** 2 / 1100))
    rain += 27 * np.exp(-((x - 48) ** 2 / 300 + (y - 150) ** 2 / 600))
    rain += gaussian_filter(rng.normal(size=(192, 192)), 2) * 10
    history = []
    for frame in range(3):
        field = shift(rain, (frame * -1.4, frame * 2.35), order=1, mode="constant")
        field += gaussian_filter(rng.normal(size=(192, 192)), 1.5) * 2
        history.append(np.where(field >= 10, field, -10))
    velocity = np.array([2.35 + .2 * np.sin(y / 25), -1.4 + .15 * np.cos(x / 30)])
    return np.ascontiguousarray(history), velocity.astype(velocity_dtype)
