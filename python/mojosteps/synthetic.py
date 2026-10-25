# SPDX-License-Identifier: BSD-3-Clause
"""Reproducible moving rain cells generated without any observational data."""
import numpy as np
from scipy.ndimage import gaussian_filter, shift


def rain_cells(seed=2468, velocity_dtype="float32", size=192):
    y, x = np.mgrid[:size, :size]
    scale=size/192
    rng = np.random.RandomState(seed)
    rain = 41 * np.exp(-((x - 103*scale) ** 2 / (700*scale**2) + (y - 88*scale) ** 2 / (1100*scale**2)))
    rain += 27 * np.exp(-((x - 48*scale) ** 2 / (300*scale**2) + (y - 150*scale) ** 2 / (600*scale**2)))
    rain += gaussian_filter(rng.normal(size=(size,size)), 2) * 10
    history = []
    for frame in range(3):
        field = shift(rain, (frame * -1.4, frame * 2.35), order=1, mode="constant")
        field += gaussian_filter(rng.normal(size=(size,size)), 1.5) * 2
        history.append(np.where(field >= 10, field, -10))
    velocity = np.array([2.35 + .2 * np.sin(y / 25), -1.4 + .15 * np.cos(x / 30)])
    return np.ascontiguousarray(history), velocity.astype(velocity_dtype)
