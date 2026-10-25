# SPDX-License-Identifier: BSD-3-Clause
"""One regional ensemble, many local queries. Observation/motion handling is Python."""

from dataclasses import dataclass
import numpy as np
from .radar import Grid, prepare_input


@dataclass(frozen=True)
class AreaTarget:
    name: str
    weights: np.ndarray

    def __post_init__(self):
        weights = np.array(self.weights, dtype=float, copy=True)
        if (
            weights.ndim != 2
            or not np.isfinite(weights).all()
            or (weights < 0).any()
            or weights.sum() <= 0
        ):
            raise ValueError(
                "target weights must be finite, nonnegative and have positive area"
            )
        weights.flags.writeable = False
        object.__setattr__(self, "weights", weights)

    @classmethod
    def circle(cls, name, grid, x, y, radius_m=1000.0):
        """Pixel intersection areas for a 256-edge circle in the grid's metre CRS."""
        from shapely.geometry import Point, box

        if not np.isfinite([x, y, radius_m]).all() or radius_m <= 0:
            raise ValueError("target centre/radius must be finite and radius positive")
        circle = Point(x, y).buffer(radius_m, quad_segs=64)
        yy, xx = np.indices(grid.shape)
        xs = grid.x0 + xx * grid.dx
        ys = grid.y0 + yy * grid.dy
        candidate = (np.abs(xs - x) <= radius_m + abs(grid.dx) / 2) & (
            np.abs(ys - y) <= radius_m + abs(grid.dy) / 2
        )
        weights = np.zeros(grid.shape)
        for row, column in zip(*np.nonzero(candidate)):
            cx, cy = xs[row, column], ys[row, column]
            weights[row, column] = circle.intersection(
                box(
                    cx - abs(grid.dx) / 2,
                    cy - abs(grid.dy) / 2,
                    cx + abs(grid.dx) / 2,
                    cy + abs(grid.dy) / 2,
                )
            ).area
        if weights.sum() < circle.area * (1 - 1e-8):
            raise ValueError("target extends outside the forecast grid")
        return cls(name, weights)

    @classmethod
    def lonlat_circle(cls, name, grid, longitude, latitude, radius_m=1000.0):
        from pyproj import Transformer

        x, y = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True).transform(
            longitude, latitude
        )
        return cls.circle(name, grid, x, y, radius_m)


@dataclass
class RegionalForecast:
    ensemble: np.ndarray
    inputs: object
    config: object

    def _states(self, target, threshold, wet_fraction):
        if target.weights.shape != self.inputs.grid.shape:
            raise ValueError("target weights do not match forecast grid")
        if not np.isfinite(threshold) or not 0 < wet_fraction <= 1:
            raise ValueError("threshold must be finite and wet_fraction in (0,1]")
        selected = target.weights > 0
        weights = target.weights[selected]
        values = self.ensemble[..., selected]
        eligible = np.isfinite(values).all(axis=-1)
        wet = (
            np.sum((values >= threshold) * weights, axis=-1) / weights.sum()
            >= wet_fraction
        )
        return wet, eligible

    def probabilities(
        self, target, thresholds=(15.0, 20.0), *, wet_fraction=0.2, minimum_members=1
    ):
        """Area-event probability and usable member count at each forecast lead."""
        if (
            isinstance(minimum_members, bool)
            or not isinstance(minimum_members, int)
            or minimum_members < 1
        ):
            raise ValueError("minimum_members must be a positive integer")
        result = {}
        for threshold in thresholds:
            wet, eligible = self._states(target, threshold, wet_fraction)
            count = eligible.sum(axis=0)
            probability = np.divide(
                (wet & eligible).sum(axis=0),
                count,
                out=np.full(count.shape, np.nan),
                where=count >= minimum_members,
            )
            result[str(float(threshold))] = dict(
                probability=probability, valid_members=count
            )
        return result

    def window_risk(
        self,
        target,
        start_minutes,
        duration_minutes,
        *,
        threshold=15.0,
        wet_fraction=0.2,
        minimum_members=1,
    ):
        """Probability of any wet state in an inclusive window, using whole trajectories."""
        times = (
            np.array([start_minutes, duration_minutes], dtype=float)
            / self.config.timestep
        )
        if (
            not np.isfinite(times).all()
            or not np.allclose(times, np.rint(times))
            or times[0] < 0
            or times[1] <= 0
        ):
            raise ValueError(
                "window must use nonnegative scan-aligned start and positive duration"
            )
        start, duration = np.rint(times).astype(int)
        if start + duration > self.ensemble.shape[1]:
            raise ValueError("window exceeds forecast horizon")
        if (
            isinstance(minimum_members, bool)
            or not isinstance(minimum_members, int)
            or minimum_members < 1
        ):
            raise ValueError("minimum_members must be a positive integer")
        wet, eligible = self._states(target, threshold, wet_fraction)
        selected = target.weights > 0
        weights = target.weights[selected]
        latest = self.inputs.history[-1][selected]
        current = (
            np.sum((latest >= threshold) * weights) / weights.sum() >= wet_fraction
        )
        wet = np.concatenate([np.full((len(wet), 1), current), wet], axis=1)
        eligible = np.concatenate(
            [np.ones((len(wet), 1), dtype=bool), eligible], axis=1
        )
        usable = eligible[:, start : start + duration + 1].all(axis=1)
        count = int(usable.sum())
        risk = (
            float(wet[usable, start : start + duration + 1].any(axis=1).mean())
            if count >= minimum_members
            else None
        )
        return dict(probability=risk, valid_members=count)


class RegionalForecaster:
    def __init__(self, engine):
        self.engine = engine

    def forecast(
        self,
        frames,
        grid,
        origin,
        velocity,
        leads,
        *,
        priority,
        members=96,
        workers=1,
        seed=20260817,
    ):
        inputs = prepare_input(
            frames, grid, self.engine.config, origin, priority=priority
        )
        # Velocity must be on the target grid, x/y pixels per configured scan.
        # A caller may estimate it from inputs.history using an external motion API.
        ensemble = self.engine.forecast(
            inputs.history, velocity, leads, members=members, workers=workers, seed=seed
        )
        return RegionalForecast(ensemble, inputs, self.engine.config)


@dataclass(frozen=True)
class Tile:
    grid: Grid
    row: int
    column: int
    query_slice: tuple[slice, slice]


def tiles(grid, size, *, halo=32):
    """Overlapping square forecast tiles for a rectangular observation domain.

    Query only query_slice; the halo supplies surrounding motion context.
    Ensembles from separate tiles are not joint member trajectories.
    """
    if (
        any(isinstance(v, bool) or not isinstance(v, int) for v in (size, halo))
        or size < 1
        or halo < 0
        or 2 * halo >= size
    ):
        raise ValueError("require integer size and 0 <= halo < size/2")
    if min(grid.shape) < size:
        raise ValueError("observation domain is smaller than a forecast tile")

    def starts(length):
        return sorted(
            set([*range(0, length - size + 1, size - 2 * halo), length - size])
        )

    for row in starts(grid.shape[0]):
        for column in starts(grid.shape[1]):
            yield Tile(
                grid.window(row, column, size),
                row,
                column,
                (slice(halo, size - halo), slice(halo, size - halo)),
            )
