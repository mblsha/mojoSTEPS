# SPDX-License-Identifier: BSD-3-Clause
"""Source adapters and timestamp-exact mosaics, outside the native forecast core."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import numpy as np
from scipy.ndimage import map_coordinates


def utc(value):
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError("observation times must be timezone-aware datetimes")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class Grid:
    """Regular Cartesian grid. x0/y0 are the first pixel's centre, in metres."""

    shape: tuple[int, int]
    crs: str
    x0: float
    y0: float
    dx: float
    dy: float

    def __post_init__(self):
        from pyproj import CRS

        if len(self.shape) != 2 or any(
            isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1
            for n in self.shape
        ):
            raise ValueError("grid shape must contain two positive integers")
        crs = CRS.from_user_input(self.crs)
        if not crs.is_projected or any(
            abs(axis.unit_conversion_factor - 1) > 1e-12 for axis in crs.axis_info[:2]
        ):
            raise ValueError("grid CRS must be projected with metre axes")
        if (
            not np.isfinite([self.x0, self.y0, self.dx, self.dy]).all()
            or not self.dx
            or not self.dy
        ):
            raise ValueError("grid coordinates and nonzero spacing must be finite")
        object.__setattr__(self, "shape", tuple(map(int, self.shape)))
        object.__setattr__(self, "crs", crs.to_wkt())

    def window(self, row, column, size):
        if (
            row < 0
            or column < 0
            or row + size > self.shape[0]
            or column + size > self.shape[1]
        ):
            raise ValueError("tile extends outside source grid")
        return Grid(
            (size, size),
            self.crs,
            self.x0 + column * self.dx,
            self.y0 + row * self.dy,
            self.dx,
            self.dy,
        )


@dataclass(frozen=True)
class RadarFrame:
    source_id: str
    valid_time: datetime
    grid: Grid
    dbz: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("source_id is required")
        values = np.array(
            np.ma.filled(np.ma.asarray(self.dbz, dtype=float), np.nan), copy=True
        )
        if values.shape != self.grid.shape:
            raise ValueError("frame and grid shapes differ")
        values[~np.isfinite(values)] = np.nan
        values.flags.writeable = False
        object.__setattr__(self, "dbz", values)
        object.__setattr__(self, "valid_time", utc(self.valid_time))
        object.__setattr__(self, "metadata", dict(self.metadata))


def from_dbz(source_id, valid_time, grid, values, *, valid=None, metadata=None):
    values = np.array(
        np.ma.filled(np.ma.asarray(values, dtype=float), np.nan), copy=True
    )
    if valid is not None:
        mask = np.asarray(valid)
        if mask.shape != values.shape or mask.dtype != bool:
            raise ValueError("valid must be a boolean array matching the field")
        values[~mask] = np.nan
    return RadarFrame(source_id, valid_time, grid, values, metadata or {})


def from_rain_rate(
    source_id,
    valid_time,
    grid,
    mm_per_hour,
    *,
    zr_a,
    zr_b,
    norain_dbz=-10.0,
    valid=None,
):
    """Convert an explicitly chosen Z=aR**b relation; missing/negative stays missing."""
    if not np.isfinite([zr_a, zr_b, norain_dbz]).all() or zr_a <= 0 or zr_b <= 0:
        raise ValueError("Z-R coefficients must be finite and positive")
    rain = np.ma.filled(np.ma.asarray(mm_per_hour, dtype=float), np.nan)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        dbz = 10 * np.log10(zr_a) + 10 * zr_b * np.log10(rain)
    dbz = np.where(rain == 0, norain_dbz, dbz)
    dbz = np.where(rain < 0, np.nan, dbz)
    return from_dbz(
        source_id,
        valid_time,
        grid,
        dbz,
        valid=valid,
        metadata={"quantity": "rain_rate_converted", "zr_a": zr_a, "zr_b": zr_b},
    )


def from_palette(source_id, valid_time, grid, rgba, palette):
    """Explicit RGBA->proxy dBZ mapping. Unknown colours remain missing."""
    pixels = np.asarray(rgba)
    if pixels.shape != (*grid.shape, 4) or pixels.dtype != np.uint8:
        raise ValueError("palette input must be uint8 [height,width,4]")
    dbz = np.full(grid.shape, np.nan)
    for colour, value in palette.items():
        if (
            len(colour) != 4
            or not all(
                isinstance(v, (int, np.integer)) and 0 <= v <= 255 for v in colour
            )
            or not np.isfinite(value)
        ):
            raise ValueError("palette requires RGBA byte tuples and finite dBZ values")
        dbz[(pixels == colour).all(axis=-1)] = value
    return RadarFrame(source_id, valid_time, grid, dbz, {"quantity": "display_proxy"})


def read_cf_netcdf(
    path,
    source_id,
    *,
    reflectivity="reflectivity",
    time="valid_time",
    x="x",
    y="y",
    valid=None,
):
    """Read a single CF projected dBZ field, including BOM RF3 ground reflectivity.

    CF packing and fill values are decoded by netCDF4. Quality/error fields are
    not guessed: pass an explicitly interpreted boolean valid mask when needed.
    """
    from netCDF4 import Dataset, num2date
    from pyproj import CRS

    path = Path(path)
    with Dataset(path) as data:
        variable = data[reflectivity]
        if variable.units.lower() != "dbz" or variable.dimensions != (y, x):
            raise ValueError(
                "expected a two-dimensional CF dBZ field with (y,x) dimensions"
            )
        coords = []
        for name in (x, y):
            axis = data[name]
            factor = {
                "m": 1.0,
                "metres": 1.0,
                "meters": 1.0,
                "km": 1000.0,
                "kilometres": 1000.0,
            }.get(axis.units.lower())
            raw_axis = axis[:]
            if np.ma.getmaskarray(raw_axis).any():
                raise ValueError("CF coordinate axes must not contain masked values")
            values = np.asarray(raw_axis, dtype=float)
            if (
                factor is None
                or values.ndim != 1
                or len(values) < 2
                or not np.isfinite(values).all()
            ):
                raise ValueError("CF coordinates must be finite regular metre/km axes")
            values *= factor
            spacing = values[1] - values[0]
            if not spacing or not np.allclose(
                np.diff(values), spacing, rtol=1e-8, atol=1e-6
            ):
                raise ValueError("CF coordinates must be regularly spaced")
            coords.append((values[0], spacing, len(values)))
        projection = data[variable.grid_mapping]
        crs = CRS.from_cf(
            {key: projection.getncattr(key) for key in projection.ncattrs()}
        )
        clock = data[time]
        raw_time = clock[:]
        if np.ma.getmaskarray(raw_time).any() or not np.isfinite(raw_time).all():
            raise ValueError("CF valid time must not be masked or nonfinite")
        if np.size(raw_time) != 1:
            raise ValueError("expected a single CF valid time")
        stamp = num2date(
            float(np.asarray(raw_time).item()),
            clock.units,
            calendar=getattr(clock, "calendar", "standard"),
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )
        xx, yy = coords
        grid = Grid((yy[2], xx[2]), crs.to_wkt(), xx[0], yy[0], xx[1], yy[1])
        values = variable[:]
    return from_dbz(
        source_id,
        stamp.replace(tzinfo=timezone.utc),
        grid,
        values,
        valid=valid,
        metadata={
            "quantity": "ground_reflectivity",
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    )


def resample(frame, target):
    """Bilinear reflectivity alignment; every contributing pixel must be valid."""
    from pyproj import Transformer

    yy, xx = np.indices(target.shape, dtype=float)
    xx = target.x0 + xx * target.dx
    yy = target.y0 + yy * target.dy
    if frame.grid.crs != target.crs:
        xx, yy = Transformer.from_crs(
            target.crs, frame.grid.crs, always_xy=True
        ).transform(xx, yy)
    coords = np.array(
        [(yy - frame.grid.y0) / frame.grid.dy, (xx - frame.grid.x0) / frame.grid.dx]
    )
    rounded = np.rint(coords)
    coords = np.where(np.abs(coords - rounded) < 1e-8, rounded, coords)
    valid = np.isfinite(frame.dbz)
    invalid_weight = map_coordinates(
        (~valid).astype(float),
        coords,
        order=1,
        mode="constant",
        cval=1,
        prefilter=False,
    )
    values = map_coordinates(
        np.where(valid, frame.dbz, 0),
        coords,
        order=1,
        mode="constant",
        cval=0,
        prefilter=False,
    )
    return np.where(invalid_weight == 0, values, np.nan)


class CoverageError(ValueError):
    """No complete, timestamp-aligned observational support for this tile."""


@dataclass(frozen=True)
class PreparedInput:
    history: np.ndarray
    grid: Grid
    origin: datetime
    source_ids: tuple[str, ...]
    provenance: np.ndarray


def prepare_input(frames, grid, config, origin, *, priority):
    """Choose first valid source per pixel at exact times; never average sources."""
    origin = utc(origin)
    if grid.shape != (config.size, config.size) or not np.allclose(
        [abs(grid.dx), abs(grid.dy)], config.kmperpixel * 1000, rtol=0, atol=1e-6
    ):
        raise ValueError(
            "target grid must match the native size and physical pixel scale"
        )
    priority = tuple(priority)
    if not priority or len(set(priority)) != len(priority):
        raise ValueError("priority must contain unique source ids")
    indexed = {}
    for frame in frames:
        key = (frame.source_id, frame.valid_time)
        if frame.source_id not in priority or key in indexed:
            raise ValueError("unknown source or duplicate source/time")
        indexed[key] = frame
    history = np.full((3, *grid.shape), np.nan)
    provenance = np.full(history.shape, -1, dtype=np.int32)
    for i in range(3):
        stamp = origin - timedelta(minutes=(2 - i) * config.timestep)
        for rank, source in enumerate(priority):
            frame = indexed.get((source, stamp))
            if frame is None:
                continue
            values = resample(frame, grid)
            take = np.isnan(history[i]) & np.isfinite(values)
            history[i][take] = values[take]
            provenance[i][take] = rank
        missing = np.count_nonzero(~np.isfinite(history[i]))
        if missing:
            raise CoverageError(
                f"{stamp.isoformat()}: {missing} uncovered pixels; forecast unavailable"
            )
    history.flags.writeable = False
    provenance.flags.writeable = False
    return PreparedInput(history, grid, origin, priority, provenance)
