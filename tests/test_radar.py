from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import numpy as np
import pytest
from mojosteps import ForecastConfig
from mojosteps.radar import (
    Grid,
    RadarFrame,
    CoverageError,
    from_dbz,
    from_palette,
    from_rain_rate,
    prepare_input,
    resample,
    read_cf_netcdf,
)
from mojosteps.regional import AreaTarget, RegionalForecast, tiles

STAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)


def grid(shape=(64, 64), dx=500.0, dy=-500.0, x0=0.0, y0=0.0):
    return Grid(shape, "EPSG:3577", x0, y0, dx, dy)


def test_timestamp_mosaic_priority_and_missing():
    g = grid()
    config = ForecastConfig(size=64)
    frames = []
    for i in range(3):
        stamp = STAMP - timedelta(minutes=(2 - i) * 5)
        west = np.full(g.shape, 15.0)
        west[:, 32:] = np.nan
        frames.extend(
            [
                RadarFrame("west", stamp, g, west),
                RadarFrame("east", stamp, g, np.full(g.shape, 25.0)),
            ]
        )
    prepared = prepare_input(frames, g, config, STAMP, priority=["west", "east"])
    assert (prepared.history[:, :, :32] == 15).all()
    assert (prepared.history[:, :, 32:] == 25).all()
    assert np.array_equal(prepared.provenance[0], np.tile(np.arange(64) >= 32, (64, 1)))
    with pytest.raises(CoverageError, match="uncovered"):
        prepare_input(frames[:1], g, config, STAMP, priority=["west", "east"])
    with pytest.raises(ValueError, match="duplicate"):
        prepare_input(frames + [frames[0]], g, config, STAMP, priority=["west", "east"])
    with pytest.raises(CoverageError):
        prepare_input(
            frames, g, config, STAMP + timedelta(minutes=1), priority=["west", "east"]
        )


def test_resampling_and_invalid_support():
    source = grid((5, 5), dx=1000, dy=-1000)
    y, x = np.indices(source.shape)
    field = x + 10 * y
    frame = RadarFrame("radar", STAMP, source, field)
    target = grid((7, 7), x0=500, y0=-500)
    result = resample(frame, target)
    yy, xx = np.indices(target.shape)
    np.testing.assert_allclose(result, (xx + 1) / 2 + 10 * (yy + 1) / 2)
    bad = field.astype(float)
    bad[2, 2] = np.nan
    result = resample(RadarFrame("radar", STAMP, source, bad), target)
    assert np.isnan(result[2, 2])
    assert np.isfinite(result[0, 0])
    # Even a very small nonzero contribution from an invalid neighbour is missing.
    tiny = grid((1, 1), dx=1000, dy=-1000, x0=1000.001, y0=-1000.001)
    assert np.isnan(resample(RadarFrame("radar", STAMP, source, bad), tiny)[0, 0])

    from pyproj import Transformer

    lon, lat = Transformer.from_crs(source.crs, "EPSG:3857", always_xy=True).transform(
        0, 0
    )
    transformed = Grid((1, 1), "EPSG:3857", lon, lat, 500, -500)
    assert abs(resample(frame, transformed)[0, 0]) < 1e-6


def test_adapters_do_not_invent_dry_observations():
    g = grid((2, 2))
    valid = np.array([[True, False], [True, True]])
    frame = from_dbz("x", STAMP, g, np.ones((2, 2)), valid=valid)
    assert np.isnan(frame.dbz[0, 1])
    rain = from_rain_rate("x", STAMP, g, [[0, 1], [-1, np.nan]], zr_a=200, zr_b=1.6)
    np.testing.assert_allclose(rain.dbz[0], [-10, 10 * np.log10(200)])
    assert np.isnan(rain.dbz[1]).all()
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    rgba[1, 1] = [255, 0, 0, 255]
    palette = from_palette("x", STAMP, g, rgba, {(0, 0, 0, 0): -10})
    assert np.isnan(palette.dbz[1, 1])
    with pytest.raises(ValueError, match="timezone-aware"):
        RadarFrame("x", datetime(2026, 1, 1), g, np.ones((2, 2)))
    with pytest.raises(ValueError, match="projected"):
        Grid((2, 2), "EPSG:4326", 0, 0, 0.1, 0.1)


def test_cf_netcdf_packing_geometry_and_fill(tmp_path):
    from netCDF4 import Dataset
    from pyproj import CRS

    path = tmp_path / "synthetic.nc"
    with Dataset(path, "w") as data:
        data.createDimension("x", 3)
        data.createDimension("y", 2)
        for name, values in [("x", [1, 1.5, 2]), ("y", [5, 4.5])]:
            v = data.createVariable(name, "f8", (name,))
            v.units = "km"
            v[:] = values
        clock = data.createVariable("valid_time", "i8")
        clock.units = "seconds since 1970-01-01 00:00:00 UTC"
        clock[:] = STAMP.timestamp()
        projection = data.createVariable("proj", "i1")
        projection.setncatts(CRS.from_epsg(3577).to_cf())
        v = data.createVariable("reflectivity", "i2", ("y", "x"), fill_value=-32768)
        v.units = "dBZ"
        v.grid_mapping = "proj"
        v.scale_factor = 0.01
        v[:] = np.ma.array(
            [[10, 20, 30], [5, 15, 25]],
            mask=[[False, True, False], [False, False, False]],
        )
    frame = read_cf_netcdf(path, "synthetic")
    assert frame.valid_time == STAMP
    assert frame.grid.shape == (2, 3)
    assert (frame.grid.x0, frame.grid.y0, frame.grid.dx, frame.grid.dy) == (
        1000,
        5000,
        500,
        -500,
    )
    assert np.isnan(frame.dbz[0, 1])
    assert frame.dbz[1, 2] == 25
    assert len(frame.metadata["file_sha256"]) == 64
    with Dataset(path, "a") as data:
        data["y"][0] = np.ma.masked
    with pytest.raises(ValueError, match="masked"):
        read_cf_netcdf(path, "synthetic")
    with Dataset(path, "a") as data:
        data["y"][0] = 5
        data["valid_time"][:] = np.ma.masked
    with pytest.raises(ValueError, match="masked"):
        read_cf_netcdf(path, "synthetic")


def test_area_probabilities_preserve_joint_trajectories():
    g = grid((2, 2))
    weights = np.array([[3.0, 1.0], [0, 0]])
    target = AreaTarget("office", weights)
    ensemble = np.full((3, 2, 2, 2), -10.0)
    ensemble[0, 0, 0, 0] = 20
    ensemble[1, 1, 0, 0] = 20
    ensemble[2, :, 0, 0] = np.nan
    inputs = SimpleNamespace(grid=g, history=np.full((3, 2, 2), -10.0))
    result = RegionalForecast(ensemble, inputs, ForecastConfig())
    product = result.probabilities(target, thresholds=[15], minimum_members=2)["15.0"]
    np.testing.assert_array_equal(product["valid_members"], [2, 2])
    np.testing.assert_array_equal(product["probability"], [0.5, 0.5])
    # Marginals are .5 each, but disjoint member events make window risk 1.
    assert result.window_risk(target, 5, 5, minimum_members=2) == dict(
        probability=1.0, valid_members=2
    )
    assert result.window_risk(target, 5, 5, minimum_members=3)["probability"] is None
    assert np.isnan(
        result.probabilities(target, minimum_members=3)["15.0"]["probability"]
    ).all()


def test_circle_and_rectangular_tile_coordinates():
    g = grid((256, 320))
    target = AreaTarget.circle("office", g, 50000, -50000, 1000)
    assert target.weights.sum() == pytest.approx(np.pi * 1000**2, rel=0.0002)
    with pytest.raises(ValueError, match="outside"):
        AreaTarget.circle("edge", g, 0, 0, 1000)
    plan = list(tiles(g, 128, halo=16))
    assert len(plan) > 1
    assert plan[-1].grid.x0 == (320 - 128) * 500
    assert plan[-1].grid.y0 == -(256 - 128) * 500
    assert plan[-1].query_slice == (slice(16, 112), slice(16, 112))


def test_one_regional_forecast_serves_multiple_targets(engine):
    from mojosteps.regional import RegionalForecaster
    from mojosteps.synthetic import rain_cells
    from mojosteps import Forecast

    config = ForecastConfig(size=64)
    history, velocity = rain_cells(size=64)
    g = grid()
    frames = []
    for i, field in enumerate(history):
        stamp = STAMP - timedelta(minutes=(2 - i) * 5)
        west = field.copy()
        west[:, 32:] = np.nan
        frames.extend(
            [RadarFrame("west", stamp, g, west), RadarFrame("east", stamp, g, field)]
        )
    local = Forecast(engine.backend, config=config)
    regional = RegionalForecaster(local).forecast(
        frames, g, STAMP, velocity, 3, priority=["west", "east"], members=2
    )
    np.testing.assert_array_equal(regional.inputs.history, history)
    direct = local.forecast(history, velocity, 3, members=2)
    np.testing.assert_array_equal(regional.ensemble, direct)
    for x in (12000, 20000):
        target = AreaTarget.circle(str(x), g, x, -16000)
        assert regional.probabilities(target)["15.0"]["probability"].shape == (3,)
