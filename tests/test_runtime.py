"""Full-array checks for the runtime ABI, including real Metal when opted in."""

from dataclasses import replace
import numpy as np
import pytest
from mojosteps import Forecast, ForecastConfig
from mojosteps.synthetic import rain_cells
from mojosteps.validation import compare, reference_forecast


@pytest.mark.parametrize(
    "size,km,step,threshold,dry",
    [
        (64, 0.25, 2.5, 10.0, -10.0),
        (128, 1.0, 10.0, 10.0, -10.0),
        (256, 0.5, 5.0, 12.0, -12.0),
        (512, 1.0, 5.0, 10.0, -10.0),
    ],
)
def test_runtime_profiles(engine, size, km, step, threshold, dry):
    config = ForecastConfig(
        size=size, kmperpixel=km, timestep=step, precip_thr=threshold, norain_thr=dry
    )
    history, velocity = rain_cells(size=size, velocity_dtype="float64")
    history = np.where(history < threshold, dry, history)
    actual = engine.forecast(history, velocity, 3, members=2, seed=0, config=config)
    reference = reference_forecast(
        history, velocity, 3, members=2, seed=0, config=config
    )
    result = compare(reference, actual)
    assert result["accepted"], result


def test_runtime_horizon_and_units(engine):
    config = ForecastConfig(size=64, timestep=1.0, max_leads=18)
    h, v = rain_cells(size=64)
    actual = engine.forecast(h, v, 18, members=2, config=config)
    result = compare(reference_forecast(h, v, 18, members=2, config=config), actual)
    assert result["accepted"], result
    other = engine.forecast(h, v, 2, members=2, config=replace(config, timestep=5.0))
    assert not np.array_equal(actual[:, :2], other, equal_nan=True)


@pytest.mark.parametrize(
    "option,value",
    [
        ("size", 63),
        ("size", 384),
        ("kmperpixel", 0),
        ("timestep", float("nan")),
        ("precip_thr", -10),
        ("max_leads", 289),
        ("max_members", False),
    ],
)
def test_bad_configuration(option, value):
    with pytest.raises(ValueError):
        ForecastConfig(**{option: value})


def test_old_library_configuration_rejected(tmp_path):
    import subprocess
    import sys

    source = tmp_path / "legacy.c"
    source.write_text("int steps_forecast_native(void) { return 0; }\n")
    output = tmp_path / ("legacy.dylib" if sys.platform == "darwin" else "legacy.so")
    flags = ["-dynamiclib"] if sys.platform == "darwin" else ["-shared", "-fPIC"]
    subprocess.run(["cc", *flags, str(source), "-o", str(output)], check=True)
    with pytest.raises(ValueError, match="runtime configuration ABI"):
        Forecast(library=output)


def test_runtime_larger_ensemble(engine):
    config = ForecastConfig(size=64, max_members=97)
    h, v = rain_cells(size=64)
    actual = engine.forecast(h, v, 1, members=97, config=config)
    result = compare(reference_forecast(h, v, 1, members=97, config=config), actual)
    assert result["accepted"], result


@pytest.mark.parametrize(
    "seed,leads", [(633713063, 10), (83773847, 8), (300923882, 4), (359157101, 3)]
)
def test_large_grid_correlation_rank_regression(engine, seed, leads):
    config = ForecastConfig(size=512)
    h, v = rain_cells(size=512)
    actual = engine.forecast(h, v, leads, members=1, seed=seed, config=config)
    result = compare(
        reference_forecast(h, v, leads, members=1, seed=seed, config=config), actual
    )
    assert result["accepted"], result
