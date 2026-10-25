# SPDX-License-Identifier: BSD-3-Clause
"""Compare a Python-free runtime C caller with the NumPy boundary."""

from pathlib import Path
import argparse
import subprocess
import sys
import numpy as np
from mojosteps import Forecast, ForecastConfig
from mojosteps.synthetic import rain_cells

ROOT = Path(__file__).resolve().parents[1]


def run(backend):
    work = ROOT / ".build/runtime-c"
    work.mkdir(parents=True, exist_ok=True)
    binary = work / "forecast-c"
    subprocess.run(
        ["cc", "-O2", str(ROOT / "examples/forecast.c"), "-o", str(binary)]
        + (["-ldl"] if sys.platform.startswith("linux") else []),
        check=True,
    )
    config = ForecastConfig(size=128, kmperpixel=1, timestep=10)
    history, velocity = rain_cells(size=config.size)
    np.concatenate([history, velocity.astype(float)]).tofile(work / "input.f64")
    engine = Forecast(backend)
    expected = engine.forecast(history, velocity, 3, members=2, config=config)
    subprocess.run(
        [
            str(binary),
            str(engine.path),
            backend,
            str(work / "input.f64"),
            str(work / "output.f64"),
            "2",
            "3",
            "1",
            "128",
            "1",
            "10",
        ],
        check=True,
    )
    actual = np.fromfile(work / "output.f64", dtype=float).reshape(expected.shape)
    assert np.array_equal(actual, expected, equal_nan=True)
    print(backend, "runtime C ABI matches Python boundary", expected.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu", "all"), default="cpu")
    backend = parser.parse_args().backend
    for selected in ("cpu", "gpu") if backend == "all" else (backend,):
        run(selected)
