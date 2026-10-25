# SPDX-License-Identifier: BSD-3-Clause
"""Synthetic two-source mosaic and three hyperlocal queries from one native call."""

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
import time
import numpy as np
from mojosteps import Forecast, ForecastConfig
from mojosteps.radar import Grid, from_dbz, from_rain_rate
from mojosteps.regional import AreaTarget, RegionalForecaster
from mojosteps.synthetic import rain_cells


def run(backend="cpu", size=192, kmperpixel=0.5, timestep=5.0, members=8, leads=6):
    config = ForecastConfig(
        size=size, kmperpixel=kmperpixel, timestep=timestep, max_leads=max(14, leads)
    )
    grid = Grid((size, size), "EPSG:3577", 0, 0, kmperpixel * 1000, -kmperpixel * 1000)
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    history, velocity = rain_cells(size=size)
    frames = []
    for index, field in enumerate(history):
        stamp = origin - timedelta(minutes=(2 - index) * timestep)
        west = np.ones(grid.shape, dtype=bool)
        west[:, size // 2 :] = False
        frames.append(from_dbz("west-dbz", stamp, grid, field, valid=west))
        # Synthetic alternative input quantity, with an explicitly chosen Z-R law.
        rain = np.where(field >= 10, (10 ** (field / 10) / 200) ** (1 / 1.6), 0)
        frames.append(
            from_rain_rate("east-rain-rate", stamp, grid, rain, zr_a=200, zr_b=1.6)
        )
    worker = RegionalForecaster(Forecast(backend, config=config))
    start = time.perf_counter()
    forecast = worker.forecast(
        frames,
        grid,
        origin,
        velocity,
        leads,
        priority=["west-dbz", "east-rain-rate"],
        members=members,
    )
    seconds = time.perf_counter() - start
    products = {}
    for name, fraction in [("west", 0.35), ("centre", 0.5), ("east", 0.65)]:
        target = AreaTarget.circle(
            name, grid, fraction * size * grid.dx, 0.5 * size * grid.dy
        )
        products[name] = forecast.probabilities(target)

    def encode(value):
        if isinstance(value, np.ndarray):
            return [None if not np.isfinite(v) else float(v) for v in value]
        raise TypeError(type(value).__name__)

    receipt = dict(
        backend=backend,
        configuration=asdict(config),
        members=members,
        leads=leads,
        origin=origin.isoformat(),
        seconds=seconds,
        native_forecast_calls=1,
        source_pixel_counts={
            source: int((forecast.inputs.provenance == i).sum())
            for i, source in enumerate(forecast.inputs.source_ids)
        },
        products=products,
        claim="Synthetic radar-echo probabilities, not validated surface-rain guidance",
    )
    return json.loads(json.dumps(receipt, default=encode))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--kmperpixel", type=float, default=0.5)
    parser.add_argument("--timestep", type=float, default=5.0)
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--leads", type=int, default=6)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2, allow_nan=False))
