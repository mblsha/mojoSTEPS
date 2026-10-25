# SPDX-License-Identifier: BSD-3-Clause
"""Synthetic demo and reproducible, end-to-end native-core benchmark."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import time
import numpy as np
from . import Forecast, __version__
from .synthetic import rain_cells
from .validation import compare, reference_forecast


def _digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def _plot(path, history, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    last = result[:, -1]
    count = np.isfinite(last).sum(axis=0)
    mean = np.divide(np.nansum(last, axis=0), count, out=np.full_like(last[0], np.nan), where=count > 0)
    probability = np.divide(((last.astype(np.float32) >= 15) & np.isfinite(last)).sum(axis=0),
                            count, out=np.full_like(last[0], np.nan), where=count > 0)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
    for ax, data, title, limits, cmap in zip(axes,
        (history[-1], mean, probability),
        ("Synthetic latest input", f"Ensemble mean at +{result.shape[1]*5} min", "Probability ≥ 15 dBZ"),
        ((10, 45), (10, 45), (0, 1)), ("turbo", "turbo", "viridis")):
        image = ax.imshow(data, vmin=limits[0], vmax=limits[1], cmap=cmap, origin="upper")
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(image, ax=ax, shrink=.75)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("demo", help="generate moving synthetic cells and forecast them")
    d.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    d.add_argument("--members", type=int, default=4)
    d.add_argument("--leads", type=int, default=3)
    d.add_argument("--workers", type=int, default=1)
    d.add_argument("--seed", type=int, default=20260817)
    d.add_argument("--velocity-dtype", choices=("float32", "float64"), default="float32")
    d.add_argument("--runs", type=int, default=3, help="warm native runs after one cold call")
    d.add_argument("--reference-runs", type=int, default=1, help="warm reference runs after one cold call")
    d.add_argument("--compare", action="store_true")
    d.add_argument("--no-plot", action="store_true")
    d.add_argument("--no-save-arrays", action="store_true")
    d.add_argument("--output", type=Path, default=Path(".build/demo"))
    args = p.parse_args(argv)
    if args.runs < 1 or args.reference_runs < 1:
        p.error("timing runs must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    history, velocity = rain_cells(velocity_dtype=args.velocity_dtype)
    engine = Forecast(args.backend)
    options = dict(members=args.members, workers=args.workers, seed=args.seed)
    timings, digests = [], []
    result = None
    for run in range(args.runs + 1):
        start = time.perf_counter()
        value = engine.forecast(history, velocity, args.leads, **options)
        timings.append(time.perf_counter() - start)
        digests.append(_digest(value))
        if run == 0:
            result = value
        del value
    if len(set(digests)) != 1:
        raise RuntimeError("forecast changed across identical calls")
    report = dict(version=__version__, backend=args.backend, fixture="rain_cells_v1",
                  synthetic_seed=2468, **options, leads=args.leads,
                  velocity_dtype=args.velocity_dtype, platform=platform.platform(),
                  machine=platform.machine(), native_cold_seconds=timings[0],
                  native_warm_seconds=timings[1:], native_median_seconds=statistics.median(timings[1:]),
                  output_sha256=digests[0],
                  input_sha256=dict(history=_digest(history), velocity=_digest(velocity)),
                  library_sha256=hashlib.sha256(engine.path.read_bytes()).hexdigest(),
                  versions={pkg: importlib.metadata.version(pkg) for pkg in ("numpy", "scipy")},
                  thread_environment={key: os.environ.get(key) for key in
                      ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})
    manifest = engine.path.with_suffix(".manifest.json")
    if manifest.exists():
        built = json.loads(manifest.read_text())
        report["source_sha256"] = built["source_sha256"]
        report["mojo_version"] = built["mojo_version"]
    if args.compare:
        ref_times = []
        reference = None
        for run in range(args.reference_runs + 1):
            start = time.perf_counter()
            value = reference_forecast(history, velocity, args.leads, **options)
            ref_times.append(time.perf_counter() - start)
            if run == 0:
                reference = value
            elif not np.array_equal(reference, value, equal_nan=True):
                raise RuntimeError("pySTEPS reference changed across identical calls")
            del value
        report["reference_cold_seconds"] = ref_times[0]
        report["reference_warm_seconds"] = ref_times[1:]
        report["reference_median_seconds"] = statistics.median(ref_times[1:])
        report["speedup"] = report["reference_median_seconds"] / report["native_median_seconds"]
        report["reference_sha256"] = _digest(reference)
        report["comparison"] = compare(reference, result)
        report["versions"]["pysteps"] = importlib.metadata.version("pysteps")
    if not args.no_save_arrays:
        np.savez(args.output / "forecast.npz", history=history, velocity=velocity, forecast=result)
        # Raw little-endian inputs are also usable by examples/forecast.c.
        np.concatenate((history.ravel(), velocity.ravel())).astype("<f8").tofile(args.output / "input.f64")
    if not args.no_plot:
        _plot(args.output / "preview.png", history, result)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.compare and not report["comparison"]["accepted"]:
        raise SystemExit("numerical acceptance failed; see summary.json")

if __name__ == "__main__":
    main()
