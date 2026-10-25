# SPDX-License-Identifier: BSD-3-Clause
"""Interleave matched GPU libraries; time complete calls and compare every output.

Build both libraries without STEPS_GPU_PROFILE. Each library gets an untimed
warm-up per shape. Timed calls alternate AB / BA to reduce ordering bias.
"""
import argparse
import ctypes
import gc
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time
import numpy as np
from mojosteps import Forecast, ForecastConfig
from mojosteps.synthetic import rain_cells
from validate_regional import comparison, digest


def run(args):
    libraries = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    report = dict(
        platform=platform.platform(),
        host=platform.node(),
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() if (Path(__file__).resolve().parents[1] / ".git").exists() else None,
        libraries={key: dict(name=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest(), manifest=json.loads(p.with_suffix(".manifest.json").read_text()) if p.with_suffix(".manifest.json").exists() else None) for key, p in libraries.items()},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        notes=["Complete NumPy API call including allocation, model fit, GPU stages, synchronization and float64 output conversion.",
               "One untimed warm-up per library and size. Timed order alternates AB/BA; host runs under normal load.",
               "Complete output comparison outside timers; each timed result must reproduce its own library output hash. No previous output held during calls.",
               "Comparisons are against merged-baseline GPU outputs; pySTEPS qualification is separate.",
               "No GPU cache-stall percentage is inferred from these wall-clock timings."],
        cases=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for size in args.sizes:
        config = ForecastConfig(size=size)
        h, v = rain_cells(size=size)
        engines = {key: Forecast("gpu", library=path, config=config) for key, path in libraries.items()}
        addresses = [ctypes.cast(e._entry, ctypes.c_void_p).value for e in engines.values()]
        if addresses[0] == addresses[1]:
            raise RuntimeError("Both paths loaded the same native entrypoint. On macOS, copied dylibs need distinct install names; see docs/GPU_OPTIMIZATION.md.")
        # Qualify outside the timers, then release both large outputs. Holding
        # the previous library's output would give the second call a larger
        # live working set, especially for 2.625 GiB regional outputs.
        warm = {key:e.forecast(h, v, args.leads, members=args.members, workers=args.workers) for key,e in engines.items()}
        result = comparison(warm["baseline"], warm["candidate"], config.precip_thr)
        assert result["accepted"], result
        hashes = {key:digest(value) for key,value in warm.items()}
        del warm
        row = dict(size=size, members=args.members, leads=args.leads, workers=args.workers,
                   input_sha256={"history":hashlib.sha256(h.tobytes()).hexdigest(), "velocity":hashlib.sha256(v.tobytes()).hexdigest()},
                   seconds={key: [] for key in engines}, comparison=result, output_sha256=hashes)
        for repeat in range(args.repeats):
            order = ["baseline", "candidate"] if repeat % 2 == 0 else ["candidate", "baseline"]
            for key in order:
                gc.collect()
                start = time.perf_counter()
                actual = engines[key].forecast(h, v, args.leads, members=args.members, workers=args.workers)
                row["seconds"][key].append(time.perf_counter() - start)
                assert digest(actual) == hashes[key], "forecast changed between repeats"
                del actual
            print(size, repeat, {key:row["seconds"][key][-1] for key in engines}, flush=True)
        row["median_seconds"] = {key:statistics.median(values) for key,values in row["seconds"].items()}
        row["speedup"] = row["median_seconds"]["baseline"] / row["median_seconds"]["candidate"]
        report["cases"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--sizes", nargs="+", type=int, choices=[64,128,192,256,512], default=[192,512])
    p.add_argument("--members", type=int, default=96)
    p.add_argument("--leads", type=int, default=14)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
