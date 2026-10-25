# SPDX-License-Identifier: BSD-3-Clause
"""Record full-array parity for runtime configurations on real CPU/Metal hardware."""

import argparse
from dataclasses import asdict
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import numpy as np
from mojosteps import Forecast, ForecastConfig
from mojosteps.synthetic import rain_cells
from mojosteps.validation import compare, reference_forecast

ROOT = Path(__file__).resolve().parents[1]


def digest(array):
    code = hashlib.sha256()
    for member in array:
        for plane in member:
            code.update(memoryview(plane).cast("B"))
    return code.hexdigest()


def comparison(reference, actual, threshold):
    # Keep diagnostic memory bounded even for multi-GiB forecast outputs.
    finite = 0
    error_sum = 0.0
    maximum = 0.0
    masks = True
    exact = True
    states = {str(t): 0 for t in sorted({10.0, 15.0, 20.0, threshold})}
    for m in range(len(actual)):
        for lead in range(actual.shape[1]):
            r, a = reference[m, lead], actual[m, lead]
            result = compare(r, a)
            count = int((np.isfinite(r) & np.isfinite(a)).sum())
            finite += count
            error_sum += result["mean_absolute_error_dbz"] * count
            maximum = max(maximum, result["max_absolute_error_dbz"])
            masks &= result["finite_masks_identical"]
            exact &= result["float32_arrays_identical"]
            r32, a32 = r.astype(np.float32), a.astype(np.float32)
            for key in states:
                states[key] += int(
                    np.count_nonzero((r32 >= float(key)) != (a32 >= float(key)))
                )
    return dict(
        shape=list(actual.shape),
        finite_masks_identical=masks,
        max_absolute_error_dbz=maximum,
        mean_absolute_error_dbz=error_sum / finite if finite else 0.0,
        float32_arrays_identical=exact,
        threshold_state_differences=states,
        accepted=masks and maximum <= 1e-5 and not any(states.values()),
    )


def run(args):
    cases = [
        (64, 0.25, 2.5, 10.0, -10.0, 3, 3),
        (128, 1.0, 10.0, 10.0, -10.0, 3, 3),
        (192, 0.5, 5.0, 10.0, -10.0, 3, 3),
        (256, 0.5, 5.0, 12.0, -12.0, 3, 3),
        (512, 1.0, 5.0, 10.0, -10.0, 3, 3),
        (64, 0.5, 1.0, 10.0, -10.0, 3, 18),
        (64, 0.5, 5.0, 10.0, -10.0, 97, 1),
    ]
    if args.full:
        cases.extend(
            [(192, 0.5, 5.0, 10.0, -10.0, 96, 14), (512, 0.5, 5.0, 10.0, -10.0, 96, 14)]
        )
    engines = {backend: Forecast(backend) for backend in args.backends}
    report = dict(
        platform=platform.platform(),
        machine=platform.machine(),
        cases=[],
        versions={
            key: importlib.metadata.version(key)
            for key in ("numpy", "scipy", "pysteps", "netCDF4", "pyproj", "shapely")
        },
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "python/mojosteps").glob("*.py"))
        },
        native_manifests={
            key: json.loads(e.path.with_suffix(".manifest.json").read_text())
            for key, e in engines.items()
        },
        notes=[
            "Synthetic source only; array/threshold agreement does not establish weather skill.",
            "First and warm timings are diagnostic observations under normal host load, not controlled architecture comparisons.",
            "Reference output is retained outside native timings; these are not memory benchmarks.",
            "Configuration changes use the same library instance; no profile rebuild or per-source compilation.",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for n, km, step, thr, dry, members, leads in cases:
        config = ForecastConfig(
            size=n,
            kmperpixel=km,
            timestep=step,
            precip_thr=thr,
            norain_thr=dry,
            max_members=max(96, members),
            max_leads=max(14, leads),
        )
        h, v = rain_cells(size=n)
        h = np.where(h < thr, dry, h)
        print("reference", n, members, leads, flush=True)
        reference = reference_forecast(
            h, v, leads, members=members, workers=args.workers, config=config
        )
        refhash = digest(reference)
        inputs = {
            key: hashlib.sha256(value.tobytes()).hexdigest()
            for key, value in [("history", h), ("velocity", v)]
        }
        for backend, engine in engines.items():
            times = []
            for repeat in range(2):
                gc.collect()
                start = time.perf_counter()
                actual = engine.forecast(
                    h, v, leads, members=members, workers=args.workers, config=config
                )
                times.append(time.perf_counter() - start)
                if not repeat:
                    del actual
            result = comparison(reference, actual, thr)
            row = dict(
                backend=backend,
                configuration=asdict(config),
                members=members,
                workers=args.workers,
                leads=leads,
                input_sha256=inputs,
                output_sha256=digest(actual),
                reference_sha256=refhash,
                native_seconds=times,
                comparison=result,
            )
            report["cases"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                backend,
                n,
                members,
                leads,
                "seconds",
                times,
                "max_error",
                result["max_absolute_error_dbz"],
                "accepted",
                result["accepted"],
                flush=True,
            )
            assert result["accepted"], row
            del actual
        del reference
    print(
        "All", len(report["cases"]), "backend/configuration checks passed", flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends", nargs="+", choices=("cpu", "gpu"), default=["cpu", "gpu"]
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="also qualify 96 members x 14 leads at 192 and 512 square",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / ".build/regional-validation.json"
    )
    parser.add_argument("--workers", type=int, default=1)
    run(parser.parse_args())
