# SPDX-License-Identifier: BSD-3-Clause
"""Run a library built with scripts/build.py --backend gpu --profile.

Native STEPS_GPU_STAGE lines report synchronized wall-clock stage totals.
Synchronization changes scheduling. Use benchmark_gpu.py with ordinary builds
for end-to-end comparisons; these are not hardware cache-stall counters.
"""
import argparse
from pathlib import Path
import time
from mojosteps import Forecast, ForecastConfig
from mojosteps.synthetic import rain_cells

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", type=Path, default=Path(".build/libmojosteps_gpu_profile.dylib"))
    p.add_argument("--sizes", nargs="+", type=int, choices=[64,128,192,256,512], default=[192,512])
    p.add_argument("--members", type=int, default=96)
    p.add_argument("--leads", type=int, default=14)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--repeats", type=int, default=2)
    args = p.parse_args()
    for size in args.sizes:
        h, v = rain_cells(size=size)
        engine = Forecast("gpu", library=args.library, config=ForecastConfig(size=size))
        for repeat in range(args.repeats):
            start = time.perf_counter()
            actual = engine.forecast(h, v, args.leads, members=args.members, workers=args.workers)
            print("STEPS_GPU_RUN", size, repeat, time.perf_counter() - start, flush=True)
            del actual
