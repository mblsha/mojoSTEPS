# Synthetic benchmark results

Measured 25 October 2026 with the public `rain_cells_v1` generator: 96 members,
14 leads, 192 × 192 pixels, seed 20260817, float32 velocity. The CPU FFT backend
is BSD PocketFFT. These measurements include the complete native forecast call,
allocation, marshalling and final GPU readback. They exclude input generation,
hashing, comparisons and plotting.

| Hardware | CPU cores / memory | CPU, 1 worker | CPU, multiple workers | GPU, 1 CPU worker |
| --- | --- | ---: | ---: | ---: |
| Apple M5 Ultra | 36 / 256 GiB | 7.390 s | 0.855 s (24 workers) | 0.333 s |
| Apple M1 Ultra | 20 / 128 GiB | 8.807 s | 0.621 s (16 workers) | 0.762 s |

The M5 Ultra GPU is about 2.57× as fast as its 24-worker CPU result. On M1 Ultra,
the 16-worker CPU result is about 18% lower latency than GPU. These are measured
settings, not a search for each machine's optimal worker count. Host load and
scheduling were not controlled; do not use this small experiment as a general
CPU architecture ranking.

Each entry is the median of five warm native runs after one cold call. The JSON
receipts retain all timings, source hashes, compiler/library identity, input and
output hashes, environment and reference comparison. The reference uses the
same CPU worker count as each corresponding native run, with one warm timing
after one cold call. Native and reference run in separate blocks, so the reported
reference speedups are workload observations, not an interleaved statistical study.

All six complete forecasts pass: maximum error is **2.28e-9 dBZ** for CPU and
**1.93e-9 dBZ** for GPU, with identical finite masks and 10/15/20 dBZ decisions.
Full float32 arrays are not identical on this fixture; the numerical tolerance
and threshold contract pass. CPU outputs are byte-identical across the tested
worker counts and machines; GPU outputs are byte-identical across the two Macs.
CPU and GPU outputs are not byte-identical to each other or to pySTEPS.

The real-hardware test suite passes 35 tests on each Mac, including boundary
seed streams, CDF/mask probes, dry/sparse cases and float32/float64 motion.
[`verification.json`](../benchmarks/verification.json) distinguishes these runs
from hosted CPU CI and records the local Python-free C caller checks.

Reproduce using [REPRODUCIBILITY.md](REPRODUCIBILITY.md). For multicore CPU rows,
change `--workers 1` to 24 (M5 Ultra) or 16 (M1 Ultra). `workers=1` limits
application parallelism; it does not impose process CPU affinity or a CPU quota.
No operational radar data or results from the earlier FFTW prototype are used.

For a controlled comparison of CPU FFT libraries, see the [matched FFTW/PocketFFT study](FFT_COMPARISON.md), including measured/patient plans and saved-wisdom restarts.

The [RAM and Metal resource measurements](MEMORY.md) compare the CPU/GPU cores with pySTEPS using the same full forecast.
