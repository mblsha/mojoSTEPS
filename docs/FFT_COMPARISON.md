# Matched FFTW and PocketFFT results

Measured 25 October 2026. **Tuned FFTW gives a modest CPU forecast improvement**
for this fixture. In the rotating 16-worker comparison, PATIENT reduces median
latency by **6.6% on M5 Ultra** and **4.8% on M1 Ultra**. It is faster in all five
paired rounds on each host. Tuning costs about four seconds; saved wisdom avoids
repeating it. ESTIMATE does not consistently beat PocketFFT.

The full workload is 96 members × 14 leads × 192 × 192 pixels, seed 20260817,
with the exact same synthetic history and float32 motion arrays on both hosts.
Each FFT uses one CPU thread. Worker counts below limit application parallelism,
not process CPU affinity or an OS quota. The same compiled Mojo forecast library
is used for all FFT providers on a host. Mojo is 1.2.0, MAX is 26.7.0 and FFTW is 3.3.11.

## Full matrix

Seconds per complete warm forecast, median of five calls after one cold call.
Each cell is a separate process with initially empty FFTW wisdom. Cell order is
shuffled with a fixed seed. These blocks are followed by a rotating-order check.

| Hardware | Workers | PocketFFT | FFTW ESTIMATE | FFTW MEASURE | FFTW PATIENT |
| --- | ---: | ---: | ---: | ---: | ---: |
| M5 Ultra | 1 | 7.351 | 7.501 | 7.155 | 7.215 |
| M5 Ultra | 4 | 1.890 | 1.910 | 1.796 | 1.805 |
| M5 Ultra | 16 | 1.014 | 0.978 | 0.979 | 0.923 |
| M1 Ultra | 1 | 8.809 | 9.105 | 8.646 | 8.634 |
| M1 Ultra | 4 | 2.374 | 2.434 | 2.312 | 2.306 |
| M1 Ultra | 16 | 0.646 | 0.645 | 0.596 | 0.596 |

## Rotating-order check at 16 workers

Each of five rounds rotates/reverses the four library modes. Each mode gets a
fresh process, one warmup forecast and one timed warm forecast. FFTW imports a
frozen copy of its matrix-cell wisdom using `WISDOM_ONLY`, keeping the selected
plans fixed. Table changes are reductions in the ratio of median latencies;
paired per-round changes and all samples are retained in the JSON receipts.

| Library / mode | M5 Ultra | Latency reduction vs PocketFFT | M1 Ultra | Latency reduction vs PocketFFT |
| --- | ---: | ---: | ---: | ---: |
| PocketFFT | 1.006 s | baseline | 0.628 s | baseline |
| FFTW ESTIMATE | 0.994 s | +1.2% | 0.637 s | -1.5% |
| FFTW MEASURE | 0.983 s | +2.3% | 0.599 s | +4.6% |
| FFTW PATIENT | 0.939 s | +6.6% | 0.597 s | +4.8% |

The paired-round median PATIENT reductions are 6.5% and 4.8%, respectively.
M5 Ultra has more variation: PocketFFT spans 0.943–1.042 s and PATIENT spans
0.930–0.974 s. This is evidence for a small gain, not a general hardware ranking
or a guarantee for every field. On M1 Ultra, MEASURE captures almost all the
PATIENT benefit with much less planning work.

## Planning and saved wisdom

These are the original 16-worker matrix cold calls, before any FFT warmup in
that process. Planner time is the sum inside FFTW's planner mutex; total forecast
time includes that cost. A separate fresh process imports saved wisdom and
requires compatible plans without retuning.

| Hardware | Mode | FFTW planner time | First forecast, no wisdom | First forecast, saved wisdom | Wisdom import |
| --- | --- | ---: | ---: | ---: | ---: |
| M5 Ultra | MEASURE | 0.494 s | 1.453 s | 1.014 s | 0.541 ms |
| M5 Ultra | PATIENT | 4.129 s | 5.068 s | 0.995 s | 0.221 ms |
| M1 Ultra | MEASURE | 0.473 s | 1.073 s | 0.606 s | 0.242 ms |
| M1 Ultra | PATIENT | 3.973 s | 4.570 s | 0.607 s | 0.267 ms |

Warm forecasts still include construction/destruction of 194 FFTW plans, with
automatic wisdom reuse and serialized planner access, matching the current Mojo
core's object lifetimes. We did not redesign the core to retain plans between
forecasts. PocketFFT cache initialization is included in its cold wall time;
its zero-valued FFTW-specific planner counter does not mean zero setup work.

## FFT-only measurement

Microseconds per 192 × 192 real forward + inverse transform pair, including
inverse normalization and excluding plan construction. These are the main
16-worker cell's post-forecast microbenchmarks: one FFT thread, five batches of
1,000 pairs, seeded random-normal input. They isolate FFT work from RNG, CDF
sorting, masks, advection and other forecast stages.

| Hardware | PocketFFT | FFTW ESTIMATE | FFTW MEASURE | FFTW PATIENT |
| --- | ---: | ---: | ---: | ---: |
| M5 Ultra | 151.9 | 188.5 | 127.7 | 125.9 |
| M1 Ultra | 190.6 | 243.7 | 165.7 | 167.6 |

## Numerical validation and interpretation

All **228 complete forecast calls** pass the pySTEPS acceptance contract:
maximum finite error is **2.27e-09 dBZ**, finite masks match, and
10/15/20 dBZ decisions after float32 conversion are identical. The CPU output
bytes are identical across every tested library, planner mode, worker count,
repeat and host on this fixture. They are not bitwise identical to pySTEPS.
The input and reference SHA256 identities also match across hosts.

The experiment changes only the CPU FFT provider. The GPU backend uses its own
Mojo/Metal transforms. PocketFFT remains the default: FFTW tuning offers a small
latency improvement here and adds an external dependency, planning/wisdom
management and separate licensing requirements for linked binaries. The default
build and release package do not link FFTW.

Full raw timing samples, plan descriptions, numerical comparisons, process CPU
time, source/compiler/provider identities and receipt hashes are retained in
[`m5-ultra.json`](../benchmarks/fft-comparison/m5-ultra.json) and
[`m1-ultra.json`](../benchmarks/fft-comparison/m1-ultra.json). Shared build/input
metadata is factored out once per host; individual cell samples are preserved.
Earlier CPU/GPU receipts use different worker settings and were measured at a
different time; use this matched study for FFT-library comparisons.

## Protocol and reproduction

This experiment compares the CPU forecast using the same synthetic arrays,
Mojo source, compiled forecast core, output validation and application worker
counts on Apple M5 Ultra and M1 Ultra. It covers PocketFFT and FFTW 3.3.11
`ESTIMATE`, `MEASURE` and `PATIENT`, with 1, 4 and 16 workers on each host.
Each FFT is single-threaded; the worker count controls ensemble members. It is
not a process CPU quota or affinity setting.

The default build and vendored PocketFFT implementation are unchanged. The
optional experiment builds one Mojo shared library with unresolved FFT-provider
symbols. Each isolated Python subprocess loads one provider globally, then the
same forecast library. The core binary hash is identical across providers on a
host. FFTW is an externally installed library; no FFTW implementation or linked
native binary is distributed by this experiment.

## Timing and correctness

The workload is 96 members × 14 leads × 192 × 192 pixels, using `rain_cells_v1`,
float32 velocity and seed 20260817. The exact generated fixture is copied between
hosts; receipts record array and reference hashes. Both reference environments
use NumPy 2.5.1, SciPy 1.18.0 and pySTEPS 1.21.2.

Each cell runs in a fresh process and starts with empty FFTW wisdom. One cold
forecast includes initial model fitting, runtime startup and FFT planning; five
warm forecasts follow. All calls include validation, allocation, all model and
ensemble stages, output conversion and cleanup. Library loading and input
preparation happen before the timer. Hashing, reference comparison and JSON
writing happen afterward. A fixed shuffled cell order reduces a systematic
library-order bias, but this is a block experiment, not randomized interleaving
of every warm call. Only one timed process runs per host.

FFTW planning and destruction use a mutex as required by its shared planner
state. `fftw_planner_seconds_sum` measures plan creation inside that lock; it
excludes waiting for the lock and destruction. Those costs remain inside total
forecast wall time. FFTW is allowed to overwrite its newly allocated scratch
arrays while planning; input history and velocity are preserved. Each forecast
recreates plans as the existing Mojo core does; automatic wisdom reuse remains
in effect for warm runs. There is no planner time limit.

Separate fresh-process restart trials import each tuned mode's saved wisdom
and require `FFTW_WISDOM_ONLY`: a missing compatible plan is an error, rather
than silently retuning. These report a cold forecast plus import time, not a
warm median. An FFT-only microbenchmark runs after all forecast calls in a cell
so it cannot seed the cold plan cache. It measures 1,000 forward/inverse pairs
plus inverse normalization, with planning excluded, repeated in five batches.

Every distinct forecast output hash is checked against the full reference.
Acceptance requires identical finite masks, maximum finite error ≤1e-5 dBZ and
identical decisions at 10/15/20 dBZ after float32 conversion. Identical repeats
reuse that comparison. Numerical agreement is independent of speed.

## Rotating-order follow-up

A second experiment checks the 16-worker result across five rounds. Each round
rotates/reverses library order. Each cell uses a fresh process, one unmeasured
warmup forecast and one measured warm forecast. FFTW imports immutable copies of
the corresponding matrix cell's wisdom with `WISDOM_ONLY`, so planning choices
stay fixed across rounds. PocketFFT warms its normal cache. Processes run
serially; no idle benchmark worker pools compete in the background. The runner
also retains the existing post-forecast microbenchmark and parity checks.

This separates the fast multicore comparison from a single library block's
position in time. It does not control unrelated host activity, thermal state or
OS scheduling. The five paired rounds are useful evidence for modest effects,
not a formal performance guarantee.

## Reproduction

Install the regular research environment and a local FFTW development package
(`brew install fftw` on these Macs), then:

```sh
.venv/bin/python scripts/benchmark_fft.py build
.venv/bin/python scripts/benchmark_fft.py prepare
.venv/bin/python scripts/benchmark_fft.py run --workers 1 4 16 --runs 5
```

For the follow-up after the matrix:

```sh
.venv/bin/python scripts/benchmark_fft_interleaved.py --workers 16 --rounds 5
```

`MOJO_BIN`, `CXX` and `FFTW_PREFIX` select existing installations. The experiment
currently supports Apple silicon macOS. All compiled providers, wisdom files,
reference arrays and temporary results go under `.build/fft-compare/`. The
public release's default CPU/GPU build remains independent of FFTW.

To reproduce exact input identity on another host, copy `fixture.npz` and use
`prepare --fixture PATH` before `run`. Build manifests retain source, compiler,
FFTW library and provider hashes. Receipts also retain process CPU time, cold
and warm planning time, plan descriptions and FFT microbenchmark samples.
