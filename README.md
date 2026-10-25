# mojoSTEPS

[![CPU research checks](https://github.com/mblsha/mojoSTEPS/actions/workflows/ci.yml/badge.svg)](https://github.com/mblsha/mojoSTEPS/actions/workflows/ci.yml)

A standalone **Mojo/MAX forecast core** with runtime geometry and physical parameters for the
[pySTEPS](https://pysteps.github.io/) STEPS stochastic precipitation nowcast.
CPU and Apple Metal GPU backends execute the forecast without Python. A small
Python interface aligns multiple radar sources, answers regional/local queries and checks results
against pySTEPS 1.21.2. A [C example](examples/forecast.c) calls the same core.

**Research software.** This ports the spatial AR(2) STEPS model, rather than the full
pySTEPS library. It has numerical parity tests; forecast skill on real storms
and operational reliability require separate evaluation.

![Synthetic moving-cell demonstration](docs/images/synthetic-demo.png)

## Quick start

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/), Git,
a C++17 compiler (Xcode Command Line Tools on macOS), and a supported Mojo host:
Apple silicon macOS or Linux x86-64 with AVX2 / x86-64-v3.
For the macOS pySTEPS reference install, run `brew install llvm libomp` first. The GPU backend is
qualified on Apple M1 Ultra and M5 Ultra; CUDA/ROCm and Intel Macs are untested
by this project. See [hardware and installation](docs/INSTALL.md).

```sh
git clone https://github.com/mblsha/mojoSTEPS.git
cd mojoSTEPS
sh scripts/bootstrap.sh
.venv/bin/python scripts/build.py --backend cpu
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -m mojosteps demo --compare
```

Open `.build/demo/preview.png`. The demo writes synthetic input and forecast
arrays, a raw C input file, and a JSON receipt with timing, input/output hashes,
source identities and numerical errors. It uses no downloaded radar data.

On Apple silicon:

```sh
.venv/bin/python scripts/build.py --backend all
.venv/bin/python -m mojosteps demo --backend gpu --compare --output .build/demo-gpu
MOJOSTEPS_TEST_GPU=1 .venv/bin/python -m pytest -q
```

The GPU backend keeps RNG, FFT/cascades, ensemble evolution, CDF matching,
masking and advection on Metal, with a final output readback. Model fitting is
on Metal for 64/128/192-square grids and native CPU float64 for 256/512 to
preserve large-grid numerical parity. CPU
validation, allocation, launch scheduling and final conversion remain. `workers=1`
is supported; it limits application parallelism, not OS scheduling or driver threads.

## Supported configuration

| Contract | Value |
| --- | --- |
| Input | Three N × N dBZ fields; N = 64, 128, 192, 256 or 512 |
| Motion | Two N × N fields: x/y pixels per configured scan |
| Ensemble / horizon | Defaults 1–96 members / 1–14 leads; configurable limits through 4096 / 288 |
| Scale | Runtime pixel scale and scan interval; defaults 0.5 km / 5 min |
| Model | Six Gaussian cascades; nonparametric noise; AR(2) |
| Postprocessing | Incremental mask; empirical CDF matching |
| Motion perturbation | BPS |
| Rain threshold / no rain | Runtime thresholds; defaults 10 / −10 dBZ |
| Arithmetic | CPU float64; GPU paired float32 (~48 significant bits) |
| Output | float64 [member, lead, y, x]; NaN outside advected domain |

```python
from mojosteps import Forecast
from mojosteps.synthetic import rain_cells
history, velocity = rain_cells()
forecast = Forecast("cpu").forecast(history, velocity, 14, members=96, workers=1)
```

Input motion dtype matters: float32 and float64 follow the corresponding pySTEPS
interpolation semantics. Fewer than 16 rain pixels across all history fields
uses latest-field persistence. Degenerate models are rejected. See [API](docs/API.md)
for validation, units, thread safety, memory and the C ABI.

## Multi-source regional forecasting

One library contains Mojo generic kernels for all five grid sizes. Grid selection,
resolution, scan interval and thresholds are runtime arguments: changing a radar
source does not require rebuilding. The native forecast core remains Python-free.

```python
from mojosteps import Forecast, ForecastConfig
engine = Forecast("gpu")
config = ForecastConfig(size=512, kmperpixel=1.0, timestep=5.0)
# history: [3,512,512] dBZ; velocity: [2,512,512] pixels per scan
forecast = engine.forecast(history, velocity, 14, members=96, config=config)
```

The optional `radar` package extra provides projected-grid/CF NetCDF adapters.
The [regional API](docs/REGIONAL.md) accepts dBZ, explicit Z–R rain-rate conversions
and explicit PNG colour mappings; aligns sources to one grid; selects by priority
at exact observation times; and serves multiple area probabilities from one
ensemble. Missing coverage abstains. Rectangular domains use overlapping tiles.

Run the synthetic two-source / three-location example after bootstrap/build:

```sh
.venv/bin/python examples/multi_source.py --backend gpu --size 256 --kmperpixel 1
```

## Evidence and reproducibility

The acceptance contract is identical finite masks, ≤ `1e-5 dBZ` maximum error
on finite values, and identical decisions at 10/15/20 dBZ after float32 conversion.
This is numerical agreement, not a promise of bitwise equality to pySTEPS.
[Reproduction commands and benchmark methodology](docs/REPRODUCIBILITY.md)
separate hosted CPU CI from real-hardware GPU checks. Public receipts use only
the included synthetic generator. See [benchmark results](docs/BENCHMARKS.md) and the
[matched FFTW/PocketFFT comparison](docs/FFT_COMPARISON.md).
The [GPU optimization study](docs/GPU_OPTIMIZATION.md) compares the merged and
optimized kernels, with reproducible stage profiling and cache-counter limits.
[Measured RAM and Metal resource use](docs/MEMORY.md) covers the same full workload.

CPU CI builds and tests on Linux and macOS. GPU CI is an opt-in manual workflow
for an owner-provisioned Apple runner; hosted CPU success does not certify GPU
execution. The included hardware receipts document actual Metal runs.

Read the [architecture](docs/ARCHITECTURE.md), [contributing guide](CONTRIBUTING.md)
and [changelog](CHANGELOG.md). To cite this software, use [CITATION.cff](CITATION.cff)
and also cite the [pySTEPS paper](https://doi.org/10.5194/gmd-12-4185-2019).

## License

mojoSTEPS source is **BSD-3-Clause**, with retained pySTEPS, NumPy/RandomKit and
PocketFFT notices. The CPU FFT bridge uses BSD PocketFFT; FFTW is not required.
**Mojo/MAX remains an external dependency under its own terms.** This release
ships source, not SDK/runtime binaries or prebuilt native wheels. Read
[LICENSE](LICENSE), [NOTICE](NOTICE) and [dependency licensing](docs/LICENSING.md).
