# Reproducing results

Start with the pinned setup and build in the README. Keep the reference and SDK
environments separate. Python 3.12 is used for qualification. Set these variables
for reference comparisons and timings:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MPLCONFIGDIR=.build/mpl
.venv/bin/python -m pytest -q
# On real Apple Metal hardware, after --backend all:
MOJOSTEPS_TEST_GPU=1 .venv/bin/python -m pytest -q
```

Hosted CI checks both CPU builds, API errors, worker invariance, a changed seed,
float32/float64 motion, dry/sparse persistence, RNG streams at four boundary
seeds, PocketFFT roundtrip, masks, and the C caller. GPU tests add 96-member seed
chains and CDF/mask boundary cases. GPU tests are explicitly skipped without
hardware opt-in; they are not counted as successful GPU checks in hosted CI.
The manual GPU workflow needs an owner-provisioned self-hosted runner labeled
`mojosteps-gpu`; none is provisioned automatically. Only dispatch it on trusted main.

For the full 49,545,216-value synthetic forecast:

```sh
.venv/bin/python -m mojosteps demo --backend cpu --members 96 --leads 14 \
  --workers 1 --runs 5 --reference-runs 1 --compare --no-plot --no-save-arrays \
  --output .build/benchmark-cpu
.venv/bin/python -m mojosteps demo --backend gpu --members 96 --leads 14 \
  --workers 1 --runs 5 --reference-runs 1 --compare --no-plot --no-save-arrays \
  --output .build/benchmark-gpu
```

A receipt records one cold call then median warm timings. Native timing includes
Python validation, allocation, model fitting, RNG, all leads, transfers and
cleanup inside the synchronous call. Input generation, hashing, comparisons and
plotting are outside timing. Each backend runs as a block, followed by a reference
block; this is a simple latency experiment, not an interleaved randomized study.
Reference timings use the same worker count. Short timings depend on host load,
thermal state, SDK, libraries and memory; speedups apply only to this workload.
`workers=1` does not enforce physical-core affinity or a process CPU quota.

The synthetic field uses SciPy shifts and smoothing of seeded NumPy RandomState
noise. Motion varies spatially and preserves the chosen dtype. The default seed
is 20260817; synthetic generation seed is 2468. The receipt contains hashes of
both inputs, native source and outputs, plus native library identity.
For a second case use `--seed 0 --velocity-dtype float64`.

Acceptance requires identical finite masks, maximum finite error ≤1e-5 dBZ and
identical 10/15/20 dBZ decisions after float32 conversion. Full float32 equality
is reported separately. This is tested evidence for specific fixtures/seeds,
not a proof across all possible radar fields or future SDK versions. Stochastic
nowcast skill and rare/pathological models require a broader research evaluation.

Before packaging:

```sh
.venv/bin/python -m build --outdir .build/dist
.venv/bin/python scripts/check_distribution.py .build/dist
```

The sdist contains native sources and vendored PocketFFT; the wheel contains the
Python boundary only. Both carry root and third-party notices. No SDK or compiled
native objects should appear in either artifact. CI uploads receipts/manifests,
not native libraries.


## Runtime grids and multi-source extension

```sh
# Source/geometry/time/quality contracts and native parity, all CPU sizes:
.venv/bin/python -m pytest -q
# Complete hardware matrix, including 96 x 14 x 512 x 512 forecasts:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python scripts/validate_regional.py --full --workers 16 \
  --output .build/regional-validation.json
.venv/bin/python scripts/check_runtime_c.py --backend all
.venv/bin/python examples/multi_source.py --backend gpu --size 256 --kmperpixel 1
```

The full validation runner defaults to CPU and real Metal; pass `--backends cpu`
on Linux. It compares the second native output for each setting, while recording
both initial and warm call timings. Full-output hashing/comparison is outside
timing and uses spatial planes to keep diagnostic memory bounded. The reference
uses the same requested worker count. Retained oracle arrays mean this runner is
not a memory benchmark. The matrix uses one forecast engine/library per backend
while changing runtime configurations.

[Regional qualification receipts](../benchmarks/regional/) record all 18 complete
output comparisons per host (36 total) and source/library/fixture identities.
All native and reference output hashes match between the two Macs, including
retained default 192-square outputs. The 71-test suite and Python-free runtime C
caller pass on both hosts. Source-adapter tests use only synthetic fields/NetCDF.
The four initially failing large-grid member seeds are retained as regressions.
These are numerical checks; no new operational source or forecast-skill claim is
qualified by them. See [regional API and boundaries](REGIONAL.md).
