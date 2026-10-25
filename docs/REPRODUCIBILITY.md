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
