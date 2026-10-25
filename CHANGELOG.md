# Changelog

## Unreleased

- Faster Metal FFT butterflies, merge-path CDF sorting and per-member cutoffs;
  complete outputs retain the merged GPU hashes on M1/M5 Ultra. Reproducible
  interleaved benchmarks and opt-in GPU stage profiling.

- Runtime configurable size, resolution, scan interval, thresholds and bounds,
  using genuine Mojo generic CPU/Metal kernels in one library (64–512 square).
- Runtime C ABI plus retained legacy entrypoints; generic radix-2 Metal FFTs.
- Optional CF NetCDF/dBZ/rain-rate/palette adapters, exact-time priority mosaics,
  projection, provenance, coverage abstention and overlapping rectangular tiles.
- Shared regional ensembles, area probabilities and whole-window trajectory risk;
  synthetic multi-source example and CPU/Metal parity/source-contract tests.

## 0.1.0 — 2026-10-25

Initial source research release extracted from the standalone STEPS experiment.
Includes CPU and complete Apple Metal forecast cores, BSD PocketFFT CPU transforms,
NumPy-compatible RNG, Python and C boundaries, a synthetic moving-cell demo,
pySTEPS parity tests, Linux/macOS CPU CI, manual hardware GPU CI, reproducibility
receipts and retained dependency licenses. No native binary release is provided.
