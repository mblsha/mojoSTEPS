# GPU optimization study

Measured 25 October 2026 against merged main commit `0063190`, using the same
synthetic inputs, seed 20260817, 96 ensemble members, 14 leads and 16 CPU workers.
The native forecast remains Python-free. Model fitting still uses CPU float64
at 256/512 square to preserve the existing parity contract.

## Complete forecast latency

These are medians of five warm calls per implementation, except M1 192-square
which uses a balanced six-call repeat after a variable first block. Both blocks
are retained in the receipts. Baseline and candidate
alternate AB/BA within each shape. Timings include model fitting, allocation,
GPU execution, readback, final float64 conversion and the NumPy call boundary.
Input generation, hashing and comparisons are outside the timers. Each timed
result must reproduce its implementation's complete output hash. No previous
forecast output is retained during a timed call.

| Host | Grid | Merged baseline | Optimized | Speedup |
| --- | --- | ---: | ---: | ---: |
| koubou5, M5 Ultra | 192 × 192 | 0.315 s | 0.237 s | 1.33× |
| koubou5, M5 Ultra | 512 × 512 | 3.639 s | 2.180 s | 1.67× |
| koubou1, M1 Ultra | 192 × 192 | 0.731 s | 0.444 s | 1.65× |
| koubou1, M1 Ultra | 512 × 512 | 10.640 s | 5.022 s | 2.12× |

The first M1 192-square block ranged from 0.861–1.275 s for baseline and
0.550–0.744 s for candidate. Its repeat ranged from 0.730–0.749 s and
0.442–0.463 s respectively; the table uses that balanced repeat, rather than
combining measurements from visibly different host conditions.

Raw timings, input/output hashes, library identities and source manifests are in
[`benchmarks/gpu-optimization/`](../benchmarks/gpu-optimization/). Host load,
frequency and thermal conditions were not controlled; these are matched workload
measurements, not a general hardware ranking. The recorded Git revision is the
base revision; the candidate manifest records the exact working source hashes.

## Changes

- **Merge-path sorting:** each lane partitions a merge once, emits four results,
  then stages them in shared memory for consecutive global writes. This replaces
  a binary search and scattered global write for every individual key. Ordering
  remains paired-float value followed by original pixel index. The 192-square
  partial final merge and member boundaries are explicitly tested.
- **FFT butterflies:** one lane computes both outputs of a butterfly, sharing
  its paired-float complex multiplication. The addition order, twiddles and
  transforms remain the same.
- **CDF cutoff:** compute the target cutoff once per member per lead, instead of
  repeating the same binary search and interpolation at every pixel. The
  per-pixel kernel reads that cutoff and writes the matched value.

A separate tiled FFT transpose was tried and rejected: additional kernels and
memory passes increased latency. Its complete outputs still matched the
baseline. That exploration is not used for the final timing table.

## Cache stalls: what was measured

The baseline stage profile locates most GPU time in FFTs and sorting. For the
512-square warm call, those stages together took approximately 2.34 s on M5 and
8.44 s on M1. These synchronized stage totals are diagnostic, not an additive
model of an asynchronously scheduled production forecast.

The sorting change reduces redundant memory searches and makes writes
contiguous. The FFT change reduces duplicate arithmetic. Their speedups do
**not** establish a cache-miss or cache-stall percentage, nor do they isolate
whether cache latency, bandwidth, barriers or arithmetic throughput dominated.

Both machines' public `MTLDevice.counterSets` inventories expose only the
`GPUTimestamp` counter. Metal System Trace was recorded on koubou5, including
an explicit Metal GPU Counters recording. The exported profiles contained no
usable cache-stall samples. This does not establish that Instruments cannot
expose additional counters with a different configuration/toolchain. Hardware
cache-stall attribution remains unmeasured.

## Correctness

Complete 192- and 512-square outputs are bit-for-bit identical to the merged GPU
baseline on both hosts, including NaN locations. Their original published
output hashes are retained. This preserves the earlier complete-array pySTEPS
qualification; bitwise equality to pySTEPS itself is not promised.

All **86 tests pass on each Mac**, and the Python-free CPU/GPU C caller matches
the Python boundary on both. The real-Metal suite adds 15 direct sorting cases across all five sizes,
including duplicate values, paired-float low components and multiple members.
Existing CDF tests for empty/full/edge/tied fields now exercise the production
merge and precomputed-cutoff path. Runtime tests also compare forecasts against
pySTEPS at all grid sizes, alternate physical parameters and the four recorded
large-grid rank-regression seeds.

## Reproduce

Before changing source, preserve the merged baseline. On macOS a copied dylib
retains its original install name: give the copy a distinct identity, or dyld
may silently load the same code for both paths. The benchmark checks that the
two native entrypoint addresses differ and rejects a reused library.

```sh
mkdir -p .build/gpu-perf
cp .build/libmojosteps_gpu.dylib .build/gpu-perf/baseline.dylib
install_name_tool -id "$PWD/.build/gpu-perf/baseline.dylib" \
  .build/gpu-perf/baseline.dylib
codesign --force --sign - .build/gpu-perf/baseline.dylib
# Apply the optimization source, then build the ordinary library.
.venv/bin/python scripts/build.py --backend gpu
OMP_NUM_THREADS=1 .venv/bin/python scripts/benchmark_gpu.py \
  --baseline .build/gpu-perf/baseline.dylib \
  --candidate .build/libmojosteps_gpu.dylib --repeats 5 \
  --output .build/gpu-perf/comparison.json
MOJOSTEPS_TEST_GPU=1 .venv/bin/python -m pytest -q
```

To profile stages, build a separate library. The default build does not execute
the timing synchronizations. The profiler reports aggregate seconds across all
leads, plus setup and final conversion; cleanup can account for the difference
between their sum and the complete call time.

```sh
.venv/bin/python scripts/build.py --backend gpu --profile
.venv/bin/python scripts/profile_gpu.py --sizes 192 512
```

To inventory public Metal counters:

```sh
xcrun clang++ -fobjc-arc scripts/gpu_probe/counters.mm \
  -framework Foundation -framework Metal -o .build/gpu-perf/counters
.build/gpu-perf/counters
```
