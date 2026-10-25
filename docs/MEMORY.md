# Forecast process RAM and GPU resource use

Measured 25 October 2026 on Apple M5 Ultra and M1 Ultra, using the same published
synthetic fixture: 96 members × 14 leads × 192 × 192 pixels, float64 output,
float32 motion, six cascades, seed 20260817. Mojo/MAX and pySTEPS versions match
the other research receipts. This measures the forecast process, not radar
loading, rendering or a full application replay.

**The CPU Mojo backend is the leanest configuration.** It peaks at approximately
0.42 GiB with one worker or 0.52–0.56 GiB with 16 workers. The GPU backend uses
roughly 2.6–2.9 GiB of total process memory, including about 2.33 GiB of Metal
resource allocations. The tested CPU-only pySTEPS path peaks at about 2.5–3.0 GiB.

## Peak process memory charge

Values below are **macOS physical-footprint lifetime peaks**, in GiB (2³⁰ bytes),
over one cold and one warm forecast in a fresh process per row. They include
Python, imported libraries, the observer, inputs, the runtime and one returned
output. They include charged graphics memory for the GPU run.

| Backend | CPU workers | M5 Ultra | M1 Ultra | Metal resource allocation peak |
| --- | ---: | ---: | ---: | ---: |
| mojoSTEPS CPU (PocketFFT) | 1 | 0.416 | 0.416 | CPU-only |
| mojoSTEPS CPU (PocketFFT) | 16 | 0.522 | 0.557 | CPU-only |
| mojoSTEPS GPU (Metal) | 1 | 2.845 | 2.621 | ~2.335 GiB on both |
| mojoSTEPS GPU (Metal) | 16 | 2.845 | 2.619 | ~2.335 GiB on both |
| pySTEPS 1.21.2 (NumPy FFT) | 1 | 3.016 | 2.822 | CPU-only |
| pySTEPS 1.21.2 (NumPy FFT) | 16 | 2.768 | 2.542 | CPU-only |

Apple silicon uses unified memory. These Macs have no separate VRAM pool: the
Metal allocation counter is a view of resources backed by the same shared RAM.
It overlaps with process memory accounting and must not be added to the RAM
column. The peaks of different counters can also occur at different times.

The physical-footprint counter is an OS memory charge, not a measurement of
whole-machine free-RAM reduction. It includes private/compressed/graphics
accounting and differs from RSS. Allocator reuse, object lifetimes, garbage
collection and compression influence observed peaks; memory is not necessarily
monotonic in the CPU worker limit. These short runs are not a capacity bound or
a long-running leak test.

## CPU-visible resident pages (RSS)

For comparison with `/usr/bin/time -l`, the process RSS lifetime peaks are:

| Backend | Workers | M5 Ultra | M1 Ultra |
| --- | ---: | ---: | ---: |
| mojoSTEPS CPU (PocketFFT) | 1 | 0.431 GiB | 0.430 GiB |
| mojoSTEPS CPU (PocketFFT) | 16 | 0.537 GiB | 0.571 GiB |
| mojoSTEPS GPU (Metal) | 1 | 0.815 GiB | 0.813 GiB |
| mojoSTEPS GPU (Metal) | 16 | 0.815 GiB | 0.811 GiB |
| pySTEPS 1.21.2 (NumPy FFT) | 1 | 2.992 GiB | 3.185 GiB |
| pySTEPS 1.21.2 (NumPy FFT) | 16 | 2.798 GiB | 2.595 GiB |

RSS alone misses much of the GPU backend's graphics allocation charge. Its
approximately 0.81 GiB RSS should therefore not be read as its total RAM demand.
The receipts retain both counter families, compression and graphics ledger
samples, before/after snapshots and values after deleting the output and GC.
Metal allocation falls below 1 MiB after each native GPU call in these runs;
CPU allocator/runtime memory can remain cached afterward.

## Why the implementations differ

The returned float64 forecast alone is exactly **396,361,728 bytes = 378 MiB**.
That output is included in every peak, and grows linearly with members × leads.
The CPU Mojo core writes directly into the final array and allocates member
working buffers for active workers. This keeps its peak relatively close to
the output floor.

The GPU implementation batches all 96 members and keeps large FFT and ensemble
buffers resident. Its paired-float32 representation uses eight bytes per real
value and sixteen per complex value. Static large-buffer sizes for this case
include:

| Buffer group | Logical storage |
| --- | ---: |
| Three complex cascade FFT buffers | 972 MiB |
| AR(2) ensemble state | 324 MiB |
| Normalized noise cascades | 162 MiB |
| GPU-visible paired-float final readback | 378 MiB |
| Complex input / row / spectrum buffers | 162 MiB |

Other fields, masks, sort keys/indices, motion and model-fitting buffers account
for the remaining Metal resources. The CPU float64 return array is a separate
378 MiB allocation. These buffer sizes explain the GPU memory tradeoff; this
study does not change allocation/reuse behavior.

The pySTEPS reference retains ensemble/cascade state and intermediate arrays and
collects/stack-copies output fields. Its flexible Python/NumPy execution path
uses more live/cached storage for this configuration. These observations apply
to the tested default NumPy-FFT reference path; other pySTEPS configurations,
streaming callbacks, output policies or FFT backends can change memory use.

## Measurement and validation

- One fresh process per backend/worker setting; two forecasts with the prior
  output deleted and Python GC run before the next call. Worker limits are 1
  and 16 for every backend on each host. All FFTs use one thread; the worker
  count controls ensemble parallelism, not OS affinity or a CPU quota.
- A small Objective-C++ observer reads `task_info(TASK_VM_INFO)` and samples
  `MTLDevice.currentAllocatedSize` approximately every millisecond. RSS and
  physical-footprint peaks are OS lifetime high-water counters; Metal is a
  sampled allocation peak, rather than an exact allocation trace.
- A separate 64 MiB Metal allocation control increases the resource counter by
  exactly 64 MiB and releases it on both hosts. The observer sees MAX's full
  device allocation during forecasts. CPU-only cases skip Metal device creation;
  their zero GPU-work requirement is established by the selected code path.
- Output hashes match the previously qualified full CPU/GPU/pySTEPS receipts for
  all **24 forecasts**. Spatial planes are hashed directly without allocating
  a whole-output byte copy. No reference forecast array or numerical-comparison
  temporaries are held in the measured process.
- Interpreter/library setup precedes the forecast sampling window but remains
  part of the process lifetime peak. The observer's own small overhead is also
  included. Timings in these receipts are memory-observed runs, not replacements
  for the latency benchmarks.

The raw measurements are curated in
[`m5-ultra.json`](../benchmarks/memory/m5-ultra.json) and
[`m1-ultra.json`](../benchmarks/memory/m1-ultra.json). The observer, runner,
fixture, library and native-source hashes are retained. Shared metadata is
factored out once per host; every call's samples and lifecycle snapshots remain.

Metric references: Apple's [unified-memory property](https://developer.apple.com/documentation/metal/mtldevice/hasunifiedmemory),
[Metal resource-byte counter](https://developer.apple.com/documentation/metal/mtldevice/currentallocatedsize),
and [XNU footprint-accounting notes](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/vm/vm_map_xnu.h).

## Reproduction

After the regular pinned setup and `scripts/build.py --backend all` on an Apple
silicon Mac, generate the same synthetic inputs with the demo, build the observer
and run the matrix:

```sh
.venv/bin/python -m mojosteps demo --backend cpu --no-plot --output .build/demo
.venv/bin/python scripts/benchmark_memory.py build
.venv/bin/python scripts/benchmark_memory.py run --workers 1 16 --calls 2 \
  --fixture .build/demo/forecast.npz --output .build/memory/results
```

Only the demo's history and velocity arrays are read. The full benchmark input
and output hashes are checked against the public qualification receipts. The
observer requires macOS Mach APIs and the Metal framework; this measurement
runner does not support Linux. Source, compiled observer and raw results use
`scripts/memory_probe/`, `.build/memory/` and `benchmarks/memory/`, respectively.
No probe or forecast binary is included in the source research release.
