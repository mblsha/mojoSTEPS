# Forecast architecture

This implementation ports a fixed pySTEPS spatial STEPS path. There are no
Python imports or callbacks in the native forecast. Python is an optional array
boundary, synthetic generator, reference oracle and plotting environment.

The stages are history alignment → Gaussian cascade/noise-model fitting → AR(2)
coefficients → ensemble RNG and BPS draws → per-lead noise/cascade evolution →
recomposition → empirical CDF matching → incremental mask → advection.

| Module | Responsibility |
| --- | --- |
| `native_core.mojo` | CPU fitting, ensemble state and forecast ABI |
| `native_fft.mojo`, `fft_bridge.cpp` | Float64 transforms through BSD PocketFFT |
| `native_fields.mojo` | Alignment, interpolation, sorting, CDF and masks |
| `native_rng.mojo` | NumPy-compatible legacy MT19937 and distributions |
| `gpu_fft.mojo`, `gpu_noise.mojo` | Metal FFT/cascades and paired-float32 arithmetic |
| `gpu_model.mojo` | History alignment, spectral fitting and AR coefficients |
| `gpu_rng.mojo` | On-device state evolution and polar normals |
| `gpu_fields.mojo` | Device CDF sorting/matching, masks and advection |
| `gpu_resident.mojo` | GPU-resident forecast orchestration and complete ABI |
| `gpu_core.mojo` | Shared kernels and earlier experimental staging |
| `probe_native.mojo` | CPU primitive test exports |

CPU forecasts parallelize independent ensemble members through MAX. PocketFFT
is compiled with internal threading disabled to avoid nested oversubscription.
The GPU backend retains fitted model and ensemble state on Metal until final
readback. Input validation, buffer allocation, fixed scalar/twiddle construction,
kernel launches and final conversion to caller float64 still use CPU. Dry/sparse
persistence is a native CPU copy. Full output transfer is included in timings.

Metal has no native float64 path here. Paired float32 values retain approximately
48 significant bits; error can accumulate and comparisons are tolerance-based.
CDF ordering breaks paired-precision ties with pixel indices; NumPy's unstable
sort may order nonminimum ties differently. Arbitrary tied inputs are not a
universal parity guarantee. Float32 history-alignment semantics also matter:
constant no-rain regions must stay exact to preserve noise-model behavior.

The Mojo API currently uses untracked pointers for callbacks and device buffers.
Keep owner lifetimes explicit: apparently redundant post-callback owner uses in
source prevent early destruction. The GPU cached readback allocation is aligned
to 16 KiB pages. Do not simplify either mechanism without hardware checks.

Fixed grid size, thresholds, timestep, noise/mask choices and model order are
part of this release's contract. Optical flow, radar decoding, georeferencing,
real-time ingest and full pySTEPS method selection are outside this port.
