# Forecast architecture

This implementation ports the pySTEPS spatial STEPS path. There are no
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
The GPU backend uses native CPU float64 fitting for 256/512-square grids, then
retains fitted model and ensemble state on Metal until final
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

Grid-dependent functions and storage use actual Mojo `SIZE: Int` generic
parameters (`FFT[SIZE]`, `Model[SIZE]`, Metal kernels and resident orchestration).
One shared library instantiates 64/128/192/256/512-square paths. Its v2 C ABI
selects a path at runtime; `RuntimeOptions` carries physical pixel scale, timestep
and thresholds. BPS unit conversion and mask growth derive from those values.
The old 192-square C exports remain wrappers for compatibility.

The original mixed-radix 192 FFT is retained; the other GPU sizes use radix-2
FFTs. CDF merge-pass count derives from pixel count and stays even so final
buffers are deterministic. No runtime source generation, profile build or Python
callback occurs in a forecast. Native libraries grow to contain all kernels.

The optional Python `radar` and `regional` layers handle input conversion,
projection, exact-time priority mosaics, overlap tiles and local trajectory
products. Optical-flow estimation and scheduled/network acquisition remain
caller responsibilities. Forecast model choices (six Gaussian cascades, AR(2),
nonparametric noise, empirical CDF, incremental mask, BPS) remain fixed; this is
not a port of every pySTEPS method.


Large-grid scalar correlation accumulation originally changed CDF ranks of
nearly equal field values. Pairwise CPU reductions for N>=256 fix that drift.
The large-grid GPU path uses that float64 fit; its ensemble/noise, sorting,
masking and advection remain on Metal. The 192-square numerical ordering and
published default output hashes are retained. Four localized regression seeds
and complete 96-member 512-square comparisons cover the discovered failure.
