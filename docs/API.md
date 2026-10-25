# Python and C interfaces

`Forecast(backend="cpu", library=None, config=None)` loads a compiled library. Backend `gpu`
selects `steps_forecast_gpu_v2`, the complete Metal pipeline. CPU selects
`steps_forecast_cpu_v2`. Both accept runtime geometry/physical parameters. The GPU path fits
256/512-square models in native CPU float64 for parity; ensemble stages remain
on Metal. Library discovery
checks the backend environment override, `.build` in the working directory, then
`.build` beside an editable source package. An explicit path takes precedence.

`engine.forecast(history, velocity, leads, *, members=96, workers=1, seed=20260817, config=None)`
returns a newly allocated contiguous float64 array `[members, leads, N, N]`.
All calls through the Python boundary are serialized with one process-wide lock.
Other processes may run independently. Callers must not mutate inputs during a call.

- `history`: finite real values in dBZ, shape `[3,N,N]`, chronological order.
  The boundary converts to float64. The default no-rain / precipitation thresholds
  are −10 / 10 dBZ; configure other values explicitly. Input conversion/optical flow is the caller's responsibility.
- `velocity`: finite float32 or float64 array `[2,N,N]`, x then y,
  in pixels per configured input step. Preserve the dtype used for your reference.
- `members` and `leads`: positive integers up to the configured limits. Lead
  index zero is +one configured timestep. Defaults retain limits 96 / 14.
- `seed`: integer 0–2³²−1; NumPy RandomState-compatible chained ensemble streams.
- `workers`: positive integer, CPU member/readback parallelism. `1` is supported
  on both backends; this is not affinity or a process CPU quota.

NaN output marks locations outside the advected domain. Inputs must be finite.
Fewer than 16 pixels at or above the configured precipitation threshold over the entire history invokes latest-field
persistence (including sparse echoes), rather than pySTEPS' own dry-case policy.
A constant/degenerate rainy model raises `RuntimeError`. Unsupported input shape,
dtype, nonfinite input or option raises `ValueError`. GPU/runtime failures raise
`RuntimeError`. The native FFT bridge aborts on allocation/transform exceptions;
process-level recovery from out-of-memory is not provided.

## Runtime configuration

`ForecastConfig(size=192, kmperpixel=0.5, timestep=5.0, precip_thr=10.0,
norain_thr=-10.0, max_members=96, max_leads=14)` is immutable. Pass it to the
engine constructor as a default, or to each `forecast(..., config=...)` call.
Changing options or source between calls uses the same loaded library.

Supported square sizes are 64, 128, 192, 256 and 512 on both backends. Pixel
scale is 0.01–20 km and timestep 0.1–60 minutes; both must be finite. Thresholds
require −100 ≤ no-rain < precipitation ≤100 dBZ. Member/horizon limits can be
raised through 4096 / 288; these are API bounds, not memory-capacity promises or
validated weather-skill horizons. Output bytes are `members*leads*N*N*8`.
A 96-member, 14-lead 512-square output alone is 2.625 GiB; GPU scratch is larger.

Input history must already use the selected units and scan interval. Velocity
components follow array columns/rows, including row sign; they are pixels per
scan, not metres/second or raw radar Doppler velocity. The C boundary validates
options before selecting a compiled Mojo generic kernel. Rebuild old native
libraries when upgrading: the Python API rejects libraries without the v2 entry.

See [regional/source APIs](REGIONAL.md) for missing-data, quality, geometry and
local-product contracts.

## C ABI

[include/mojosteps.h](../include/mojosteps.h) declares both exports and a function
pointer type for `dlopen`. The retained legacy exports accept:

```
history*, velocity*, output*, members, leads, seed, workers, velocity_float32
```

In the legacy ABI arrays are contiguous native float64; sizes are fixed except member/lead
counts. Caller allocates output and retains all three disjoint buffers until the
synchronous call returns. `velocity_float32=1` requires values already rounded
to float32 and widened to float64. Flag 0 uses float64 semantics. Shape, capacity,
pointer validity and flag values are caller obligations. Serialize all native
calls across libraries/instances: direct C calls are not protected by the Python lock.
Status 0 = success, 1 = invalid dimensions/options, 2 = nonfinite/degenerate model,
3 = GPU/runtime failure. Do not consume output after failure.

The C example reads five little-endian float64 planes from the demo and writes
forecast planes. It assumes a little-endian host and uses the default seed and
float32 velocity semantics:

```sh
cc -O2 examples/forecast.c -o .build/forecast-c # add -ldl on Linux
.build/forecast-c "$PWD/.build/libmojosteps_cpu.dylib" cpu \
  .build/demo/input.f64 .build/demo/c-output.f64 4 3 1
.venv/bin/python scripts/check_c_output.py .build/demo
```

On Linux replace `.dylib` with `.so`; for Metal use the GPU library and `gpu`.
Python is used to generate inputs and compare outputs here, not by the C forecast.


The v2 function pointer `mojosteps_forecast_v2_fn` adds runtime geometry and units:

```
history*, velocity*, output*, size, members, leads, seed, workers,
velocity_float32, kmperpixel, timestep_minutes, precip_thr, norain_thr
```

Both `steps_forecast_cpu_v2` and `steps_forecast_gpu_v2` have this signature.
Arrays are `[3,N,N]`, `[2,N,N]`, `[members,leads,N,N]`. Caller capacity/lifetime,
disjoint-buffer, finite-input, motion-rounding and serialization rules above
still apply. The example optionally accepts `SIZE KM_PER_PIXEL TIMESTEP_MINUTES`
after its original arguments and selects the v2 entry; no Python forecast is
involved. `scripts/check_runtime_c.py` verifies both C and Python v2 calls.
