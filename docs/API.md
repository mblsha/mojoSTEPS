# Python and C interfaces

`Forecast(backend="cpu", library=None)` loads a compiled library. Backend `gpu`
selects `steps_forecast_gpu_all`, the complete Metal pipeline. Library discovery
checks the backend environment override, `.build` in the working directory, then
`.build` beside an editable source package. An explicit path takes precedence.

`engine.forecast(history, velocity, leads, *, members=96, workers=1, seed=20260817)`
returns a newly allocated contiguous float64 array `[members, leads, 192, 192]`.
All calls through the Python boundary are serialized with one process-wide lock.
Other processes may run independently. Callers must not mutate inputs during a call.

- `history`: finite real values in dBZ, shape `[3,192,192]`, chronological order.
  The boundary converts to float64. Encode no rain as −10 dBZ; precipitation
  starts at 10 dBZ. Input conversion/optical flow is the caller's responsibility.
- `velocity`: finite float32 or float64 array `[2,192,192]`, x then y,
  in pixels per five-minute input step. Preserve the dtype used for your reference.
- `members`: integer 1–96; `leads`: integer 1–14. Lead zero is +5 minutes.
- `seed`: integer 0–2³²−1; NumPy RandomState-compatible chained ensemble streams.
- `workers`: positive integer, CPU member/readback parallelism. `1` is supported
  on both backends; this is not affinity or a process CPU quota.

NaN output marks locations outside the advected domain. Inputs must be finite.
Fewer than 16 pixels ≥10 dBZ over the entire history invokes latest-field
persistence (including sparse echoes), rather than pySTEPS' own dry-case policy.
A constant/degenerate rainy model raises `RuntimeError`. Unsupported input shape,
dtype, nonfinite input or option raises `ValueError`. GPU/runtime failures raise
`RuntimeError`. The native FFT bridge aborts on allocation/transform exceptions;
process-level recovery from out-of-memory is not provided.

## C ABI

[include/mojosteps.h](../include/mojosteps.h) declares both exports and a function
pointer type for `dlopen`. Both accept:

```
history*, velocity*, output*, members, leads, seed, workers, velocity_float32
```

All arrays are contiguous native float64; sizes are fixed except member/lead
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
