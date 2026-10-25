# Installation and hardware

The tested compiler is Mojo 1.2.0 (2016d0cd), with MAX 26.7.0. The numerical
reference is Python 3.12, NumPy 2.5.1, SciPy 1.18.0 and pySTEPS 1.21.2.
`scripts/bootstrap.sh` installs these into separate `.venv` (Python research)
and `.build/toolchain` (compiler/SDK) environments using uv. Its explicit pins
and `--no-config` isolate setup from machine-specific uv configuration.
Dask 2026.8.0 enables pySTEPS worker parallelism; the native core does not use Dask.

| Platform | CPU | GPU |
| --- | --- | --- |
| Apple silicon macOS | Supported; tested M1 Ultra and M5 Ultra | Metal; tested M1 Ultra and M5 Ultra |
| Linux x86-64-v3 / AVX2 | Hosted CI | No qualified backend |
| Intel macOS / older x86 CPUs | Unsupported by the pinned toolchain | Unsupported |
| Other NVIDIA/AMD devices | Not qualified | Not qualified |

The GPU source depends on the pinned MAX GPU API and Metal double-float kernels.
Its build script deliberately permits only Apple silicon. Two float32 components
represent each numerical value; this is not native Metal float64 arithmetic.
Older AMD GCN hardware is not a supported route to this backend.

Build `--backend cpu`, `gpu` or `all`. Libraries and compiler manifests go into
`.build/`; no root install, Homebrew FFT package or FFTW is needed. `MOJO_BIN`
and `CXX` can select an existing compiler. To load a library from another
location, use `Forecast(library="/absolute/path/...")` or
`MOJOSTEPS_CPU_LIBRARY` / `MOJOSTEPS_GPU_LIBRARY`.

The Python wheel contains the array boundary and demo utilities only. Build the
native source from a checkout or sdist; installing the wheel alone does not
install a Mojo runtime or native forecast library. Packaging remains source-only
until platform/runtime distribution requirements are handled explicitly.

A full 96 × 14 forecast returns 396,361,728 bytes (about 378 MiB); GPU state,
reference arrays and comparison temporaries need additional memory. Start with
the four-member, three-lead demo. The full benchmark was tested on 128+ GiB
unified-memory Macs. This is not a measured minimum-memory requirement.

If compilation fails after upgrading Mojo, reproduce with the pinned version
before reporting an issue. Compiler warnings from API deprecations are currently
expected. GPU failures are errors and do not silently select CPU execution.
