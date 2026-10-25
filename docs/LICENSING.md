# Licensing and provenance

## Project source

The original mojoSTEPS code, documentation and synthetic generator are released
under [BSD-3-Clause](../LICENSE). The native port was extracted from an earlier
radar application experiment, with application/data/operations integration
removed. The public repository starts with a clean source snapshot; no radar
archive, credentials, downloaded SDK or previous repository history is included.
Generated synthetic preview/receipts are project outputs, also under BSD-3-Clause.

## Retained upstream material

| Material | License / retained notice | Use |
| --- | --- | --- |
| pySTEPS | BSD-3-Clause; `third_party/LICENSE.pysteps` | STEPS algorithms and adapted behavior |
| NumPy | BSD-3-Clause; `third_party/LICENSE.numpy` | RandomState-compatible algorithm port |
| RandomKit / MT19937 | BSD-3-Clause and MIT; `third_party/LICENSE.randomkit` | RNG, integer rejection, uniform and normal sampling |
| NumPy random | BSD-3-Clause option of dual BSD/NCSA; `third_party/LICENSE.numpy-random` | RNG provenance |
| PocketFFT | BSD-3-Clause; `third_party/pocketfft/LICENSE.md` | Unmodified vendored C++ header |

[NOTICE](../NOTICE) preserves attribution. PocketFFT's exact upstream commit and
file hashes are recorded in `third_party/pocketfft/UPSTREAM.json`. Additional RNG
notice URLs and hashes are recorded in `third_party/PROVENANCE.json`.
NumPy/SciPy/pySTEPS/matplotlib/Dask are installed externally, not vendored.
The optional radar adapters also install netCDF4, pyproj and Shapely externally.
Their packages and native dependencies retain their own terms; no such
dependency binaries or implementation sources are bundled in mojoSTEPS.

The default build replaces the prototype's FFTW bridge with PocketFFT. It does
not link FFTW or include its implementation. This avoids imposing FFTW's GPL
terms on a distributed linked forecast library. All project/retained component
license notices must travel with redistributions; the root license does not
replace upstream copyright or license terms.

The optional `scripts/benchmark_fft.py` experiment links a separately installed
FFTW library for local comparisons. Its API adapter is project-authored source;
FFTW itself is not vendored. FFTW's license applies to those linked benchmark
binaries, which stay under `.build/` and are not shipped. The default build,
Python wheel and CPU/GPU release libraries do not acquire an FFTW dependency.

## External Mojo/MAX toolchain

Mojo and MAX are not uniformly licensed. The compiler, standard library, GPU APIs
and runtime must be assessed by component/version. See the official
[Modular repository](https://github.com/modular/modular) and
[MAX terms](https://github.com/modular/modular/blob/main/Licenses/MAX_LICENSE).
The pinned MAX 26.7.0 wheel carries the **Modular MAX Community License, last
modified August 18, 2026**; newer repository terms may differ. Section 1.4 preserves
separately licensed components. The installed wheel's `LICENSE` and
`Third-Party-Notices` are authoritative for that artifact.

This release contains our kernels that interoperate with the public SDK APIs.
It does not vendor SDK implementation source, alter MAX, or distribute SDK/runtime
binaries. Installing the pinned toolchain is a separate action governed by its
terms. The BSD license of mojoSTEPS does not relicense MAX or make the whole
runtime stack OSI-licensed.

Native binaries/wheels, containers and hosted-service deployments need their own
review of the applicable runtime redistributables, notices and application
requirements before distribution. No such binaries or hosted service are part
of this source research release. The Python wheel contains the pure Python boundary and adapters;
it does not embed a native forecast library or runtime.
