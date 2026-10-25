# Contributing

Please open an issue or pull request with the configuration, input-generation
recipe, seed, backend, toolchain version and a small reproducible case. Do not
upload restricted radar data or credentials. Generated synthetic cases are preferred.

Follow README setup, build the affected backend, then run focused tests. At a
stable milestone run the CPU suite; GPU numerical changes also need real Metal
checks and a full 96 × 14 comparison with source hashes. Report unsupported or
untested hardware explicitly. Keep numerical tolerance, fixed configuration and
ownership/lifetime contracts visible; a faster result must still pass parity.

Add tests that distinguish the bug/failure from the intended behavior. Keep
`.build` disposable and retain original assets/source elsewhere. Do not commit
SDK/runtime binaries or prebuilt forecast libraries. Keep third-party notices.
Contributions are accepted under the project's BSD-3-Clause license with the
applicable retained upstream terms; contributors must have the right to submit them.
