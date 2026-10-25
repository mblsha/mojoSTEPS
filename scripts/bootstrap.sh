#!/bin/sh
# SPDX-License-Identifier: BSD-3-Clause
set -eu
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2; exit 1; }
uv venv --python 3.12 --allow-existing .venv
if test "$(uname -s)" = Darwin; then
  # pySTEPS' optional motion extensions pass -fopenmp even though this demo
  # supplies motion directly. Apple clang does not implement that option.
  command -v brew >/dev/null || { echo 'For the macOS reference install, first install Homebrew and: brew install llvm libomp' >&2; exit 1; }
  research_brew_prefix=$(brew --prefix)
  research_cc=${CC:-$research_brew_prefix/opt/llvm/bin/clang}
  test -x "$research_cc" || { echo 'For the macOS reference install, run: brew install llvm libomp' >&2; exit 1; }
  CC="$research_cc" ARCHFLAGS='-arch arm64' \
    CFLAGS="${CFLAGS:-} -I$research_brew_prefix/opt/libomp/include" \
    LDFLAGS="${LDFLAGS:-} -L$research_brew_prefix/opt/libomp/lib -Wl,-rpath,$research_brew_prefix/opt/libomp/lib" \
    uv --no-config pip install --python .venv/bin/python -e '.[research,demo,radar]'
else
  uv --no-config pip install --python .venv/bin/python -e '.[research,demo,radar]'
fi
uv venv --python 3.12 --allow-existing .build/toolchain
# Keep the compiler/SDK isolated from the numerical reference environment.
uv --no-config pip install --python .build/toolchain/bin/python mojo==1.2.0 max==26.7.0
