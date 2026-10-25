#!/bin/sh
# SPDX-License-Identifier: BSD-3-Clause
set -eu
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2; exit 1; }
uv venv --python 3.12 --allow-existing .venv
uv --no-config pip install --python .venv/bin/python -e '.[research,demo]'
uv venv --python 3.12 --allow-existing .build/toolchain
# Keep the compiler/SDK isolated from the numerical reference environment.
uv --no-config pip install --python .build/toolchain/bin/python mojo==1.2.0 max==26.7.0
