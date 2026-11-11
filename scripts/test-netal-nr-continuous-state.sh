#!/bin/zsh
# Headless unit and AddressSanitizer verification for ContinuousAdapterConfig and ContinuousAdapterState.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$script_dir/.." && pwd)"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/netal-nr-continuous-state.XXXXXX")"
trap 'rm -rf "$tmp_dir"' EXIT

xcrun --sdk macosx clang++ -arch arm64 -std=c++20 -O1 -g \
  -fsanitize=address -fsanitize-address-use-after-scope \
  -Wall -Wextra -Werror \
  "$repo/src/macos_host/netal_nr_continuous_state_test.cpp" \
  -o "$tmp_dir/continuous-state-test"

"$tmp_dir/continuous-state-test"
