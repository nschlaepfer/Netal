#!/bin/zsh
# GPU conformance requires explicit local reference/candidate libraries. No model inputs.
set -euo pipefail
[[ $# -eq 3 ]] || { print -u2 'usage: test-netal-exposure-kernel.sh <reference.metallib> <candidate.metallib> <report.csv>'; exit 64; }
repository="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$repository/build/macos_host"
xcrun clang++ -fobjc-arc -std=c++20 -O2 -Wall -Wextra -Werror \
  -framework Foundation -framework Metal "$repository/src/macos_host/netal_exposure_probe.mm" \
  -o "$repository/build/macos_host/netal-exposure-probe"
"$repository/build/macos_host/netal-exposure-probe" "$1" "$2" "$3"
