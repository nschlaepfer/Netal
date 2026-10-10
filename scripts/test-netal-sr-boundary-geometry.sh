#!/bin/zsh
set -euo pipefail
repository="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$repository/build/sr-boundary-tests"
xcrun clang++ -std=c++17 -Wall -Wextra -Werror \
  "$repository/src/macos_host/netal_sr_boundary_geometry_test.cpp" \
  -o "$repository/build/sr-boundary-tests/geometry-tests"
"$repository/build/sr-boundary-tests/geometry-tests"
