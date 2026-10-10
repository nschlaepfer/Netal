#!/bin/zsh
set -euo pipefail
repository="$(cd "$(dirname "$0")/.." && pwd)"
out="$repository/build/sr-control-tests"
mkdir -p "$out"
xcrun clang++ -std=c++17 -Wall -Wextra -Werror \
  "$repository/src/macos_host/netal_sr_control_policy_test.cpp" -o "$out/control-tests"
"$out/control-tests"
