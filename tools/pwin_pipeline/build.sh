#!/bin/zsh
# Build the standalone PWin stage harness and (optionally) a pipeline metallib.
# usage: tools/pwin_pipeline/build.sh [kernels.metal ...]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; repository="$(cd "$here/../.." && pwd)"
out="$repository/build/pwin_pipeline"; mkdir -p "$out"
xcrun clang++ -fobjc-arc -std=c++20 -O2 -Wall -framework Foundation -framework Metal "$here/harness.mm" -o "$out/harness"
for src in "$@"; do
  name="$(basename "${src%.metal}")"
  xcrun metal -std=metal4.0 -fno-fast-math -c "$src" -o "$out/$name.air"
  xcrun metallib "$out/$name.air" -o "$out/$name.metallib"
  print "built=$out/$name.metallib"
done
print "harness=$out/harness"
