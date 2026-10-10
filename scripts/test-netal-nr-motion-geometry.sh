#!/bin/zsh
# CPU-only reference tests; no Metal device, model execution or game launch.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
out="${1:?new evidence directory required}"
[[ ! -e "$out" ]] || { print -u2 'preserve existing evidence'; exit 2; }
mkdir -p "$out/source"
out="$(cd "$out" && pwd)"
for name in netal_nr_aspect.h netal_nr_motion_geometry.h netal_nr_motion_geometry_test.cpp; do
  cp -p "$repo/src/macos_host/$name" "$out/source/$name"
done
cp -p "$0" "$out/test-as-run.sh"
xcrun clang++ --version > "$out/compiler.txt"
xcrun clang++ -std=c++20 -O1 -g -Wall -Wextra -Werror \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  "$out/source/netal_nr_motion_geometry_test.cpp" -o "$out/motion-geometry-test"
ASAN_OPTIONS=detect_stack_use_after_return=1 UBSAN_OPTIONS=halt_on_error=1 \
  "$out/motion-geometry-test" > "$out/result.txt" 2>&1
cat "$out/result.txt"
shasum -a 256 "$out"/source/* "$out/motion-geometry-test" "$out/compiler.txt" \
  "$out/result.txt" "$out/test-as-run.sh" > "$out/sha256.txt"
