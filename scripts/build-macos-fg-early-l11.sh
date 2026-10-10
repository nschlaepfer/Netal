#!/usr/bin/env bash
set -euo pipefail

repository="$(cd "$(dirname "$0")/.." && pwd)"
attempt_root="${1:-$repository/build/fg_early_l11/macos/attempt-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ -e "$attempt_root" ]]; then
  echo "Attempt exists; preserve it: $attempt_root" >&2
  exit 2
fi
source_root="$repository/src/fg_early_l11"
build_root="$attempt_root/build"
mkdir -p "$build_root"
xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off \
  -c "$source_root/early_l11.metal" -o "$build_root/early_l11.air" \
  >"$build_root/metal-compile.log" 2>&1
xcrun -sdk macosx metallib "$build_root/early_l11.air" \
  -o "$build_root/early_l11.metallib" \
  >"$build_root/metallib-link.log" 2>&1
xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc -Wall -Wextra -Werror \
  "$source_root/macos_probe.mm" -framework Foundation -framework Metal \
  -o "$build_root/fg-early-l11-mac" \
  >"$build_root/host-compile.log" 2>&1
for poison in a5 5a; do
  for ordinal in a b; do
    run_id="l11-poison-$poison-process-$ordinal"
    run_root="$attempt_root/$run_id"
    mkdir -p "$run_root"
    "$build_root/fg-early-l11-mac" "$build_root/early_l11.metallib" \
      "$run_root" "$run_id" "$poison" \
      >"$run_root/stdout.log" 2>"$run_root/stderr.log"
  done
done
echo "attempt_root=$attempt_root"
echo "result=pass"
