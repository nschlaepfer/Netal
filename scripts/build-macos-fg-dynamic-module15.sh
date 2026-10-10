#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
root="${1:-$repo/build/fg_dynamic_module15/macos/attempt-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ -e "$root" ]]; then echo "Attempt exists: $root" >&2; exit 2; fi
src="$repo/src/fg_dynamic_module15"; build="$root/build"; mkdir -p "$build"
xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off \
  -c "$src/dynamic_module15.metal" -o "$build/kernel.air" \
  >"$build/metal.log" 2>&1
xcrun -sdk macosx metallib "$build/kernel.air" -o "$build/kernel.metallib" \
  >"$build/link.log" 2>&1
xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc -Wall -Wextra -Werror \
  -Wno-unused-function "$src/macos_probe.mm" -framework Foundation \
  -framework Metal -o "$build/probe" >"$build/host.log" 2>&1
for resolution in 720p 1080p 1440p 4k; do
  for phase in reset generated; do
    for poison in a5 5a; do
      for ordinal in a b; do
        id="$resolution-$phase-poison-$poison-process-$ordinal"; dir="$root/$id"
        mkdir -p "$dir"; artifact_mode=hash-only
        if [[ "$poison" == a5 && "$ordinal" == a ]]; then artifact_mode=full; fi
        "$build/probe" "$build/kernel.metallib" "$dir" "$id" "$poison" \
          "$resolution" "$phase" "$artifact_mode" \
          >"$dir/stdout.log" 2>"$dir/stderr.log"
      done
    done
  done
done
echo "attempt_root=$root"
echo result=pass
