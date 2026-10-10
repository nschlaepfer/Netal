#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
root="${1:-$repo/build/fg_resolution_preprocess_l22_l24/macos/attempt-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ -e "$root" ]]; then
  echo "Attempt exists: $root" >&2
  exit 2
fi
src="$repo/src/fg_resolution_preprocess_l22_l24"
build="$root/build"
mkdir -p "$build"
archive="$repo/evidence/checkpoints/2026-08-31-sm120-rcp-rom/external-evidence-sm120-rcp-rom.tar.gz"
rom="$build/rcp.bin"
tar -xOf "$archive" work/g23-rcp-architecture/m3-rcp-rom/correction-rom.bin >"$rom"
[[ "$(shasum -a 256 "$rom" | awk '{print $1}')" == 7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750 ]]
for name in module22 module24_level10 module24_level11; do
  xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off \
    -Wno-unused-variable -Wno-unused-function -c "$src/$name.metal" \
    -o "$build/$name.air" >"$build/$name-metal.log" 2>&1
done
xcrun -sdk macosx metallib \
  "$build/module22.air" "$build/module24_level10.air" \
  "$build/module24_level11.air" -o "$build/kernels.metallib" \
  >"$build/link.log" 2>&1
xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc -Wall -Wextra -Werror \
  -Wno-unused-function "$src/macos_probe.mm" -framework Foundation \
  -framework Metal -o "$build/probe" >"$build/host.log" 2>&1
for resolution in 720p 1080p 1440p 4k; do
  for phase in reset generated; do
    for poison in a5 5a; do
      for ordinal in a b; do
        id="$resolution-$phase-poison-$poison-process-$ordinal"
        dir="$root/$id"
        mkdir -p "$dir"
        "$build/probe" "$build/kernels.metallib" "$rom" "$dir" "$id" \
          "$poison" "$resolution" "$phase" \
          >"$dir/stdout.log" 2>"$dir/stderr.log"
      done
    done
  done
done
echo "attempt_root=$root"
echo result=pass
