#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
root="${1:-$repo/build/fg_dynamic_module13/macos/attempt-$(date -u +%Y%m%dT%H%M%SZ)}"
[[ ! -e "$root" ]]; mkdir -p "$root/build"
src="$repo/src/fg_dynamic_module13"
archive="$repo/evidence/checkpoints/2026-08-31-sm120-rcp-rom/external-evidence-sm120-rcp-rom.tar.gz"
tar -xOf "$archive" work/g23-rcp-architecture/m3-rcp-rom/correction-rom.bin >"$root/build/rom.bin"
xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off -c "$src/dynamic_module13.metal" -o "$root/build/k.air"
xcrun -sdk macosx metallib "$root/build/k.air" -o "$root/build/k.metallib"
xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc -Wno-unused-function "$src/macos_probe.mm" -framework Foundation -framework Metal -o "$root/build/probe"
for resolution in 720p 1080p 1440p 4k; do for phase in reset generated; do for scenario in equal-high-tie priority-collision; do for poison in a5 5a; do for ordinal in a b; do
  id="$resolution-$phase-$scenario-poison-$poison-process-$ordinal"; mkdir -p "$root/$id"; mode=hash-only
  [[ "$poison$ordinal" == a5a ]] && mode=full
  "$root/build/probe" "$root/build/k.metallib" "$root/build/rom.bin" "$root/$id" "$id" "$poison" "$resolution" "$phase" "$scenario" "$mode" >"$root/$id/stdout.log" 2>"$root/$id/stderr.log"
done; done; done; done; done
echo result=pass
