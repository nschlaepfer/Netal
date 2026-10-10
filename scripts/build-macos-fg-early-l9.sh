#!/usr/bin/env bash
set -euo pipefail

repository="$(cd "$(dirname "$0")/.." && pwd)"
attempt_root="${1:-$repository/build/fg_early_l9/macos/attempt-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ -e "$attempt_root" ]]; then
  echo "Attempt exists; preserve it: $attempt_root" >&2
  exit 2
fi
source_root="$repository/src/fg_early_l9"
build_root="$attempt_root/build"
archive="$repository/evidence/checkpoints/2026-08-31-sm120-rcp-rom/external-evidence-sm120-rcp-rom.tar.gz"
rom="$build_root/sm120-rcp-correction-rom.bin"
mkdir -p "$build_root"
tar -xOf "$archive" work/g23-rcp-architecture/m3-rcp-rom/correction-rom.bin >"$rom"
if [[ "$(shasum -a 256 "$rom" | awk '{print $1}')" != "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750" ]]; then
  echo "RCP ROM identity mismatch" >&2
  exit 3
fi
xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off \
  -c "$source_root/early_l9.metal" -o "$build_root/early_l9.air" \
  >"$build_root/metal-compile.log" 2>&1
xcrun -sdk macosx metallib "$build_root/early_l9.air" \
  -o "$build_root/early_l9.metallib" \
  >"$build_root/metallib-link.log" 2>&1
xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc -Wall -Wextra -Werror \
  -Wno-unused-function "$source_root/macos_probe.mm" \
  -framework Foundation -framework Metal \
  -o "$build_root/fg-early-l9-mac" \
  >"$build_root/host-compile.log" 2>&1
for poison in a5 5a; do
  for ordinal in a b; do
    run_id="l9-poison-$poison-process-$ordinal"
    run_root="$attempt_root/$run_id"
    mkdir -p "$run_root"
    "$build_root/fg-early-l9-mac" "$build_root/early_l9.metallib" "$rom" \
      "$run_root" "$run_id" "$poison" \
      >"$run_root/stdout.log" 2>"$run_root/stderr.log"
  done
done
echo "attempt_root=$attempt_root"
echo "result=pass"
