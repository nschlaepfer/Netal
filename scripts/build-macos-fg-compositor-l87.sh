#!/usr/bin/env bash
set -euo pipefail

repository="$(cd "$(dirname "$0")/.." && pwd)"
attempt_root="${1:-}"
if [[ -z "$attempt_root" ]]; then
  attempt_root="$repository/build/fg_compositor_l87/macos/attempt-$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -e "$attempt_root" ]]; then
  echo "Attempt directory already exists; preserve it and choose a new path: $attempt_root" >&2
  exit 2
fi

source_root="$repository/src/fg_compositor_l87"
build_root="$attempt_root/build"
archive="$repository/evidence/checkpoints/2026-08-31-sm120-rcp-rom/external-evidence-sm120-rcp-rom.tar.gz"
rom="$build_root/sm120-rcp-correction-rom.bin"
mkdir -p "$build_root"

tar -xOf "$archive" work/g23-rcp-architecture/m3-rcp-rom/correction-rom.bin >"$rom"
rom_hash="$(shasum -a 256 "$rom" | awk '{print $1}')"
if [[ "$rom_hash" != "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750" ]]; then
  echo "SM120 RCP architecture ROM identity mismatch: $rom_hash" >&2
  exit 3
fi

xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off \
  -c "$source_root/compositor_l87.metal" \
  -o "$build_root/compositor_l87.air" \
  >"$build_root/metal-compile.log" 2>&1
xcrun -sdk macosx metallib "$build_root/compositor_l87.air" \
  -o "$build_root/compositor_l87.metallib" \
  >"$build_root/metallib-link.log" 2>&1
xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc -Wall -Wextra -Werror \
  -I"$source_root" "$source_root/macos_probe.mm" \
  -framework Foundation -framework Metal \
  -o "$build_root/fg-compositor-l87-mac" \
  >"$build_root/host-compile.log" 2>&1

run_case() {
  local run_id="$1"
  local poison="$2"
  local mode="$3"
  local run_root="$attempt_root/$run_id"
  mkdir -p "$run_root"
  "$build_root/fg-compositor-l87-mac" \
    "$build_root/compositor_l87.metallib" "$rom" "$run_root" \
    "$run_id" "$poison" "$mode" \
    >"$run_root/stdout.log" 2>"$run_root/stderr.log"
}

run_case poison-a5-process-a a5 full
run_case poison-a5-process-b a5 full
run_case poison-5a-process-a 5a full
run_case poison-5a-process-b 5a full
run_case sparse-mask-branch 3c sparse

echo "attempt_root=$attempt_root"
echo "rom_sha256=$rom_hash"
echo "result=pass"
