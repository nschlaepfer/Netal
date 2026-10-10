#!/usr/bin/env bash
set -euo pipefail

repository="$(cd "$(dirname "$0")/.." && pwd)"
attempt_root="${1:-$repository/build/fg_mid_l54/semantics/macos-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ -e "$attempt_root" ]]; then
  echo "Attempt exists; preserve it: $attempt_root" >&2
  exit 2
fi
source_root="$repository/src/fg_mid_l54"
build_root="$attempt_root/build"
mkdir -p "$build_root"
xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off \
  -c "$source_root/semantics_probe.metal" \
  -o "$build_root/semantics_probe.air" \
  >"$build_root/metal-compile.log" 2>&1
xcrun -sdk macosx metallib "$build_root/semantics_probe.air" \
  -o "$build_root/semantics_probe.metallib" \
  >"$build_root/metallib-link.log" 2>&1
xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc -Wall -Wextra -Werror \
  -Wno-unused-function "$source_root/macos_semantics_probe.mm" \
  -framework Foundation -framework Metal \
  -o "$build_root/fg-mid-l54-mac-semantics" \
  >"$build_root/host-compile.log" 2>&1
"$build_root/fg-mid-l54-mac-semantics" \
  "$build_root/semantics_probe.metallib" \
  "$attempt_root/dense-ex2-minus2-to0-u32.bin" \
  >"$attempt_root/stdout.log" 2>"$attempt_root/stderr.log"
echo "attempt_root=$attempt_root"
echo "result=pass"
