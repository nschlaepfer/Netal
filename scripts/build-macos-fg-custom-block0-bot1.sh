#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
root="${1:-$repo/build/fg_conv/macos-custom-block0-bot1/attempt-$(date -u +%Y%m%dT%H%M%SZ)}"
[[ ! -e "$root" ]] || { echo "Attempt exists: $root" >&2; exit 2; }
src="$repo/src/fg_conv"
model="$repo/build/fg_default_readback/eval0-default-all-a5-a/runtime-default-buffers-post-evaluate-0"
build="$root/build"
mkdir -p "$build"
xcrun -sdk macosx metal -std=metal3.1 -I "$src" -c "$src/custom_block0_bot1.metal" -o "$build/kernel.air" >"$build/metal-compile.log" 2>&1
xcrun -sdk macosx metallib "$build/kernel.air" -o "$build/kernel.metallib" >"$build/metallib-link.log" 2>&1
xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc -Wall -Wextra -Werror -Wno-unused-function "$src/macos_custom_block0_bot1_probe.mm" -framework Foundation -framework Metal -o "$build/host" >"$build/host-compile.log" 2>&1
for poison in a5 5a; do
  for ordinal in a b; do
    run_id="block0-bot1-poison-$poison-process-$ordinal"
    run_root="$root/$run_id"
    mkdir -p "$run_root"
    "$build/host" "$build/kernel.metallib" "$model/default-token-000154.bin" "$model/default-token-000156.bin" "$run_root" "$run_id" "$poison" >"$run_root/stdout.log" 2>"$run_root/stderr.log"
  done
done
python3 "$repo/scripts/verify-fg-custom-block0-bot1.py" backend --backend mac --root "$root" --output "$root/backend-receipt.json"
echo "attempt_root=$root"
echo "result=pass"
