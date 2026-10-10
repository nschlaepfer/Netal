#!/usr/bin/env bash
set -euo pipefail

repository="$(cd "$(dirname "$0")/.." && pwd)"
attempt_root="${1:-$repository/build/fg_conv/macos-l51/attempt-$(date -u +%Y%m%dT%H%M%SZ)}"
weights="${2:-$repository/build/fg_default_readback/eval0-default-all-a5-a/runtime-default-buffers-post-evaluate-0/default-token-000095.bin}"
if [[ -e "$attempt_root" ]]; then echo "Attempt exists: $attempt_root" >&2; exit 2; fi
if [[ "$(shasum -a 256 "$weights" | awk '{print $1}')" != "477c68ebace499e2e1ac331f214fae3e9fb4df815a9e7d8142a0f1fb414b340e" ]]; then echo "Weight hash mismatch" >&2; exit 2; fi
source_root="$repository/src/fg_conv"; build_root="$attempt_root/build"; mkdir -p "$build_root"
xcrun -sdk macosx metal -std=metal3.1 -I "$source_root" -c "$source_root/conv_l51.metal" -o "$build_root/conv_l51.air" >"$build_root/metal-compile.log" 2>&1
xcrun -sdk macosx metallib "$build_root/conv_l51.air" -o "$build_root/conv_l51.metallib" >"$build_root/metallib-link.log" 2>&1
xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc -Wall -Wextra -Werror -Wno-unused-function "$source_root/macos_l51_probe.mm" -framework Foundation -framework Metal -o "$build_root/fg-conv-l51-mac" >"$build_root/host-compile.log" 2>&1
for poison in a5 5a; do
  for ordinal in a b; do
    run_id="live-model-poison-$poison-process-$ordinal"; run_root="$attempt_root/$run_id"; mkdir -p "$run_root"
    "$build_root/fg-conv-l51-mac" "$build_root/conv_l51.metallib" "$weights" "$run_root" "$run_id" "$poison" >"$run_root/stdout.log" 2>"$run_root/stderr.log"
  done
done
python3 "$repository/scripts/verify-fg-conv-l51.py" backend --backend mac --root "$attempt_root" --output "$attempt_root/backend-receipt.json" >"$attempt_root/backend-verification.log"
shasum -a 256 "$attempt_root/backend-receipt.json" >"$attempt_root/backend-receipt.sha256"
echo "attempt_root=$attempt_root"; echo "result=pass"
