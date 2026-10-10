#!/usr/bin/env bash
set -euo pipefail

repository="$(cd "$(dirname "$0")/.." && pwd)"
attempt_root="${1:-$repository/build/fg_conv/macos-1x1/attempt-$(date -u +%Y%m%dT%H%M%SZ)}"
shift $(( $# > 0 ? 1 : 0 ))
layers=("$@")
if [[ ${#layers[@]} -eq 0 ]]; then layers=(l22 l30 l31 l34 l35 l38 l39 l42 l43 l46 l47 l50 l51); fi
if [[ -e "$attempt_root" ]]; then echo "Attempt exists: $attempt_root" >&2; exit 2; fi
source_root="$repository/src/fg_conv"; build_root="$attempt_root/build"; weights_root="$repository/build/fg_default_readback/eval0-default-all-a5-a/runtime-default-buffers-post-evaluate-0"
mkdir -p "$build_root"
xcrun -sdk macosx metal -std=metal3.1 -I "$source_root" -c "$source_root/conv_1x1.metal" -o "$build_root/conv_1x1.air" >"$build_root/metal-compile.log" 2>&1
xcrun -sdk macosx metallib "$build_root/conv_1x1.air" -o "$build_root/conv_1x1.metallib" >"$build_root/metallib-link.log" 2>&1
xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc -Wall -Wextra -Werror -Wno-unused-function "$source_root/macos_1x1_probe.mm" -framework Foundation -framework Metal -o "$build_root/fg-conv-1x1-mac" >"$build_root/host-compile.log" 2>&1
weight_token() {
  case "$1" in l22) echo 65;; l30) echo 73;; l31) echo 75;; l34) echo 77;; l35) echo 79;; l38) echo 81;; l39) echo 83;; l42) echo 85;; l43) echo 87;; l46) echo 89;; l47) echo 91;; l50) echo 93;; l51) echo 95;; *) return 2;; esac
}
for layer_id in "${layers[@]}"; do
  token="$(weight_token "$layer_id")"; weights="$weights_root/default-token-$(printf '%06d' "$token").bin"; layer_root="$attempt_root/$layer_id"; mkdir -p "$layer_root"
  for poison in a5 5a; do
    for ordinal in a b; do
      run_id="$layer_id-poison-$poison-process-$ordinal"; run_root="$layer_root/$run_id"; mkdir -p "$run_root"
      "$build_root/fg-conv-1x1-mac" "$layer_id" "$build_root/conv_1x1.metallib" "$weights" "$run_root" "$run_id" "$poison" >"$run_root/stdout.log" 2>"$run_root/stderr.log"
    done
  done
  python3 "$repository/scripts/verify-fg-conv-1x1-family.py" backend --backend mac --layer "$layer_id" --root "$layer_root" --output "$layer_root/backend-receipt.json" >"$layer_root/backend-verification.log"
done
echo "attempt_root=$attempt_root"; echo "layer_count=${#layers[@]}"; echo "result=pass"
