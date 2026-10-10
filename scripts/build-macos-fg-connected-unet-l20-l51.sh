#!/usr/bin/env bash
set -euo pipefail

repository="$(cd "$(dirname "$0")/.." && pwd)"
attempt_root="${1:-}"
model_root="${2:-$repository/build/fg_default_readback/eval0-default-all-a5-a/runtime-default-buffers-post-evaluate-0}"
if [[ -z "$attempt_root" ]]; then
  attempt_root="$repository/build/fg_connected_unet_l20_l51/macos/attempt-$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -e "$attempt_root" ]]; then
  echo "Attempt directory already exists; preserve it: $attempt_root" >&2
  exit 2
fi

source_root="$repository/src/fg_connected_unet_l20_l51"
conv_root="$repository/src/fg_conv"
build_root="$attempt_root/build"
mkdir -p "$build_root"
xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off \
  -I "$conv_root" -c "$conv_root/conv_3x3.metal" \
  -o "$build_root/conv_3x3.air" >"$build_root/conv-3x3-compile.log" 2>&1
xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off \
  -I "$conv_root" -c "$conv_root/conv_1x1.metal" \
  -o "$build_root/conv_1x1.air" >"$build_root/conv-1x1-compile.log" 2>&1
xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off \
  -c "$repository/src/fg_pooling/pooling.metal" \
  -o "$build_root/pooling.air" >"$build_root/pooling-compile.log" 2>&1
xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off \
  -c "$repository/src/fg_upscale/upscale.metal" \
  -o "$build_root/upscale.air" >"$build_root/upscale-compile.log" 2>&1
xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off \
  -c "$repository/src/fg_elementwise/elementwise_family.metal" \
  -o "$build_root/elementwise.air" >"$build_root/elementwise-compile.log" 2>&1
xcrun -sdk macosx metallib \
  "$build_root/conv_3x3.air" "$build_root/conv_1x1.air" \
  "$build_root/pooling.air" "$build_root/upscale.air" \
  "$build_root/elementwise.air" -o "$build_root/unet.metallib" \
  >"$build_root/metallib-link.log" 2>&1
xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc \
  -Wall -Wextra -Werror -Wno-unused-function \
  "$source_root/macos_probe.mm" -framework Foundation -framework Metal \
  -o "$build_root/fg-connected-unet-l20-l51-mac" \
  >"$build_root/host-compile.log" 2>&1

for poison in a5 5a; do
  for ordinal in a b; do
    run_id="unet-poison-$poison-process-$ordinal"
    run_root="$attempt_root/$run_id"
    mkdir -p "$run_root"
    "$build_root/fg-connected-unet-l20-l51-mac" \
      "$build_root/unet.metallib" "$model_root" "$run_root" \
      "$run_id" "$poison" >"$run_root/stdout.log" 2>"$run_root/stderr.log"
  done
done

echo "attempt_root=$attempt_root"
echo "result=pass"
