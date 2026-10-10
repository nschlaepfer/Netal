#!/bin/bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
attempt="${1:?Provide a fresh attempt directory}"
[[ ! -e "$attempt" ]] || { echo 'Preserve prior attempt' >&2;exit 2; }
mkdir -p "$attempt/source" "$attempt/weights"
python3 "$repo/scripts/generate-nr-prefix5-probe.py" --output "$attempt/source/generated"
for name in qmma_fixed.metalh mlp_probe.mm wide_execute.h swin32_mlp.metal swin32_qkv.metal swin32_attention.metal swin32_block.metal swin32_chain.metal swin32_downsample.metal swin_wide.metal; do
  cp "$repo/src/nr_swin/$name" "$attempt/source/generated/$name"
done
for name in generate-nr-prefix5-probe.py test-nr-prefix5.sh verify-nr-prefix5.py; do cp "$repo/scripts/$name" "$attempt/source/$name"; done
for block in 1 2 3 4; do cp "$repo/build/nr_swin_downsample/reference-v1/weights/block$block.bin" "$attempt/weights/"; done
cp "$repo/build/nr_swin64/reference-v1/weight.bin" "$attempt/weights/block5.bin"
if metal_path="$(xcrun -f metal 2>/dev/null)"; then metallib_path="$(xcrun -f metallib)"; else
  metal_path=/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v17.6.109.0.ez7LwL/Metal.xctoolchain/usr/bin/metal
  metallib_path=/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v17.6.109.0.ez7LwL/Metal.xctoolchain/usr/bin/metallib
fi
"$metal_path" -isysroot "$(xcrun --show-sdk-path)" -std=metal3.1 -fno-fast-math -ffp-contract=off -c "$attempt/source/generated/prefix5.metal" -o "$attempt/prefix.air"
"$metallib_path" "$attempt/prefix.air" -o "$attempt/prefix.metallib"
xcrun -sdk macosx clang++ -std=c++20 -O2 -fobjc-arc -Wall -Wextra -Werror -Wno-deprecated-declarations "$attempt/source/generated/probe.mm" -framework Foundation -framework Metal -o "$attempt/probe"
for poison in a5 5a; do for ordinal in a b; do
  run="$attempt/$poison-$ordinal"
  "$attempt/probe" "$attempt/prefix.metallib" "$repo/build/nr_swin_block/reference-v2/input.e4m3" "$attempt/weights" "$poison" "$run" 5 >"$run.stdout.log" 2>"$run.stderr.log"
  echo "Completed $poison-$ordinal"
done;done
python3 "$repo/scripts/verify-nr-prefix5.py" --attempt "$attempt" --output "$attempt/verification.json"
