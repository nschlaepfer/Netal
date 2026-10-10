#!/bin/bash
# GPU acceptance workload; reserve the Mac in NR_AGENT_COORDINATION.md first.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
attempt="${1:?Provide a fresh attempt directory}"
reference="${2:-$repo/build/nr_encoder22/reference-v1}"
[[ ! -e "$attempt" ]] || { echo 'Preserve prior attempt' >&2; exit 2; }
mkdir -p "$attempt"
python3 "$repo/scripts/generate-nr-prefix22-probe.py" --output "$attempt/source"
mkdir -p "$attempt/source/scripts" "$attempt/weights"
for name in qmma_fixed.metalh mlp_probe.mm wide_execute.h wide_boundaries.h native_encoder_exports.h native_encoder_exports.metal swin32_mlp.metal swin32_qkv.metal swin32_attention.metal swin32_block.metal swin32_chain.metal swin32_downsample.metal swin_wide.metal wide_boundaries.metal; do
  cp "$repo/src/nr_swin/$name" "$attempt/source/$name"
done
for name in test-nr-prefix22.sh verify-nr-encoder-prefix.py generate-nr-prefix22-probe.py build-nr-encoder-reference.py verify-nr-middle-view-abi.py; do
  cp "$repo/scripts/$name" "$attempt/source/scripts/$name"
done
for block in $(seq 1 22); do cp "$reference/weights/block$block.bin" "$attempt/weights/"; done
if metal_path="$(xcrun -f metal 2>/dev/null)"; then metallib_path="$(xcrun -f metallib)"; else
  metal_path=/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v17.6.109.0.ez7LwL/Metal.xctoolchain/usr/bin/metal
  metallib_path=/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v17.6.109.0.ez7LwL/Metal.xctoolchain/usr/bin/metallib
fi
"$metal_path" --version > "$attempt/metal-version.txt"
xcrun clang++ --version > "$attempt/clang-version.txt"
"$metal_path" -isysroot "$(xcrun --show-sdk-path)" -std=metal3.1 -fno-fast-math -ffp-contract=off -c "$attempt/source/prefix8.metal" -o "$attempt/prefix.air"
"$metallib_path" "$attempt/prefix.air" -o "$attempt/prefix.metallib"
for width in 128 256; do
  "$metal_path" -isysroot "$(xcrun --show-sdk-path)" -std=metal3.1 -fno-fast-math -ffp-contract=off "-DNR_WIDE_CHANNELS=$width" -c "$attempt/source/wide$width.metal" -o "$attempt/wide$width.air"
  "$metallib_path" "$attempt/wide$width.air" -o "$attempt/wide$width.metallib"
done
xcrun -sdk macosx clang++ -std=c++20 -O2 -fobjc-arc -Wall -Wextra -Werror -Wno-deprecated-declarations "$attempt/source/probe.mm" -framework Foundation -framework Metal -o "$attempt/probe"
for poison in a5 5a; do for ordinal in a b; do
  run="$attempt/$poison-$ordinal"
  "$attempt/probe" "$attempt/prefix.metallib" "$repo/build/nr_swin_block/reference-v2/input.e4m3" "$attempt/weights" "$poison" "$run" 22 "$attempt/wide128.metallib" "$attempt/wide256.metallib" > "$run.stdout.log" 2> "$run.stderr.log"
  echo "Completed $poison-$ordinal"
done; done
python3 "$repo/scripts/verify-nr-encoder-prefix.py" --attempt "$attempt" --reference "$reference" --last-block 22 --output "$attempt/verification.json"
