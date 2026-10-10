#!/bin/bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
attempt="${1:?Provide a fresh output directory}"
reference="$repo/build/nr_swin_chain/reference-v1"
[[ ! -e "$attempt" ]] || { echo "Preserve existing attempt: $attempt" >&2;exit 2; }
mkdir -p "$attempt/source/src/nr_swin" "$attempt/source/scripts"
for name in qmma_fixed.metalh mlp_probe.mm chain_probe.mm swin32_mlp.metal swin32_qkv.metal swin32_attention.metal swin32_block.metal swin32_chain.metal; do
  cp "$repo/src/nr_swin/$name" "$attempt/source/src/nr_swin/$name"
done
for name in test-nr-swin-chain.sh verify-nr-swin-chain.py build-nr-swin-chain-reference.py build-nr-block1-buffer-reference.py; do
  cp "$repo/scripts/$name" "$attempt/source/scripts/$name"
done
if metal_path="$(xcrun -f metal 2>/dev/null)"; then
  metallib_path="$(xcrun -f metallib)"
else
  metal_path=/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v17.6.109.0.ez7LwL/Metal.xctoolchain/usr/bin/metal
  metallib_path=/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v17.6.109.0.ez7LwL/Metal.xctoolchain/usr/bin/metallib
fi
"$metal_path" --version >"$attempt/metal-version.txt"
xcrun clang++ --version >"$attempt/clang-version.txt"
"$metal_path" -isysroot "$(xcrun --show-sdk-path)" -std=metal3.1 \
  -fno-fast-math -ffp-contract=off -c "$attempt/source/src/nr_swin/swin32_chain.metal" \
  -o "$attempt/chain.air"
"$metallib_path" "$attempt/chain.air" -o "$attempt/chain.metallib"
xcrun -sdk macosx clang++ -std=c++20 -O2 -fobjc-arc -Wall -Wextra \
  -Werror -Wno-deprecated-declarations "$attempt/source/src/nr_swin/chain_probe.mm" \
  -framework Foundation -framework Metal -o "$attempt/probe"
for poison in a5 5a; do
  for ordinal in a b; do
    run="$attempt/$poison-$ordinal"
    "$attempt/probe" "$attempt/chain.metallib" \
      "$repo/build/nr_swin_block/reference-v2/input.e4m3" "$reference/weights" \
      "$poison" "$run" 3 >"$run.stdout.log" 2>"$run.stderr.log"
    echo "Completed $poison-$ordinal"
  done
done
python3 "$repo/scripts/verify-nr-swin-chain.py" --reference "$reference" \
  --attempt "$attempt" --output "$attempt/verification.json"
