#!/bin/bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
attempt="${1:?Provide a fresh output directory}"
fixtures="${2:-$repo/build/nr_swin_mlp/fixtures}"
[[ ! -e "$attempt" ]] || { echo "Preserve existing attempt: $attempt" >&2;exit 2; }
mkdir -p "$attempt/build"
if metal_path="$(xcrun -f metal 2>/dev/null)"; then
  metallib_path="$(xcrun -f metallib)"
else
  # Use the installed toolchain without changing xcode-select or global state.
  metal_path=/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v17.6.109.0.ez7LwL/Metal.xctoolchain/usr/bin/metal
  metallib_path=/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v17.6.109.0.ez7LwL/Metal.xctoolchain/usr/bin/metallib
fi
[[ -x "$metal_path" && -x "$metallib_path" ]]
"$metal_path" -isysroot "$(xcrun --show-sdk-path)" -std=metal3.1 \
  -fno-fast-math -ffp-contract=off -c "$repo/src/nr_swin/swin32_mlp.metal" \
  -o "$attempt/build/mlp.air"
"$metallib_path" "$attempt/build/mlp.air" -o "$attempt/build/mlp.metallib"
xcrun -sdk macosx clang++ -std=c++20 -O2 -fobjc-arc -Wall -Wextra \
  -Werror -Wno-deprecated-declarations "$repo/src/nr_swin/mlp_probe.mm" \
  -framework Foundation -framework Metal -o "$attempt/build/probe"
for cta in 0 1 2 3 4 5 6 7 8; do
  for poison in a5 5a; do
    for ordinal in a b; do
      run="$attempt/cta${cta}-${poison}-${ordinal}"
      "$attempt/build/probe" "$attempt/build/mlp.metallib" \
        "$fixtures/cta$cta/inputs/input.e4m3" \
        "$fixtures/cta$cta/inputs/weight.bin" 64 "$poison" "$run" \
        >"$run.stdout.log" 2>"$run.stderr.log"
    done
  done
done
python3 "$repo/scripts/verify-nr-swin32-mlp.py" --fixtures "$fixtures" \
  --attempt "$attempt" --output "$attempt/verification.json"
