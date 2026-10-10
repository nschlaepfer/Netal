#!/bin/bash
set -euo pipefail

repository="$(cd "$(dirname "$0")/.." && pwd)"
attempt="${1:-$repository/build/fg_conv_hmma_u32/attempt-$(date -u +%Y%m%dT%H%M%SZ)}"
model="${2:-$repository/build/fg_default_readback/eval0-default-all-a5-a/runtime-default-buffers-post-evaluate-0}"
if [[ -e "$attempt" ]]; then
  echo "attempt exists; preserve it: $attempt" >&2
  exit 2
fi
mkdir -p "$attempt/build" "$attempt/runs"

xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math \
  -ffp-contract=off -I "$repository/src/fg_conv" \
  -I "$repository/src/fg_conv_simd_l20" \
  -c "$repository/src/fg_conv/conv_3x3.metal" \
  -o "$attempt/build/scalar.air"
xcrun -sdk macosx metallib "$attempt/build/scalar.air" \
  -o "$attempt/build/scalar.metallib"

xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math \
  -ffp-contract=off -I "$repository/src/fg_conv" \
  -I "$repository/src/fg_conv_simd_l20" \
  -include "$repository/src/fg_conv_simd_l20/hmma_exact_u32_override.metalh" \
  -c "$repository/src/fg_conv/conv_3x3.metal" \
  -o "$attempt/build/u32.air"
xcrun -sdk macosx metallib "$attempt/build/u32.air" \
  -o "$attempt/build/u32.metallib"

xcrun -sdk macosx clang++ -std=c++20 -O3 -ffp-contract=off \
  -Wall -Wextra -Werror \
  "$repository/src/fg_conv_simd_l20/test_u32_equivalence.cc" \
  -o "$attempt/build/test-u32-equivalence"
xcrun -sdk macosx clang++ -std=c++20 -O3 -fobjc-arc -Wall -Wextra \
  -Werror -Wno-unused-function -Wno-deprecated-declarations \
  "$repository/src/fg_conv/macos_3x3_probe.mm" \
  -framework Foundation -framework Metal -framework Security \
  -o "$attempt/build/probe"
xcrun -sdk macosx clang++ -std=c++20 -O3 -fobjc-arc -Wall -Wextra \
  -Werror -Wno-unused-function -Wno-deprecated-declarations \
  "$repository/src/fg_conv_simd_l20/benchmark.mm" \
  -framework Foundation -framework Metal -framework Security \
  -o "$attempt/build/benchmark"

"$attempt/build/test-u32-equivalence" \
  >"$attempt/u32-equivalence.stdout.log" \
  2>"$attempt/u32-equivalence.stderr.log"

for spec in l20:63 l24:67 l26:69 l28:71; do
  layer="${spec%%:*}"
  token="${spec##*:}"
  weights="$model/default-token-0000$token.bin"
  for poison in a5 5a; do
    for ordinal in a b; do
      run="$attempt/runs/$layer-$poison-$ordinal"
      "$attempt/build/probe" "$layer" "$attempt/build/u32.metallib" \
        "$weights" "$run" "$layer-$poison-$ordinal" "$poison" \
        >"$run.stdout.log" 2>"$run.stderr.log"
    done
  done
  "$attempt/build/benchmark" "$attempt/build/scalar.metallib" \
    "$attempt/build/u32.metallib" "$weights" \
    "$attempt/runs/$layer-benchmark.json" a5 fg_conv_3x3 "$layer" \
    >"$attempt/runs/$layer-benchmark.stdout.log" \
    2>"$attempt/runs/$layer-benchmark.stderr.log"
done

echo "attempt=$attempt"
