#!/usr/bin/env bash
set -euo pipefail

repository="$(cd "$(dirname "$0")/.." && pwd)"
attempt="${1:-$repository/build/fg_connected_subrect_l3_l83/macos/early-$(date -u +%Y%m%dT%H%M%SZ)}"
mode="${2:-matrix}"
model="${3:-$repository/build/fg_default_readback/eval0-default-all-a5-a/runtime-default-buffers-post-evaluate-0}"
rom="${4:-$repository/build/fg_connected_tail/macos-l56-l87/attempt-exact-v1/sm120-rcp-correction-rom.bin}"
sigmoid="${5:-$repository/build/fg_postprocess_l84/rtx/oracle-006/sigmoid-process-a/sm120-half-logit-sigmoid-lut.bin}"
ex2_unit="${6:-$repository/build/fg_math_ex2/unit-v1/unit-minus1-to-zero-u32.bin}"
ex2_fraction="${7:-$repository/build/fg_math_ex2/rtx-v3/fraction-minus2-to-minus1-u32.bin}"
if [[ -e "$attempt" ]]; then
  echo "Attempt exists; preserve it: $attempt" >&2
  exit 2
fi
if [[ "$mode" != smoke && "$mode" != matrix ]]; then
  echo "Mode must be smoke or matrix" >&2
  exit 3
fi
libs="$attempt/libs"
mkdir -p "$libs"

compile_metal() {
  local source="$1" name="$2"
  xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off \
    -I "$repository/src/fg_conv" \
    -c "$source" -o "$libs/$name.air"
  xcrun -sdk macosx metallib "$libs/$name.air" -o "$libs/$name.metallib"
}

compile_metal "$repository/src/fg_connected_subrect_l3_l83/subrect_module11.metal" subrect_module11
compile_metal "$repository/src/fg_connected_subrect_l3_l83/subrect_module15.metal" subrect_module15
compile_metal "$repository/src/fg_early_l5_l15/early_l5_l15.metal" module13
compile_metal "$repository/src/fg_early_l6_l16/early_l6_l16.metal" module21
compile_metal "$repository/src/fg_early_l7_l17/early_l7_l17.metal" module23
compile_metal "$repository/src/fg_early_l8_l18/early_l8_l18.metal" module25
compile_metal "$repository/src/fg_dynamic_module14/dynamic_module14.metal" module14
compile_metal "$repository/src/fg_early_l11/early_l11.metal" module4
compile_metal "$repository/src/fg_connected_subrect_l3_l83/subrect_module5.metal" module5

xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc -Wall -Wextra -Werror \
  -Wno-unused-function -Wno-deprecated-declarations \
  -I "$repository/src/fg_connected_subrect_l3_l83" \
  "$repository/src/fg_connected_subrect_l3_l83/macos_subrect_early.mm" \
  -framework Foundation -framework Metal -o "$attempt/subrect-early"

if [[ "$mode" == smoke ]]; then
  poisons=(a5)
  processes=(a)
else
  poisons=(a5 5a)
  processes=(a b)
fi
for poison in "${poisons[@]}"; do
  for process in "${processes[@]}"; do
    run_id="subrect-early-poison-$poison-process-$process"
    run="$attempt/$run_id"
    mkdir -p "$run"
    "$attempt/subrect-early" "$libs" "$model" "$rom" "$sigmoid" \
      "$ex2_unit" "$ex2_fraction" "$run" "$run_id" "$poison" \
      >"$run/stdout.log" 2>"$run/stderr.log"
  done
done

echo "attempt_root=$attempt"
echo "result=pass"
