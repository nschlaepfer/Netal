#!/usr/bin/env bash
set -euo pipefail
repository="$(cd "$(dirname "$0")/.." && pwd)"
attempt_root="${1:-$repository/build/fg_connected_dynamic_l3_l23/macos/attempt-$(date -u +%Y%m%dT%H%M%SZ)}"
model_root="${2:-$repository/build/fg_default_readback/eval0-default-all-a5-a/runtime-default-buffers-post-evaluate-0}"
run_mode="${3:-matrix}"
smoke_resolution="${4:-720p}"
if [[ -e "$attempt_root" ]]; then echo "Attempt exists: $attempt_root" >&2; exit 2; fi
weight63="$model_root/default-token-000063.bin"
weight65="$model_root/default-token-000065.bin"
[[ "$(shasum -a 256 "$weight63"|awk '{print $1}')" == "44595b02307a6b934a124abe476cfd327440e2acba7f5ed69f73195ca743e3d9" ]] || exit 3
[[ "$(shasum -a 256 "$weight65"|awk '{print $1}')" == "eb99328524fddc9f5f3fac12b8e3a4011e66a75e071cd07126a70c553458bc60" ]] || exit 4
base_generated="$attempt_root/base-generated"
generated="$attempt_root/generated"
python3 "$repository/scripts/generate-fg-connected-dynamic-l3-l21.py" --repository "$repository" --output "$base_generated"
python3 "$repository/scripts/generate-fg-connected-dynamic-l3-l23.py" --base "$base_generated" --output "$generated"
libs="$attempt_root/libs";mkdir -p "$libs"
archive="$repository/evidence/checkpoints/2026-08-31-sm120-rcp-rom/external-evidence-sm120-rcp-rom.tar.gz"
rom="$attempt_root/sm120-rcp-correction-rom.bin"
tar -xOf "$archive" work/g23-rcp-architecture/m3-rcp-rom/correction-rom.bin >"$rom"
cp "$weight63" "$weight65" "$libs/"
compile_metal(){ local source="$1" name="$2";xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off -I "$repository/src/fg_conv" -c "$repository/$source" -o "$libs/$name.air";xcrun -sdk macosx metallib "$libs/$name.air" -o "$libs/$name.metallib";}
compile_metal src/fg_dynamic_module11/dynamic_module11.metal module11
compile_metal src/fg_dynamic_module15/dynamic_module15.metal module15
compile_metal src/fg_dynamic_module13/dynamic_module13.metal module13
compile_metal src/fg_dynamic_module21/dynamic_module21.metal module21
compile_metal src/fg_resolution_preprocess_l22_l24/module22.metal module22
compile_metal src/fg_dynamic_module23/dynamic_module23_mode0.metal module23
xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off -I "$repository/src/fg_conv" -c "$repository/src/fg_resolution_preprocess_l22_l24/module24_level10.metal" -o "$libs/module24-level10.air"
xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off -I "$repository/src/fg_conv" -c "$repository/src/fg_resolution_preprocess_l22_l24/module24_level11.metal" -o "$libs/module24-level11.air"
xcrun -sdk macosx metallib "$libs/module24-level10.air" "$libs/module24-level11.air" -o "$libs/module24.metallib"
compile_metal src/fg_dynamic_module25/dynamic_module25.metal module25
compile_metal src/fg_dynamic_module14/dynamic_module14.metal module14
compile_metal src/fg_early_status_l10_l12/status_write.metal status
compile_metal src/fg_early_l11/early_l11.metal module4
compile_metal src/fg_connected_dynamic_early/module5_mixed.metal module5
xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off -c "$repository/src/fg_conv/conv_l20.metal" -o "$libs/l20.air"
xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off -c "$repository/src/fg_pooling/pooling.metal" -o "$libs/l21.air"
xcrun -sdk macosx metallib "$libs/l20.air" "$libs/l21.air" -o "$libs/l20-l21.metallib"
xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off -I "$repository/src/fg_conv" -c "$repository/src/fg_connected_dynamic_l3_l23/conv_1x1_ptx_zero.metal" -o "$libs/l22.air"
xcrun -sdk macosx metallib "$libs/l22.air" "$libs/l21.air" -o "$libs/l22-l23.metallib"
xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc -Wall -Wextra -Werror -Wno-unused-function -I "$repository/src" -I "$repository/src/fg_connected_dynamic_early" -I "$repository/src/fg_connected_dynamic_l3_l21" "$generated/generated_macos.mm" -framework Foundation -framework Metal -o "$attempt_root/host"
if [[ "$run_mode" == smoke ]];then resolutions=("$smoke_resolution");poisons=(a5);ordinals=(a);else resolutions=(720p 1080p 1440p 4k);poisons=(a5 5a);ordinals=(a b);fi
for resolution in "${resolutions[@]}";do for poison in "${poisons[@]}";do for ordinal in "${ordinals[@]}";do run_id="$resolution-poison-$poison-process-$ordinal";run_root="$attempt_root/$run_id";mkdir -p "$run_root";artifact=hash-only;[[ "$poison$ordinal" == a5a ]]&&artifact=full;"$attempt_root/host" "$libs" "$rom" "$run_root" "$run_id" "$poison" "$resolution" "$artifact">"$run_root/stdout.log" 2>"$run_root/stderr.log";done;done;done
echo "attempt_root=$attempt_root"
echo result=pass
