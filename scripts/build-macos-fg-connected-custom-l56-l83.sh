#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/.."&&pwd)";root="${1:-$repo/build/fg_connected_custom/macos-l56-l83/attempt-$(date -u +%Y%m%dT%H%M%SZ)}";[[ ! -e "$root" ]]||{ echo "Attempt exists" >&2;exit 2;};libs="$root/libs";mkdir -p "$libs"
compile(){ local src="$1" name="$2";xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off -I "$repo/src/fg_conv" -c "$repo/$src" -o "$libs/$name.air";xcrun -sdk macosx metallib "$libs/$name.air" -o "$libs/$name.metallib";}
compile src/fg_initial_merge/initial_merge.metal initial_merge
compile src/fg_conv/custom_conv_pre.metal conv_pre
compile src/fg_conv/custom_conv0_c8.metal conv0
compile src/fg_conv/custom_conv1.metal conv1
compile src/fg_conv/custom_conv2.metal conv2
compile src/fg_connected_custom_l56_l83/connected_block0_bot0.metal bot0
compile src/fg_connected_custom_l56_l83/connected_block0_bot1.metal bot1
compile src/fg_custom_upsample/custom_upsample.metal upsample
compile src/fg_central_block/central_block.metal central
compile src/fg_conv/custom_block1_conv0.metal block1_conv0
compile src/fg_conv/custom_block1_conv1.metal block1_conv1
compile src/fg_conv/custom_block1_conv2.metal block1_conv2
compile src/fg_connected_custom_l56_l83/connected_block1_bot0.metal block1_bot0
compile src/fg_connected_custom_l56_l83/connected_block1_bot1.metal block1_bot1
xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc -Wall -Wextra -Werror -Wno-unused-function "$repo/src/fg_connected_custom_l56_l83/macos_connected_l56_l83.mm" -framework Foundation -framework Metal -o "$root/host"
model="$repo/build/fg_default_readback/eval0-default-all-a5-a/runtime-default-buffers-post-evaluate-0"
for poison in a5 5a;do for ord in a b;do id="connected-l56-l83-poison-$poison-process-$ord";run="$root/$id";mkdir -p "$run";"$root/host" "$libs" "$model" "$run" "$id" "$poison">"$run/stdout.log" 2>"$run/stderr.log";done;done
python3 "$repo/scripts/verify-fg-connected-custom-l56-l83.py" backend --backend mac --root "$root" --output "$root/backend-receipt.json";echo "attempt_root=$root";echo result=pass
