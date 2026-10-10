#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/.."&&pwd)";root="${1:-$repo/build/fg_connected_tail/macos-l56-l87/attempt-$(date -u +%Y%m%dT%H%M%SZ)}";[[ ! -e "$root" ]]||{ echo "Attempt exists" >&2;exit 2;};libs="$root/libs";mkdir -p "$libs"
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
compile src/fg_postprocess_l84/postprocess_l84.metal postprocess_l84
compile src/fg_postprocess_l85/postprocess_l85.metal postprocess_l85
compile src/fg_postprocess_l86/postprocess_l86_mode0.metal postprocess_l86
compile src/fg_compositor_l87/compositor_l87.metal compositor_l87
xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc -Wall -Wextra -Werror -Wno-unused-function -Wno-deprecated-declarations "$repo/src/fg_connected_tail_l84_l87/macos_connected_l56_l87.mm" -framework Foundation -framework Metal -o "$root/host"
archive="$repo/evidence/checkpoints/2026-08-31-sm120-rcp-rom/external-evidence-sm120-rcp-rom.tar.gz";rom="$root/sm120-rcp-correction-rom.bin";tar -xOf "$archive" work/g23-rcp-architecture/m3-rcp-rom/correction-rom.bin >"$rom";sigmoid="$repo/build/fg_postprocess_l84/rtx/oracle-006/sigmoid-process-a/sm120-half-logit-sigmoid-lut.bin";[[ "$(shasum -a 256 "$rom"|awk '{print $1}')" == "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750" ]];[[ "$(shasum -a 256 "$sigmoid"|awk '{print $1}')" == "394394a5258bad437495d68076be75d8413fa0ed5b752400e3947c332437b850" ]]
model="$repo/build/fg_default_readback/eval0-default-all-a5-a/runtime-default-buffers-post-evaluate-0"
for poison in a5 5a;do for ord in a b;do id="connected-l56-l87-poison-$poison-process-$ord";run="$root/$id";mkdir -p "$run";"$root/host" "$libs" "$model" "$rom" "$sigmoid" "$run" "$id" "$poison">"$run/stdout.log" 2>"$run/stderr.log";done;done
python3 "$repo/scripts/verify-fg-connected-l56-l87.py" backend --backend mac --root "$root" --output "$root/backend-receipt.json";echo "attempt_root=$root";echo result=pass
