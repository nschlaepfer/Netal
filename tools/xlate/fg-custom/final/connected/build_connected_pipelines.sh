#!/usr/bin/env bash
set -euo pipefail
W="$(cd "$(dirname "$0")" && pwd)"; L="$W/build_cp/libs"; rm -rf "$W/build_cp"; mkdir -p "$L"
c32(){ xcrun -sdk macosx metal -std=metal3.2 -fno-fast-math -ffp-contract=off -I "$W/src/fg_conv" -c "$W/$1" -o "$L/$2.air"; xcrun -sdk macosx metallib "$L/$2.air" -o "$L/$2.metallib"; }
c(){ xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off -I "$W/src/fg_conv" -c "$W/$1" -o "$L/$2.air"; xcrun -sdk macosx metallib "$L/$2.air" -o "$L/$2.metallib"; }
c src/fg_initial_merge/initial_merge.metal initial_merge
c src/fg_custom_upsample/custom_upsample.metal upsample; c src/fg_custom_upsample/custom_upsample_planes3.metal upsample3
c src/fg_central_block/central_block.metal central
c src/fg_conv/custom_conv_pre.metal conv_pre; c src/fg_conv/custom_conv0_c8.metal conv0; c src/fg_conv/custom_conv1.metal conv1; c src/fg_conv/custom_conv2.metal conv2
c src/fg_connected_custom_l56_l83/connected_block0_bot0.metal bot0; c src/fg_connected_custom_l56_l83/connected_block0_bot1.metal bot1
c src/fg_conv/custom_block1_conv0.metal block1_conv0; c src/fg_conv/custom_block1_conv1.metal block1_conv1; c src/fg_conv/custom_block1_conv2.metal block1_conv2
c src/fg_connected_custom_l56_l83/connected_block1_bot0.metal block1_bot0; c src/fg_connected_custom_l56_l83/connected_block1_bot1.metal block1_bot1
c src/fg_conv/compact_suspects.metal compact_suspects; c src/fg_conv/compact_ordered.metal compact_ordered; c src/fg_conv/connected_upsamples.metal connected_upsamples
c src/fg_conv/custom_conv_pre_pipeline_wq.metal p_conv_pre; c src/fg_conv/custom_conv0_c8_pipeline_wq.metal p_conv0; c src/fg_conv/custom_conv1_pipeline_wq.metal p_conv1
c src/fg_conv/custom_conv2_bounded_v13.metal p_conv2b; c src/fg_conv/custom_conv2_repair_v6.metal p_conv2r; c32 src/fg_conv/custom_conv2_repair_v8.metal p_conv2r8; c32 src/fg_conv/compact_ordered_ind.metal compact_ordered_ind
c src/fg_conv/custom_bot0_pipeline_wq.metal p_bot0
c src/fg_conv/custom_block1_conv0_pipeline_wq.metal p_b1conv0; c src/fg_conv/custom_block1_conv1_pipeline_wq.metal p_b1conv1
c src/fg_conv/custom_block1_conv2_bounded_wq.metal p_b1conv2b; c32 src/fg_conv/custom_block1_conv2_repair_ind.metal p_b1conv2r
for extra in "$@"; do n=$(basename "$extra" .metal); c "$extra" "p_$n"; done
xcrun -sdk macosx clang++ -std=c++20 -O2 -fobjc-arc -Wno-deprecated-declarations "$W/src/fg_connected_custom_l56_l83/macos_connected_pipelines.mm" -framework Foundation -framework Metal -o "$W/build_cp/host"
echo built
