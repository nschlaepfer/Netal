#!/usr/bin/env bash
# Build the connected-block probe (launches 56-83 in one command buffer, 28 checkpoints against the RTX-matched
# hashes) with the fused-bilinear-fetch candidates of the four bottom families and the operand-level fetch check.
# Every kernel is compiled with -fno-fast-math -ffp-contract=off through `xcrun -sdk macosx metal`; if xcrun has no
# Metal toolchain, put a forwarding xcrun first on PATH.
#   src/fg_connected_custom_l56_l83/build_fu_probe.sh [out dir, default build/fu_probe]
# Run: <out>/host <out>/libs <model dir> evidence/specs/fg-connected-custom-l56-l83-v1/verification.json [reps] [formal] [rounds] [scalar reps]
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"
out="${1:-$repo/build/fu_probe}"; L="$out/libs"
rm -rf "$out"; mkdir -p "$L"
c()   { xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off -I "$repo/src/fg_conv" -c "$repo/$1" -o "$L/$2.air"; xcrun -sdk macosx metallib "$L/$2.air" -o "$L/$2.metallib"; }
c32() { xcrun -sdk macosx metal -std=metal3.2 -fno-fast-math -ffp-contract=off -I "$repo/src/fg_conv" -c "$repo/$1" -o "$L/$2.air"; xcrun -sdk macosx metallib "$L/$2.air" -o "$L/$2.metallib"; }
# scalar oracles (clock reference and cross-check)
c src/fg_initial_merge/initial_merge.metal initial_merge
c src/fg_custom_upsample/custom_upsample.metal upsample
c src/fg_custom_upsample/custom_upsample_planes3.metal upsample3
c src/fg_central_block/central_block.metal central
c src/fg_conv/custom_conv_pre.metal conv_pre; c src/fg_conv/custom_conv0_c8.metal conv0; c src/fg_conv/custom_conv1.metal conv1; c src/fg_conv/custom_conv2.metal conv2
c src/fg_connected_custom_l56_l83/connected_block0_bot0.metal bot0; c src/fg_connected_custom_l56_l83/connected_block0_bot1.metal bot1
c src/fg_conv/custom_block1_conv0.metal block1_conv0; c src/fg_conv/custom_block1_conv1.metal block1_conv1; c src/fg_conv/custom_block1_conv2.metal block1_conv2
c src/fg_connected_custom_l56_l83/connected_block1_bot0.metal block1_bot0; c src/fg_connected_custom_l56_l83/connected_block1_bot1.metal block1_bot1
# guarded pipelines as the runtime builds them (build-macos-fg-runtime.sh)
c src/fg_conv/compact_suspects.metal compact_suspects; c src/fg_conv/compact_ordered.metal compact_ordered; c32 src/fg_conv/compact_ordered_ind.metal compact_ordered_ind
c src/fg_conv/connected_upsamples.metal connected_upsamples
c src/fg_conv/custom_conv_pre_pipeline_wq.metal p_conv_pre; c src/fg_conv/custom_conv0_c8_pipeline_wq.metal p_conv0; c src/fg_conv/custom_conv1_pipeline_wq.metal p_conv1
c src/fg_conv/custom_conv2_bounded_v13.metal p_conv2b; c src/fg_conv/custom_conv2_repair_v6.metal p_conv2r; c32 src/fg_conv/custom_conv2_repair_v8.metal p_conv2r8
c src/fg_conv/custom_bot0_pipeline_wq.metal p_bot0; c src/fg_conv/custom_block0_bot1_pipeline_wq.metal p_b0bot1
c src/fg_conv/custom_block1_conv0_pipeline_wq.metal p_b1conv0; c src/fg_conv/custom_block1_conv1_pipeline_wq.metal p_b1conv1
c src/fg_conv/custom_block1_conv2_bounded_wq.metal p_b1conv2b; c32 src/fg_conv/custom_block1_conv2_repair_ind.metal p_b1conv2r
c src/fg_conv/custom_block1_bot0_pipeline_wq.metal p_b1bot0; c src/fg_conv/custom_block1_bot1_pipeline_wq.metal p_b1bot1
# candidate: fused bilinear fetch (make_fu.py) and its operand-level check
c src/fg_conv/custom_bot0_pipeline_fu.metal p_bot0_fu; c src/fg_conv/custom_block0_bot1_pipeline_fu.metal p_b0bot1_fu
c src/fg_conv/custom_block1_bot0_pipeline_fu.metal p_b1bot0_fu; c src/fg_conv/custom_block1_bot1_pipeline_fu.metal p_b1bot1_fu
c src/fg_conv/fu_fetch_check.metal fu_fetch_check
rm -f "$L"/*.air
xcrun -sdk macosx clang++ -std=c++20 -O2 -fobjc-arc -Wno-deprecated-declarations \
  "$repo/src/fg_connected_custom_l56_l83/macos_connected_pipelines.mm" -framework Foundation -framework Metal -o "$out/host"
echo "built $out"
