#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
attempt="${1:-$repo/build/fg_runtime/macos/attempt-$(date -u +%Y%m%dT%H%M%SZ)}"
model="${2:-$repo/build/fg_default_readback/eval0-default-all-a5-a/runtime-default-buffers-post-evaluate-0}"
rom="${3:-$repo/build/fg_connected_tail/macos-l56-l87/attempt-exact-v1/sm120-rcp-correction-rom.bin}"
sigmoid="${4:-$repo/build/fg_postprocess_l84/rtx/oracle-006/sigmoid-process-a/sm120-half-logit-sigmoid-lut.bin}"
ex2_unit="${FG_EX2_UNIT_TABLE:-$repo/build/fg_math_ex2/unit-v1/unit-minus1-to-zero-u32.bin}"
ex2_fraction="${FG_EX2_FRACTION_TABLE:-$repo/build/fg_math_ex2/rtx-v3/fraction-minus2-to-minus1-u32.bin}"
ex2_mode="${FG_RUNTIME_EX2_MODE:-reference}"
auxiliary_mode="${FG_RUNTIME_AUXILIARY_MODE:-internal}"
hmma_mode="${FG_RUNTIME_HMMA_MODE:-loop}"
l52_source="${FG_RUNTIME_L52_SOURCE:-src/fg_production_contract_l3_l87/l52_production_exact.metal}"
l54_source="${FG_RUNTIME_L54_SOURCE:-src/fg_production_contract_l3_l87/l54_production_exact.metal}"
conv3_source="${FG_RUNTIME_CONV3_SOURCE:-src/fg_conv/conv_3x3.metal}"
bot0_source="${FG_RUNTIME_BOT0_SOURCE:-src/fg_connected_custom_l56_l83/connected_block0_bot0.metal}"
if [[ -e "$attempt" ]]; then
  echo "Attempt exists; preserve it: $attempt" >&2
  exit 2
fi
if [[ "$hmma_mode" != loop && "$hmma_mode" != clz && "$hmma_mode" != u32 ]]; then
  echo "Unknown HMMA mode: $hmma_mode" >&2
  exit 2
fi
libs="$attempt/libs"
mkdir -p "$libs" "$attempt/build"

compile_metal() {
  local source="$1" name="$2"
  local source_path="$source"
  local -a hmma_flags=()
  if [[ "$source" != /* ]]; then source_path="$repo/$source"; fi
  if [[ "$hmma_mode" != loop ]] && grep -q 'hmma_exact\.metalh' "$source_path"; then
    if [[ "$hmma_mode" == clz ]]; then
      hmma_flags=(-include "$repo/src/fg_conv_simd_l20/hmma_exact_clz_override.metalh")
    else
      hmma_flags=(-include "$repo/src/fg_conv_simd_l20/hmma_exact_u32_override.metalh")
    fi
    printf '%s\n' "$source" >> "$attempt/build/hmma-${hmma_mode}-sources.txt"
    # The hardware lowering of the same kernel, chosen at run time by NSFG_PREFIX_MODE=hw (game path):
    # a dedicated source where one exists (vectorised loads, oc-major threads), else the fp32 FMA
    # override of the exact helper.
    local hw_source="$repo/src/fg_conv/${name}_hw.metal"
    if [[ -f "$hw_source" ]]; then
      xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off -I "$repo/src/fg_conv" -c "$hw_source" -o "$libs/${name}_hw.air"
    else
      xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off -I "$repo/src/fg_conv" -I "$repo/src/fg_math_ex2" \
        -include "$repo/src/fg_conv_simd_l20/hmma_hw_override.metalh" -c "$source_path" -o "$libs/${name}_hw.air"
    fi
    xcrun -sdk macosx metallib "$libs/${name}_hw.air" -o "$libs/${name}_hw.metallib"
  fi
  xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math \
    -ffp-contract=off -I "$repo/src/fg_conv" -I "$repo/src/fg_math_ex2" \
    ${hmma_flags[@]+"${hmma_flags[@]}"} \
    -c "$source_path" \
    -o "$libs/$name.air"
  xcrun -sdk macosx metallib "$libs/$name.air" -o "$libs/$name.metallib"
}

if [[ -n "${FG_RUNTIME_REUSE_LIBS:-}" ]]; then
  cp -R "$FG_RUNTIME_REUSE_LIBS/." "$libs/"
  if [[ -n "${FG_RUNTIME_L52_SOURCE:-}" || -n "${FG_RUNTIME_L54_SOURCE:-}" ]]; then
    compile_metal "$l52_source" mid_l52
    compile_metal "$l54_source" mid_l54
  fi
  if [[ -n "${FG_RUNTIME_CONV3_SOURCE:-}" ]]; then
    compile_metal "$conv3_source" conv_3x3
  fi
  if [[ -n "${FG_RUNTIME_BOT0_SOURCE:-}" ]]; then
    compile_metal "$bot0_source" bot0
  fi
else
compile_metal src/fg_early_l3_l13/early_l3_l13.metal l3_l13
compile_metal src/fg_early_l4_l14/early_l4_l14.metal l4_l14
compile_metal src/fg_early_l5_l15/early_l5_l15.metal l5_l15
compile_metal src/fg_early_l6_l16/early_l6_l16.metal l6_l16
compile_metal src/fg_early_l7_l17/early_l7_l17.metal l7_l17
compile_metal src/fg_early_l8_l18/early_l8_l18.metal l8_l18
compile_metal src/fg_early_l9/early_l9.metal l9
compile_metal src/fg_early_status_l10_l12/status_write.metal l10_l12
compile_metal src/fg_early_l11/early_l11.metal l11
compile_metal src/fg_early_l19/early_l19.metal l19
compile_metal "$conv3_source" conv_3x3
compile_metal src/fg_production_contract_l3_l87/conv_1x1_temporal.metal conv_1x1
compile_metal src/fg_pooling/pooling.metal pooling
compile_metal src/fg_upscale/upscale.metal upscale-prefix
compile_metal src/fg_elementwise/elementwise_family.metal elementwise
compile_metal "$l52_source" mid_l52
compile_metal src/fg_mid_l53/mid_l53.metal mid_l53
compile_metal "$l54_source" mid_l54
compile_metal src/fg_mid_l55/fg_mid_l55.metal mid_l55
compile_metal src/fg_initial_merge/initial_merge.metal initial_merge
compile_metal src/fg_conv/custom_conv_pre.metal conv_pre
compile_metal src/fg_conv/custom_conv0_c8.metal conv0
compile_metal src/fg_conv/custom_conv1.metal conv1
compile_metal src/fg_conv/custom_conv2.metal conv2
compile_metal "$bot0_source" bot0
compile_metal src/fg_production_contract_l3_l87/connected_block0_bot1_production.metal bot1
compile_metal src/fg_custom_upsample/custom_upsample.metal upsample
compile_metal src/fg_central_block/central_block.metal central
compile_metal src/fg_conv/custom_block1_conv0.metal block1_conv0
compile_metal src/fg_conv/custom_block1_conv1.metal block1_conv1
compile_metal src/fg_conv/custom_block1_conv2.metal block1_conv2
compile_metal src/fg_connected_custom_l56_l83/connected_block1_bot0.metal block1_bot0
compile_metal src/fg_connected_custom_l56_l83/connected_block1_bot1.metal block1_bot1
compile_metal src/fg_postprocess_l84/postprocess_l84.metal postprocess_l84
compile_metal src/fg_postprocess_l85/postprocess_l85.metal postprocess_l85
compile_metal src/fg_postprocess_l86/postprocess_l86_mode0.metal postprocess_l86
compile_metal src/fg_compositor_l87/compositor_l87.metal compositor_l87
fi

# guarded exact pipelines for the custom block (tools/xlate/fg-custom on Netal dev; 28/28 launch checkpoints vs RTX).
# Compiled whether or not a libs directory is reused: these metallibs are new names, never part of a reused set.
compile_metal32() {
  local source="$1" name="$2"
  xcrun -sdk macosx metal -std=metal3.2 -fno-fast-math -ffp-contract=off -I "$repo/src/fg_conv" -c "$repo/$source" -o "$libs/$name.air"
  xcrun -sdk macosx metallib "$libs/$name.air" -o "$libs/$name.metallib"
}
compile_metal src/fg_conv/compact_suspects.metal compact_suspects
compile_metal src/fg_conv/compact_ordered.metal compact_ordered
compile_metal32 src/fg_conv/compact_ordered_ind.metal compact_ordered_ind
compile_metal src/fg_conv/connected_upsamples.metal connected_upsamples
compile_metal src/fg_custom_upsample/custom_upsample_planes3.metal upsample3
compile_metal src/fg_conv/custom_conv_pre_pipeline_wq.metal p_conv_pre
compile_metal src/fg_conv/custom_conv0_c8_pipeline_wq.metal p_conv0
compile_metal src/fg_conv/custom_conv1_pipeline_wq.metal p_conv1
compile_metal src/fg_conv/custom_conv2_bounded_v13.metal p_conv2b
compile_metal src/fg_conv/custom_conv2_repair_v6.metal p_conv2r
compile_metal32 src/fg_conv/custom_conv2_repair_v8.metal p_conv2r8
compile_metal src/fg_conv/custom_bot0_pipeline_wq.metal p_bot0
compile_metal src/fg_conv/custom_block0_bot1_pipeline_wq.metal p_b0bot1
compile_metal src/fg_conv/custom_block1_conv0_pipeline_wq.metal p_b1conv0
compile_metal src/fg_conv/custom_block1_conv1_pipeline_wq.metal p_b1conv1
compile_metal src/fg_conv/custom_block1_conv2_bounded_wq.metal p_b1conv2b
compile_metal32 src/fg_conv/custom_block1_conv2_repair_ind.metal p_b1conv2r
compile_metal src/fg_conv/custom_block1_bot0_pipeline_wq.metal p_b1bot0
compile_metal src/fg_conv/custom_block1_bot1_pipeline_wq.metal p_b1bot1
# optional candidate, selected at run time by NSFG_FUSED_UPSAMPLE: the four bottom families fetch their bilinear
# operands inside the convolution (src/fg_conv/make_fu.py; tools/xlate/fg-custom/RESULTS.md on Netal dev)
compile_metal src/fg_conv/custom_bot0_pipeline_fu.metal p_bot0_fu
compile_metal src/fg_conv/custom_block0_bot1_pipeline_fu.metal p_b0bot1_fu
compile_metal src/fg_conv/custom_block1_bot0_pipeline_fu.metal p_b1bot0_fu
compile_metal src/fg_conv/custom_block1_bot1_pipeline_fu.metal p_b1bot1_fu

python3 "$repo/scripts/generate-netal-fg-tail.py" \
  "$repo/src/fg_connected_l20_l87/macos_connected_l20_l87.mm" \
  "$attempt/build/runtime_tail.mm" \
  "$repo/src/fg_runtime/tail_resource_cache.inc"

xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc -ffp-contract=off ${FG_RUNTIME_CXX_OPT:--O2} \
  -Wall -Wextra -Werror -Wno-unused-function -Wno-deprecated-declarations \
  -I "$repo/src/fg_runtime" -I "$repo/src/fg_production_contract_l3_l87" \
  -I "$repo/src/fg_connected_l20_l87" \
  "$repo/src/fg_runtime/fg_runtime.mm" \
  "$repo/src/fg_runtime/span_recorder.mm" \
  "$repo/src/fg_runtime/metal4_shim.mm" \
  "$repo/src/fg_runtime/tail_output_bridge.mm" \
  "$repo/src/fg_runtime/offline_host.mm" \
  "$repo/src/fg_production_contract_l3_l87/macos_live_handoff.mm" \
  "$attempt/build/runtime_tail.mm" \
  -framework Foundation -framework Metal -o "$attempt/fg-runtime-offline"

xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc -ffp-contract=off ${FG_RUNTIME_CXX_OPT:--O2} \
  -Wall -Wextra -Werror -Wno-unused-function -Wno-deprecated-declarations \
  -I "$repo/src/fg_runtime" -I "$repo/src/fg_production_contract_l3_l87" \
  -I "$repo/src/fg_connected_l20_l87" \
  "$repo/src/fg_runtime/fg_runtime.mm" \
  "$repo/src/fg_runtime/span_recorder.mm" \
  "$repo/src/fg_runtime/metal4_shim.mm" \
  "$repo/src/fg_runtime/tail_output_bridge.mm" \
  "$repo/src/fg_runtime/benchmark_host.mm" \
  "$repo/src/fg_production_contract_l3_l87/macos_live_handoff.mm" \
  "$attempt/build/runtime_tail.mm" \
  -framework Foundation -framework Metal -o "$attempt/fg-runtime-benchmark"

for poison in a5 5a; do
  for ordinal in a b; do
    run_id="fg-runtime-poison-$poison-process-$ordinal"
    run="$attempt/$run_id"
    mkdir -p "$run"
    # FG_RUNTIME_RECEIPT_PREFIX (optional, word-split): a command prefix per receipt process, e.g. a GPU lock runner
    ${FG_RUNTIME_RECEIPT_PREFIX:-} "$attempt/fg-runtime-offline" "$libs" "$model" "$rom" "$sigmoid" \
      "$ex2_unit" "$ex2_fraction" "$run" "$run_id" "$poison" \
      "$ex2_mode" \
      "$auxiliary_mode" \
      >"$run/stdout.log" 2>"$run/stderr.log"
  done
done

python3 - "$attempt" "$hmma_mode" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
hmma_mode = sys.argv[2]
runs = sorted(p for p in root.glob('fg-runtime-poison-*-process-*') if p.is_dir())
if len(runs) != 4:
    raise SystemExit('runtime fresh-process count failed')
receipts = [json.loads((run/'run-receipt.json').read_text()) for run in runs]
if len({receipt['process_id'] for receipt in receipts}) != 4:
    raise SystemExit('runtime process IDs are not fresh')
if sorted(receipt['poison_byte'] for receipt in receipts) != [90, 90, 165, 165]:
    raise SystemExit('runtime poison coverage failed')
if any(receipt['result'] != 'pass' for receipt in receipts):
    raise SystemExit('runtime process failed')
if len({receipt['ex2_mode'] for receipt in receipts}) != 1:
    raise SystemExit('runtime EX2 mode drift')
if len({receipt['auxiliary_mode'] for receipt in receipts}) != 1:
    raise SystemExit('runtime auxiliary mode drift')
if any(not receipt['resource_allocations_per_generated_frame'][0] or
       any(receipt['resource_allocations_per_generated_frame'][1:])
       for receipt in receipts):
    raise SystemExit('runtime post-warmup allocation gate failed')
roots = {json.dumps(receipt['output_sha256']) for receipt in receipts}
if len(roots) != 1:
    raise SystemExit('runtime output poison/process invariance failed')
checkpoint_roots = {
    hashlib.sha256(json.dumps(receipt['checkpoint_sha256'], sort_keys=True,
                              separators=(',', ':')).encode()).hexdigest()
    for receipt in receipts
}
if len(checkpoint_roots) != 1:
    raise SystemExit('runtime checkpoint poison/process invariance failed')
first = receipts[0]
backend = {
    'schema': 'neuralstudio.fg.runtime-backend.v1',
    'backend': 'mac-metal-runtime-prototype',
    'fresh_process_count': 4,
    'poison_bytes': [90, 165],
    'ex2_mode': first['ex2_mode'],
    'auxiliary_mode': first['auxiliary_mode'],
    'hmma_mode': hmma_mode,
    'rendered_frame_count': 16,
    'selected_launch_count': 1149,
    'checkpoint_root_sha256': next(iter(checkpoint_roots)),
    'output_sha256': first['output_sha256'],
    'caller_owned_inputs_immutable': True,
    'rejected_output_unchanged': True,
    'persistent_early_resources': True,
    'persistent_tail_resources': True,
    'no_resource_allocation_after_warmup': True,
    'tail_audit_readbacks_enabled': True,
    'tail_probe_file_io': False,
    'caller_owned_current_previous_rgba8': True,
    'caller_owned_motion_rg16f_depth_r32f': True,
    'caller_owned_generated_target_rgba8': True,
    'l53_l55_auxiliary_profile': first['l53_l55_auxiliary_profile'],
    'l53_l55_auxiliary_caller_owned': first['l53_l55_auxiliary_caller_owned'],
    'l52_l53_l55_live_early_resource_aliases':
        first['l52_l53_l55_live_early_resource_aliases'],
    'model_and_architecture_buffers_retained': True,
    'metal_resource_allocations_per_generated_frame':
        first['resource_allocations_per_generated_frame'],
    'metal_resource_allocations_after_warmup': 0,
    'cpu_allocations_counted': False,
    'expected_or_archived_intermediate_ingress': False,
    'result': 'pass',
}
(root/'backend-receipt.json').write_text(json.dumps(backend,indent=2,sort_keys=True)+'\n')
print('backend_receipt_sha256='+hashlib.sha256((root/'backend-receipt.json').read_bytes()).hexdigest())
print('result=pass')
PY

echo "attempt_root=$attempt"
echo "result=pass"
