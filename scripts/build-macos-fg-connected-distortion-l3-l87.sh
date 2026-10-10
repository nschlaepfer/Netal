#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
attempt="${1:-$repo/build/fg_connected_distortion_l3_l87/macos/full-attempt-$(date -u +%Y%m%dT%H%M%SZ)}"
model="${2:-$repo/build/fg_default_readback/eval0-default-all-a5-a/runtime-default-buffers-post-evaluate-0}"
rom="${3:-$repo/build/fg_connected_tail/macos-l56-l87/attempt-exact-v1/sm120-rcp-correction-rom.bin}"
sigmoid="${4:-$repo/build/fg_postprocess_l84/rtx/oracle-006/sigmoid-process-a/sm120-half-logit-sigmoid-lut.bin}"
ex2_unit="${FG_EX2_UNIT_TABLE:-$repo/build/fg_math_ex2/unit-v1/unit-minus1-to-zero-u32.bin}"
ex2_fraction="${FG_EX2_FRACTION_TABLE:-$repo/build/fg_math_ex2/rtx-v3/fraction-minus2-to-minus1-u32.bin}"
if [[ -e "$attempt" ]]; then
  echo "Attempt exists; preserve it: $attempt" >&2
  exit 2
fi
libs="$attempt/libs"
mkdir -p "$libs" "$attempt/build"

compile_metal() {
  local source="$1" name="$2"
  xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math \
    -ffp-contract=off -I "$repo/src/fg_conv" -c "$repo/$source" \
    -o "$libs/$name.air"
  xcrun -sdk macosx metallib "$libs/$name.air" -o "$libs/$name.metallib"
}

compile_metal src/fg_early_l3_l13/early_l3_l13.metal l3_l13
compile_metal src/fg_early_l4_l14/early_l4_l14.metal l4_l14
compile_metal src/fg_distortion_path/module10.metal distortion10
compile_metal src/fg_connected_distortion_l3_l87/module9_full_core.metal distortion9
compile_metal src/fg_early_l5_l15/early_l5_l15.metal l5_l15
compile_metal src/fg_early_l6_l16/early_l6_l16.metal l6_l16
compile_metal src/fg_early_l7_l17/early_l7_l17.metal l7_l17
compile_metal src/fg_early_l8_l18/early_l8_l18.metal l8_l18
compile_metal src/fg_early_l9/early_l9.metal l9
compile_metal src/fg_early_status_l10_l12/status_write.metal l10_l12
compile_metal src/fg_early_l11/early_l11.metal l11
compile_metal src/fg_early_l19/early_l19.metal l19
compile_metal src/fg_conv/conv_3x3.metal l20
compile_metal src/fg_conv/conv_3x3.metal conv_3x3
compile_metal src/fg_conv/conv_1x1.metal conv_1x1
compile_metal src/fg_pooling/pooling.metal pooling
compile_metal src/fg_upscale/upscale.metal upscale-prefix
compile_metal src/fg_elementwise/elementwise_family.metal elementwise
compile_metal src/fg_connected_distortion_l3_l87/l52_distortion_exact.metal mid_l52
compile_metal src/fg_mid_l53/mid_l53.metal mid_l53
compile_metal src/fg_connected_distortion_l3_l87/l54_distortion_exact.metal mid_l54
compile_metal src/fg_mid_l55/fg_mid_l55.metal mid_l55
compile_metal src/fg_initial_merge/initial_merge.metal initial_merge
compile_metal src/fg_conv/custom_conv_pre.metal conv_pre
compile_metal src/fg_conv/custom_conv0_c8.metal conv0
compile_metal src/fg_conv/custom_conv1.metal conv1
compile_metal src/fg_conv/custom_conv2.metal conv2
compile_metal src/fg_connected_custom_l56_l83/connected_block0_bot0.metal bot0
compile_metal src/fg_connected_custom_l56_l83/connected_block0_bot1.metal bot1
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

python3 - "$repo/src/fg_connected_l20_l87/macos_connected_l20_l87.mm" \
  "$attempt/build/macos_full_tail_live_main.mm" <<'PY'
import pathlib, sys
source = pathlib.Path(sys.argv[1]).read_text()
replacements = {
    '#undef main\n\n#include "../fg_conv/conv_1x1_contract.h"':
        '#undef main\n\n#include "macos_live_handoff.h"\n'
        '#include "../fg_conv/conv_1x1_contract.h"',
    'int main(int argc, const char* argv[]) {':
        'int FgConnectedFullTailMain(int argc, const char* argv[]) {',
    '      const auto prefix_input_before = PrefixInput();\n'
    '      id<MTLBuffer> prefix_input =\n'
    '          Guarded(device, &prefix_input_before, kInputBytes, 0);':
        '      id<MTLBuffer> prefix_input = FgConnectedMacLivePrefixInput();\n'
        '      if (prefix_input == nil)\n'
        '        throw std::runtime_error("live allocation37 binding missing");\n'
        '      const auto prefix_input_before = Payload(prefix_input, kInputBytes);',
    '      [l52_encoder setBuffer:l52_params_buffer offset:0 atIndex:2];':
        '      [l52_encoder setBuffer:l52_params_buffer offset:0 atIndex:2];\n'
        '      [l52_encoder setBuffer:FgConnectedMacEx2UnitTable() offset:kGuardBytes atIndex:3];\n'
        '      [l52_encoder setBuffer:FgConnectedMacEx2FractionTable() offset:kGuardBytes atIndex:4];',
    '      [l54_encoder setBytes:&l54_params length:sizeof(l54_params) atIndex:3];':
        '      [l54_encoder setBytes:&l54_params length:sizeof(l54_params) atIndex:3];\n'
        '      [l54_encoder setBuffer:FgConnectedMacEx2UnitTable() offset:kGuardBytes atIndex:4];\n'
        '      [l54_encoder setBuffer:FgConnectedMacEx2FractionTable() offset:kGuardBytes atIndex:5];',
}
for old, new in replacements.items():
    if source.count(old) != 1:
        raise SystemExit(f'accepted tail patch anchor drift: {old[:48]!r}')
    source = source.replace(old, new)
pathlib.Path(sys.argv[2]).write_text(source)
PY

xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc -Wall -Wextra -Werror \
  -Wno-unused-function -Wno-deprecated-declarations \
  -DFG_CONNECTED_FULL_CHAIN -I "$repo/src/fg_connected_distortion_l3_l87" \
  -I "$repo/src/fg_connected_l20_l87" \
  "$repo/src/fg_connected_distortion_l3_l87/macos_connected_distortion.mm" \
  "$repo/src/fg_connected_distortion_l3_l87/macos_live_handoff.mm" \
  "$attempt/build/macos_full_tail_live_main.mm" \
  -framework Foundation -framework Metal -o "$attempt/host"

for poison in a5 5a; do
  for ordinal in a b; do
    run_id="connected-distortion-l3-l91-poison-$poison-process-$ordinal"
    run="$attempt/$run_id"
    mkdir -p "$run"
    "$attempt/host" "$libs" "$model" "$rom" "$sigmoid" \
      "$ex2_unit" "$ex2_fraction" "$run" "$run_id" "$poison" \
      >"$run/stdout.log" 2>"$run/stderr.log"
  done
done

python3 - "$attempt" "$repo" <<'PY'
import hashlib, json, pathlib, sys
root, repo = map(pathlib.Path, sys.argv[1:])
runs = sorted(p for p in root.glob('connected-distortion-l3-l91-*') if p.is_dir())
if len(runs) != 4:
    raise SystemExit('fresh process count failed')
variants, outputs, pids = set(), set(), set()
for run in runs:
    early = json.loads((run/'run-receipt.json').read_text())
    tail = json.loads((run/'l20-l87'/'run-receipt.json').read_text())
    if early['process_id'] != tail['process_id'] or early['process_id'] in pids:
        raise SystemExit(f'same/fresh process gate failed: {run.name}')
    pids.add(early['process_id'])
    if early['result'] != 'pass' or tail['result'] != 'pass':
        raise SystemExit(f'run failed: {run.name}')
    if early['input_mismatched_bytes'] or early['canary_mismatched_bytes']:
        raise SystemExit(f'early integrity failed: {run.name}')
    for key in ('input_model_mismatched_bytes', 'canary_mismatched_bytes',
                'readonly_tail_mismatched_bytes'):
        if tail[key]:
            raise SystemExit(f'tail integrity failed: {run.name}/{key}')
    if (not early['l20_live_handoff'] or
            not early['distortion_live_device_handoffs'] or
            early['host_distortion_checkpoint_ingress'] or
            early['distortion_handoff_mismatched_bytes'] or
            early['allocation37_poison_halfwords']):
        raise SystemExit(f'handoff/write coverage failed: {run.name}')
    checkpoints = dict(early['checkpoint_sha256'])
    checkpoints.update({f'launch-{int(k.removeprefix("launch-"))+4:06d}': v
                        for k,v in tail['checkpoint_sha256'].items()})
    expected = {f'launch-{i:06d}' for i in range(3, 92)}
    if set(checkpoints) != expected or len(checkpoints) != 89:
        raise SystemExit(f'checkpoint coverage failed: {run.name}')
    combined = {
        'schema': 'neuralstudio.fg.connected-distortion-l3-l91-run.v1',
        'backend': 'mac-metal', 'run_id': run.name,
        'process_id': early['process_id'], 'poison_byte': early['poison_byte'],
        'launch_range': [3, 91], 'launch_count': 89,
        'checkpoint_sha256': checkpoints,
        'allocation37_sha256': early['allocation37_sha256'],
        'live_device_handoff_l19_l20': True,
        'live_distortion_adapter_handoffs': True,
        'distortion_checkpoint_ingress': False,
        'host_intermediate_ingress': False,
        'expected_or_archived_tensor_ingress': False,
        'address_fields_recorded': False,
        'final_output_bytes': tail['final_output_bytes'],
        'final_output_sha256': tail['final_output_sha256'], 'result': 'pass'}
    (run/'combined-receipt.json').write_text(
        json.dumps(combined, indent=2, sort_keys=True)+'\n')
    variants.add(json.dumps(checkpoints, sort_keys=True))
    outputs.add(tail['final_output_sha256'])
if len(variants) != 1 or len(outputs) != 1:
    raise SystemExit('two-poison invariance failed')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
first = json.loads((runs[0]/'combined-receipt.json').read_text())
receipt = {
    'schema': 'neuralstudio.fg.connected-distortion-l3-l91-mac-backend.v1',
    'fresh_process_count': 4, 'poison_bytes': [90, 165],
    'launch_range': [3, 91], 'launch_count': 89,
    'allocation37_sha256': first['allocation37_sha256'],
    'invariant_output_sha256': first['final_output_sha256'],
    'all_checkpoint_two_poison_invariant': True,
    'full_write_coverage_from_two_poison_invariance': True,
    'live_device_handoff_l19_l20': True,
    'live_distortion_adapter_handoffs': True,
    'distortion_checkpoint_ingress': False,
    'host_intermediate_ingress': False,
    'expected_or_archived_tensor_ingress': False,
    'address_fields_recorded': False,
    'early_source_sha256': sha(repo/'src/fg_connected_distortion_l3_l87/macos_connected_distortion.mm'),
    'live_handoff_source_sha256': sha(repo/'src/fg_connected_distortion_l3_l87/macos_live_handoff.mm'),
    'accepted_tail_source_sha256': sha(repo/'src/fg_connected_l20_l87/macos_connected_l20_l87.mm'),
    'generated_tail_source_sha256': sha(root/'build/macos_full_tail_live_main.mm'),
    'result': 'pass'}
(root/'backend-receipt.json').write_text(
    json.dumps(receipt, indent=2, sort_keys=True)+'\n')
print(f"invariant_output_sha256={first['final_output_sha256']}")
PY

echo "attempt_root=$attempt"
echo "result=pass"
