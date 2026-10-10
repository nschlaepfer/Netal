#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
attempt="${1:-$repo/build/fg_runtime/macos/clz-$(date -u +%Y%m%dT%H%M%SZ)}"

FG_RUNTIME_HMMA_MODE=clz \
FG_RUNTIME_AUXILIARY_MODE="${FG_RUNTIME_AUXILIARY_MODE:-live}" \
FG_RUNTIME_LIVE_HMMA_ZERO="${FG_RUNTIME_LIVE_HMMA_ZERO:-1}" \
  "$repo/scripts/build-macos-fg-runtime-compact-ex2.sh" "$attempt" \
    "${2:-$repo/build/fg_default_readback/eval0-default-all-a5-a/runtime-default-buffers-post-evaluate-0}" \
    "${3:-$repo/build/fg_connected_tail/macos-l56-l87/attempt-exact-v1/sm120-rcp-correction-rom.bin}" \
    "${4:-$repo/build/fg_postprocess_l84/rtx/oracle-006/sigmoid-process-a/sm120-half-logit-sigmoid-lut.bin}" \
    "${5:-$repo/build/fg_math_ex2/compact/attempt-v4/packed-a/unit-corrections.bin}" \
    "${6:-$repo/build/fg_math_ex2/compact/attempt-v4/packed-a/fraction-corrections.bin}"

python3 - "$attempt" "$repo" <<'PY'
import hashlib,json,pathlib,sys
attempt,repo=map(pathlib.Path,sys.argv[1:])
overlay=pathlib.Path(str(attempt)+'-overlay')
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
backend=json.loads((attempt/'backend-receipt.json').read_text())
if backend.get('hmma_mode')!='clz' or backend.get('result')!='pass':
    raise SystemExit('CLZ runtime backend failed')
sources=[
  'src/fg_conv/conv_3x3.metal',
  'src/fg_production_contract_l3_l87/conv_1x1_temporal.metal',
  'src/fg_conv/custom_conv_pre.metal',
  'src/fg_conv/custom_conv0_c8.metal',
  'src/fg_conv/custom_conv1.metal',
  'src/fg_conv/custom_conv2.metal',
  'src/fg_connected_custom_l56_l83/connected_block0_bot0.metal',
  'src/fg_production_contract_l3_l87/connected_block0_bot1_production.metal',
  'src/fg_conv/custom_block1_conv0.metal',
  'src/fg_conv/custom_block1_conv1.metal',
  'src/fg_conv/custom_block1_conv2.metal',
  'src/fg_connected_custom_l56_l83/connected_block1_bot0.metal',
  'src/fg_connected_custom_l56_l83/connected_block1_bot1.metal',
]
for source in sources:
    if 'hmma_exact.metalh' not in (repo/source).read_text():
        raise SystemExit('CLZ source coverage drift: '+source)
compiled_manifest=(attempt/'build/hmma-clz-sources.txt').read_text().splitlines()
expected_manifest=[
  str(overlay/'conv_3x3_live_zero.metal'),
  'src/fg_production_contract_l3_l87/conv_1x1_temporal.metal',
  'src/fg_conv/custom_conv_pre.metal',
  'src/fg_conv/custom_conv0_c8.metal',
  'src/fg_conv/custom_conv1.metal',
  'src/fg_conv/custom_conv2.metal',
  str(overlay/'block0_bot0_live_zero.metal'),
  'src/fg_production_contract_l3_l87/connected_block0_bot1_production.metal',
  'src/fg_conv/custom_block1_conv0.metal',
  'src/fg_conv/custom_block1_conv1.metal',
  'src/fg_conv/custom_block1_conv2.metal',
  'src/fg_connected_custom_l56_l83/connected_block1_bot0.metal',
  'src/fg_connected_custom_l56_l83/connected_block1_bot1.metal',
]
if compiled_manifest != expected_manifest:
    raise SystemExit('CLZ compiled-source manifest mismatch')
receipt={
  'schema':'neuralstudio.fg.runtime-clz-promotion.v1',
  'hmma_mode':'clz',
  'hmma_source_count':len(sources),
  'compiled_hmma_source_manifest':compiled_manifest,
  'hmma_source_sha256':{source:sha(repo/source) for source in sources},
  'compiled_overlay_sha256':{
    'conv_3x3_live_zero.metal':sha(overlay/'conv_3x3_live_zero.metal'),
    'block0_bot0_live_zero.metal':sha(overlay/'block0_bot0_live_zero.metal'),
    'l52_compact.metal':sha(overlay/'l52_compact.metal'),
    'l54_compact.metal':sha(overlay/'l54_compact.metal'),
  },
  'hmma_metallib_sha256':{
    name:sha(attempt/'libs'/f'{name}.metallib') for name in (
      'conv_3x3','conv_1x1','conv_pre','conv0','conv1','conv2','bot0','bot1',
      'block1_conv0','block1_conv1','block1_conv2','block1_bot0','block1_bot1')
  },
  'override_source_sha256':sha(repo/'src/fg_conv_simd_l20/hmma_exact_clz_override.metalh'),
  'clz_source_sha256':sha(repo/'src/fg_conv_simd_l20/hmma_exact_clz.metalh'),
  'checkpoint_root_sha256':backend['checkpoint_root_sha256'],
  'output_sha256':backend['output_sha256'],
  'fresh_process_count':backend['fresh_process_count'],
  'poison_bytes':backend['poison_bytes'],
  'metal_resource_allocations_per_generated_frame':
      backend['metal_resource_allocations_per_generated_frame'],
  'expected_or_archived_intermediate_ingress':False,
  'result':'pass',
}
(attempt/'clz-promotion-receipt.json').write_text(
    json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print('clz_promotion_receipt_sha256='+sha(attempt/'clz-promotion-receipt.json'))
print('result=pass')
PY
