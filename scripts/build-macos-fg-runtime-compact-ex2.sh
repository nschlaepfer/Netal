#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
attempt="${1:-$repo/build/fg_runtime/macos/compact-$(date -u +%Y%m%dT%H%M%SZ)}"
model="${2:-$repo/build/fg_default_readback/eval0-default-all-a5-a/runtime-default-buffers-post-evaluate-0}"
rom="${3:-$repo/build/fg_connected_tail/macos-l56-l87/attempt-exact-v1/sm120-rcp-correction-rom.bin}"
sigmoid="${4:-$repo/build/fg_postprocess_l84/rtx/oracle-006/sigmoid-process-a/sm120-half-logit-sigmoid-lut.bin}"
unit="${5:-$repo/build/fg_math_ex2/compact/attempt-v4/packed-a/unit-corrections.bin}"
fraction="${6:-$repo/build/fg_math_ex2/compact/attempt-v4/packed-a/fraction-corrections.bin}"
overlay="$attempt-overlay"
if [[ -e "$attempt" || -e "$overlay" ]]; then
  echo "Attempt/overlay exists; preserve it: $attempt / $overlay" >&2
  exit 2
fi
mkdir -p "$overlay"

python3 "$repo/scripts/generate-netal-fg-overlays.py" "$repo" "$overlay" "${FG_RUNTIME_LIVE_HMMA_ZERO:-0}"

conv3_override=""
bot0_override=""
if [[ "${FG_RUNTIME_LIVE_HMMA_ZERO:-0}" == 1 ]]; then
  conv3_override="$overlay/conv_3x3_live_zero.metal"
  bot0_override="$overlay/block0_bot0_live_zero.metal"
fi

reuse_libs="${FG_RUNTIME_REUSE_LIBS:-$repo/build/fg_runtime/macos/reference-v2-checkpoints/libs}"
if [[ "${FG_RUNTIME_HMMA_MODE:-loop}" != loop ]]; then
  # Every HMMA consumer must be rebuilt under the selected exact override.
  reuse_libs=""
fi

FG_RUNTIME_EX2_MODE=compact \
FG_RUNTIME_REUSE_LIBS="$reuse_libs" \
FG_EX2_UNIT_TABLE="$unit" \
FG_EX2_FRACTION_TABLE="$fraction" \
FG_RUNTIME_L52_SOURCE="$overlay/l52_compact.metal" \
FG_RUNTIME_L54_SOURCE="$overlay/l54_compact.metal" \
FG_RUNTIME_CONV3_SOURCE="$conv3_override" \
FG_RUNTIME_BOT0_SOURCE="$bot0_override" \
  "$repo/scripts/build-macos-fg-runtime.sh" "$attempt" "$model" "$rom" \
    "$sigmoid"

python3 - "$attempt" "$overlay" "$unit" "$fraction" <<'PY'
import hashlib, json, pathlib, sys
attempt, overlay, unit, fraction = map(pathlib.Path, sys.argv[1:])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
backend = json.loads((attempt/'backend-receipt.json').read_text())
if backend['ex2_mode'] != 'compact' or backend['result'] != 'pass':
    raise SystemExit('compact runtime backend failed')
receipt = {
    'schema': 'neuralstudio.fg.runtime-compact-ex2-promotion.v1',
    'checkpoint_root_sha256': backend['checkpoint_root_sha256'],
    'hmma_mode': backend['hmma_mode'],
    'output_sha256': backend['output_sha256'],
    'fresh_process_count': backend['fresh_process_count'],
    'poison_bytes': backend['poison_bytes'],
    'metal_resource_allocations_per_generated_frame':
        backend['metal_resource_allocations_per_generated_frame'],
    'unit_correction': {'bytes': unit.stat().st_size, 'sha256': sha(unit)},
    'fraction_correction': {
        'bytes': fraction.stat().st_size, 'sha256': sha(fraction)},
    'l52_compact_source_sha256': sha(overlay/'l52_compact.metal'),
    'l54_compact_source_sha256': sha(overlay/'l54_compact.metal'),
    'expected_or_archived_intermediate_ingress': False,
    'result': 'pass',
}
(attempt/'compact-promotion-receipt.json').write_text(
    json.dumps(receipt, indent=2, sort_keys=True)+'\n')
print('compact_promotion_receipt_sha256='+
      sha(attempt/'compact-promotion-receipt.json'))
print('result=pass')
PY
