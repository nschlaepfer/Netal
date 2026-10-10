#!/usr/bin/env bash
# Four fresh offline processes (poisons a5/5a x processes a/b) of an existing runtime attempt in the current
# environment (e.g. NSFG_FUSED_UPSAMPLE=1), gated against an expected verification receipt: the checkpoint root
# (1,149 checkpoints over 16 frames) and all 15 output hashes. Same process contract as build-macos-fg-runtime.sh.
# With NSFG_FUSED_UPSAMPLE set, every process must also report the fused path in its log (no silent no-op).
#   scripts/fg-runtime-receipt.sh <attempt> <tag> [expected verification.json]
# FG_RUNTIME_RECEIPT_PREFIX (optional, word-split) prefixes every process, e.g. a GPU lock runner.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
attempt="$1"; tag="$2"; expected="${3:-$repo/evidence/specs/fg-runtime-v1/live-zero-mask/verification.json}"
model="$repo/build/fg_default_readback/eval0-default-all-a5-a/runtime-default-buffers-post-evaluate-0"
rom="$repo/build/fg_connected_tail/macos-l56-l87/attempt-exact-v1/sm120-rcp-correction-rom.bin"
sigmoid="$repo/build/fg_postprocess_l84/rtx/oracle-006/sigmoid-process-a/sm120-half-logit-sigmoid-lut.bin"
unit="$repo/build/fg_math_ex2/compact/attempt-v4/packed-a/unit-corrections.bin"
fraction="$repo/build/fg_math_ex2/compact/attempt-v4/packed-a/fraction-corrections.bin"
out="$attempt/receipt-$tag"
if [[ -e "$out" ]]; then echo "receipt output exists; preserve it: $out" >&2; exit 2; fi
mkdir -p "$out"
for poison in a5 5a; do
  for ordinal in a b; do
    run_id="fg-runtime-poison-$poison-process-$ordinal"; run="$out/$run_id"; mkdir -p "$run"
    ${FG_RUNTIME_RECEIPT_PREFIX:-} "$attempt/fg-runtime-offline" "$attempt/libs" "$model" "$rom" "$sigmoid" \
      "$unit" "$fraction" "$run" "$run_id" "$poison" compact live >"$run/stdout.log" 2>"$run/stderr.log" \
      || echo "$run_id exit $?" >> "$out/exits.log"
  done
done
python3 - "$out" "$expected" "${NSFG_FUSED_UPSAMPLE:-}" <<'PY'
import hashlib, json, pathlib, sys
out, expected_path, fused = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
expected = json.loads(expected_path.read_text())
runs = sorted(p for p in out.glob('fg-runtime-poison-*-process-*') if p.is_dir())
receipts, problems = [], []
for run in runs:
    path = run / 'run-receipt.json'
    if not path.exists():
        problems.append(f'{run.name}: no run receipt'); continue
    receipts.append(json.loads(path.read_text()))
    if fused and 'fused bilinear fetch on launch' not in (run / 'stderr.log').read_text():
        problems.append(f'{run.name}: fused path not reported')
if len(receipts) != 4 or len({r['process_id'] for r in receipts}) != 4: problems.append('four fresh processes')
if sorted(r['poison_byte'] for r in receipts) != [90, 90, 165, 165]: problems.append('poison coverage')
if any(r['result'] != 'pass' for r in receipts): problems.append('a process failed')
if any(not r['resource_allocations_per_generated_frame'][0] or any(r['resource_allocations_per_generated_frame'][1:]) for r in receipts):
    problems.append('post-warm-up allocations')
roots = {hashlib.sha256(json.dumps(r['checkpoint_sha256'], sort_keys=True, separators=(',', ':')).encode()).hexdigest() for r in receipts}
outputs = {json.dumps(r['output_sha256']) for r in receipts}
if len(roots) != 1 or len(outputs) != 1: problems.append('poison/process invariance')
root = next(iter(roots)) if len(roots) == 1 else ''
if root != expected['checkpoint_root_sha256']: problems.append(f'checkpoint root {root or "?"} != expected')
outputs_equal = bool(receipts) and receipts[0]['output_sha256'] == expected['output_sha256']
if not outputs_equal: problems.append('output hashes differ from expected')
summary = {'schema': 'neuralstudio.fg.runtime-candidate-receipt.v1', 'nsfg_fused_upsample': fused,
           'expected': str(expected_path), 'fresh_process_count': len(receipts),
           'checkpoint_root_sha256': root, 'expected_checkpoint_root_sha256': expected['checkpoint_root_sha256'],
           'output_sha256_equal_expected': outputs_equal,
           'allocations_per_generated_frame': receipts[0]['resource_allocations_per_generated_frame'] if receipts else None,
           'problems': problems, 'result': 'pass' if not problems else 'FAIL'}
(out / 'receipt.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
print(json.dumps(summary, indent=2, sort_keys=True))
sys.exit(0 if not problems else 3)
PY
