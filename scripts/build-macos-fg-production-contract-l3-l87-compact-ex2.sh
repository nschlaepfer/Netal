#!/bin/bash
set -euo pipefail

repository="$(cd "$(dirname "$0")/.." && pwd)"
attempt="${1:-$repository/build/fg_production_contract_l3_l87/macos/compact-ex2-$(date -u +%Y%m%dT%H%M%SZ)}"
model="${2:-$repository/build/fg_default_readback/eval0-default-all-a5-a/runtime-default-buffers-post-evaluate-0}"
rom="${3:-$repository/build/fg_connected_tail/macos-l56-l87/attempt-exact-v1/sm120-rcp-correction-rom.bin}"
sigmoid="${4:-$repository/build/fg_postprocess_l84/rtx/oracle-006/sigmoid-process-a/sm120-half-logit-sigmoid-lut.bin}"
unit_corrections="${5:-$repository/build/fg_math_ex2/compact/attempt-v4/packed-a/unit-corrections.bin}"
fraction_corrections="${6:-$repository/build/fg_math_ex2/compact/attempt-v4/packed-a/fraction-corrections.bin}"
reference_attempt="${7:-$repository/build/fg_production_contract_l3_l87/macos/long16-v3-hmma-zero}"
overlay="$attempt.source-overlay"
if [[ -e "$attempt" || -e "$overlay" ]]; then
  echo "attempt or source overlay exists; preserve it" >&2
  exit 2
fi
mkdir -p "$overlay/scripts"
cp -R "$repository/src" "$overlay/src"
cp "$repository/scripts/build-macos-fg-production-contract-l3-l87.sh" \
  "$overlay/scripts/build-macos-fg-production-contract-l3-l87.sh"

PYTHONDONTWRITEBYTECODE=1 python3 - "$overlay" <<'PY'
from pathlib import Path
import sys

root=Path(sys.argv[1])

def rewrite(relative,replacements):
    path=root/relative
    text=path.read_text()
    for old,new,count in replacements:
        if text.count(old)!=count:
            raise SystemExit(f'compact source anchor drift: {relative}: {old!r}')
        text=text.replace(old,new)
    path.write_text(text)

common=[
    ('#include "../fg_math_ex2/sm120_ex2.metalh"',
     '#include "../fg_math_ex2/sm120_ex2_compact.metalh"',1),
    ('neuralstudio_fg_math::sm120_ex2_nonpositive(',
     'neuralstudio_fg_math::sm120_ex2_nonpositive_compact(',1),
]
rewrite('src/fg_production_contract_l3_l87/l52_production_exact.metal',
        common+[
          ('device const uint* ex2_unit_table','device const uchar* ex2_unit_table',1),
          ('device const uint* ex2_fraction_table','device const uchar* ex2_fraction_table',1),
        ])
rewrite('src/fg_production_contract_l3_l87/l54_production_exact.metal',
        common+[
          ('device const uint* ex2_unit_table','device const uchar* ex2_unit_table',2),
          ('device const uint* ex2_fraction_table','device const uchar* ex2_fraction_table',2),
        ])
rewrite('src/fg_production_contract_l3_l87/macos_temporal.mm',[
    ('ex2_unit_bytes.size()!=67108868u||Sha256(ex2_unit_bytes)!=\n'
     '         "40088ff59b58c4018114b4a2dd1fb5b124488b91e7cad79bdf3bf143bf267b75"',
     'ex2_unit_bytes.size()!=4194305u||Sha256(ex2_unit_bytes)!=\n'
     '         "60c2ae1a0869ce6f10ba093a33def68ecdbf4ba0d789cd186289586b63b9f234"',1),
    ('ex2_fraction_bytes.size()!=33554436u||Sha256(ex2_fraction_bytes)!=\n'
     '         "f7cfb2fd9df97ae01bc747b866e4928a79311e8443700f938859f43772f73d86"',
     'ex2_fraction_bytes.size()!=2097153u||Sha256(ex2_fraction_bytes)!=\n'
     '         "76bacb1adf8e53accdd6a40a64940c073f001e091232e9119b155f2ab25f1a13"',1),
])
PY

FG_EX2_UNIT_TABLE="$unit_corrections" \
FG_EX2_FRACTION_TABLE="$fraction_corrections" \
bash "$overlay/scripts/build-macos-fg-production-contract-l3-l87.sh" \
  "$attempt" "$model" "$rom" "$sigmoid"

PYTHONDONTWRITEBYTECODE=1 python3 - "$attempt" "$reference_attempt" \
  "$overlay" "$repository" "$unit_corrections" "$fraction_corrections" <<'PY'
import hashlib,json,sys
from pathlib import Path

attempt,reference,overlay,repository,unit_path,fraction_path=map(Path,sys.argv[1:])
compact_runs=sorted(p for p in attempt.glob('fg-production-contract-l3-l87-*') if p.is_dir())
reference_runs=sorted(p for p in reference.glob('fg-production-contract-l3-l87-*') if p.is_dir())
assert len(compact_runs)==4 and len(reference_runs)==4
assert len({json.loads((p/'temporal-receipt.json').read_text())['process_id'] for p in compact_runs})==4
reference_by_name={p.name:p for p in reference_runs}
checkpoint_variants=set();outputs=set();poisons=set();pids=set()
total_input_mismatches=0;total_canary_mismatches=0
for run in compact_runs:
    assert run.name in reference_by_name
    compact=json.loads((run/'temporal-receipt.json').read_text())
    prior=json.loads((reference_by_name[run.name]/'temporal-receipt.json').read_text())
    early=json.loads((run/'run-receipt.json').read_text())
    tails=[json.loads((run/f'generated-{frame}'/'l20-l87'/'run-receipt.json').read_text())
           for frame in range(1,16)]
    assert compact['result']=='pass' and early['result']=='pass'
    assert all(t['result']=='pass' for t in tails)
    assert compact['process_id']==early['process_id']
    assert all(t['process_id']==early['process_id'] for t in tails)
    assert compact['process_id'] not in pids
    pids.add(compact['process_id']);poisons.add(compact['poison_byte'])
    assert compact['checkpoint_sha256']==prior['checkpoint_sha256']
    assert compact['final_output_sha256']==prior['final_output_sha256']
    assert compact['allocation37_sha256']==prior['allocation37_sha256']
    assert compact['allocation29_sha256']==prior['allocation29_sha256']
    checkpoint_variants.add(json.dumps(compact['checkpoint_sha256'],sort_keys=True))
    outputs.add(json.dumps(compact['final_output_sha256']))
    total_input_mismatches+=early['input_mismatched_bytes']
    total_canary_mismatches+=early['canary_mismatched_bytes']
    total_input_mismatches+=sum(t['input_model_mismatched_bytes']+t['readonly_tail_mismatched_bytes'] for t in tails)
    total_canary_mismatches+=sum(t['canary_mismatched_bytes'] for t in tails)
assert poisons=={90,165} and len(checkpoint_variants)==1 and len(outputs)==1
assert total_input_mismatches==0 and total_canary_mismatches==0
first=json.loads((compact_runs[0]/'temporal-receipt.json').read_text())
canonical_checkpoints=json.dumps(first['checkpoint_sha256'],sort_keys=True,separators=(',',':')).encode()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result={
 'schema':'neuralstudio.fg.production-contract-l3-l87-compact-ex2-promotion.v1',
 'fresh_process_count':len(compact_runs),
 'poison_bytes':[90,165],
 'rendered_frame_count':16,
 'launch_count':1149,
 'checkpoint_count':len(first['checkpoint_sha256']),
 'checkpoint_root_sha256':hashlib.sha256(canonical_checkpoints).hexdigest(),
 'reference_checkpoint_exact':True,
 'two_poison_invariant':True,
 'invariant_output_sha256':first['final_output_sha256'],
 'allocation37_sha256':first['allocation37_sha256'],
 'allocation29_sha256':first['allocation29_sha256'],
 'unit_correction':{'bytes':unit_path.stat().st_size,'sha256':sha(unit_path)},
 'fraction_correction':{'bytes':fraction_path.stat().st_size,'sha256':sha(fraction_path)},
 'total_input_mismatched_bytes':total_input_mismatches,
 'total_canary_mismatched_bytes':total_canary_mismatches,
 'expected_or_archived_tensor_ingress':False,
 'address_fields_recorded':False,
 'generated_l52_sha256':sha(overlay/'src/fg_production_contract_l3_l87/l52_production_exact.metal'),
 'generated_l54_sha256':sha(overlay/'src/fg_production_contract_l3_l87/l54_production_exact.metal'),
 'generated_host_sha256':sha(overlay/'src/fg_production_contract_l3_l87/macos_temporal.mm'),
 'scope':'synthetic 640x360 reset plus 15 generated frames; not persistent caller-owned runtime or game capture',
 'result':'pass'}
result['receipt_root_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest()
(attempt/'compact-promotion-receipt.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print('checkpoint_root_sha256='+result['checkpoint_root_sha256'])
print('receipt_root_sha256='+result['receipt_root_sha256'])
print('result=pass')
PY
