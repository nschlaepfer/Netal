#!/bin/bash
set -euo pipefail

repository="$(cd "$(dirname "$0")/.." && pwd)"
unit_table="${1:?unit table required}"
fraction_table="${2:?fraction table required}"
identity_records="${3:?identity records required}"
attempt_root="${4:-$repository/build/fg_math_ex2/compact/attempt-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ -e "$attempt_root" ]]; then
  echo "attempt exists: $attempt_root" >&2
  exit 2
fi

bash "$repository/scripts/build-macos-fg-sm120-ex2-fast-tables.sh" \
  "$attempt_root/fast-tables"
mkdir -p "$attempt_root/build" "$attempt_root/packed-a" \
  "$attempt_root/packed-b"
source_root="$repository/src/fg_math_ex2"

xcrun -sdk macosx clang++ -std=c++20 -O3 -Wall -Wextra -Werror \
  -Wno-deprecated-declarations \
  "$source_root/pack_fast_table_corrections.cc" -framework Security \
  -o "$attempt_root/build/pack"

for ordinal in a b; do
  packed_root="$attempt_root/packed-$ordinal"
  fast_root="$attempt_root/fast-tables/process-$ordinal"
  "$attempt_root/build/pack" "$unit_table" \
    "$fast_root/fast-unit-u32.bin" "$packed_root/unit-corrections.bin" \
    "$packed_root/unit-receipt.json" \
    >"$packed_root/unit.stdout.log" 2>"$packed_root/unit.stderr.log"
  "$attempt_root/build/pack" "$fraction_table" \
    "$fast_root/fast-fraction-u32.bin" \
    "$packed_root/fraction-corrections.bin" \
    "$packed_root/fraction-receipt.json" \
    >"$packed_root/fraction.stdout.log" \
    2>"$packed_root/fraction.stderr.log"
done
cmp "$attempt_root/packed-a/unit-corrections.bin" \
  "$attempt_root/packed-b/unit-corrections.bin"
cmp "$attempt_root/packed-a/fraction-corrections.bin" \
  "$attempt_root/packed-b/fraction-corrections.bin"

xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math \
  -I "$source_root" -c "$source_root/sm120_ex2_compact_probe.metal" \
  -o "$attempt_root/build/probe.air"
xcrun -sdk macosx metallib "$attempt_root/build/probe.air" \
  -o "$attempt_root/build/probe.metallib"
xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc -Wall -Wextra -Werror \
  -Wno-deprecated-declarations "$source_root/macos_compact_probe.mm" \
  -framework Foundation -framework Metal -framework Security \
  -o "$attempt_root/build/host"

for poison in a5 5a; do
  for ordinal in a b; do
    run_id="compact-poison-$poison-process-$ordinal"
    "$attempt_root/build/host" "$attempt_root/build/probe.metallib" \
      "$attempt_root/packed-a/unit-corrections.bin" \
      "$attempt_root/packed-a/fraction-corrections.bin" \
      "$identity_records" "$attempt_root/$run_id" "$run_id" "$poison" \
      >"$attempt_root/$run_id.stdout.log" \
      2>"$attempt_root/$run_id.stderr.log"
  done
done

PYTHONDONTWRITEBYTECODE=1 python3 - "$attempt_root" <<'PY'
import hashlib,json,sys
from pathlib import Path

root=Path(sys.argv[1])
fast=json.loads((root/'fast-tables/backend-receipt.json').read_text())
packs=[]
for ordinal in ('a','b'):
    base=root/f'packed-{ordinal}'
    packs.append({
        'unit':json.loads((base/'unit-receipt.json').read_text()),
        'fraction':json.loads((base/'fraction-receipt.json').read_text()),
    })
runs=[]
for path in sorted(root.glob('compact-*/run-receipt.json')):
    runs.append(json.loads(path.read_text()))
assert fast['result']=='pass' and fast['fresh_process_count']==2
assert len(packs)==2
assert all(item[k]['result']=='pass' for item in packs for k in ('unit','fraction'))
assert all(item[k]['reconstruction_mismatches']==0 for item in packs for k in ('unit','fraction'))
assert len({item['unit']['packed_sha256'] for item in packs})==1
assert len({item['fraction']['packed_sha256'] for item in packs})==1
assert len(runs)==4 and len({r['process_id'] for r in runs})==4
assert {r['poison_byte'] for r in runs}=={0x5a,0xa5}
assert all(r['result']=='pass' for r in runs)
assert len({r['output_sha256'] for r in runs})==1
assert len({r['unit_correction_sha256'] for r in runs})==1
assert len({r['fraction_correction_sha256'] for r in runs})==1
unit=packs[0]['unit'];fraction=packs[0]['fraction']
original_bytes=unit['records']*4+fraction['records']*4
compact_bytes=unit['packed_bytes']+fraction['packed_bytes']
result={
 'schema':'neuralstudio.fg.sm120-ex2-compact-mac-backend.v1',
 'fresh_fast_table_process_count':fast['fresh_process_count'],
 'fast_table_receipt_root_sha256':fast['receipt_root_sha256'],
 'fresh_correction_generation_count':len(packs),
 'correction_repeat_equal':True,
 'fresh_compact_process_count':len(runs),
 'poison_bytes':[0x5a,0xa5],
 'input_records':runs[0]['input_records'],
 'input_sha256':runs[0]['input_sha256'],
 'output_sha256':runs[0]['output_sha256'],
 'unit':unit,
 'fraction':fraction,
 'original_table_bytes':original_bytes,
 'compact_correction_bytes':compact_bytes,
 'compression_ratio':original_bytes/compact_bytes,
 'saved_bytes':original_bytes-compact_bytes,
 'total_output_mismatched_bytes':sum(r['output_mismatched_bytes'] for r in runs),
 'total_input_mismatched_bytes':sum(r['input_mismatched_bytes'] for r in runs),
 'total_canary_mismatched_bytes':sum(r['canary_mismatched_bytes'] for r in runs),
 'corrections_immutable':all(
     r['unit_correction_mismatched']==0 and
     r['fraction_correction_mismatched']==0 for r in runs),
 'expected_output_bound_to_gpu':False,
 'address_fields_recorded':False,
 'scope':'all finite nonpositive FP32 SM120 scalar EX2 inputs by composition with accepted exhaustive branch audits',
 'result':'pass'}
result['receipt_root_sha256']=hashlib.sha256(
    json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest()
(root/'backend-receipt.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print('receipt_root_sha256='+result['receipt_root_sha256'])
print('compact_correction_bytes='+str(compact_bytes))
print('compression_ratio='+str(result['compression_ratio']))
print('result=pass')
PY
