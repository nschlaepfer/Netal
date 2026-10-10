#!/bin/bash
set -euo pipefail

repository="$(cd "$(dirname "$0")/.." && pwd)"
unit_table="${1:?unit table required}"
fraction_table="${2:?fraction table required}"
identity_records="${3:?identity records required}"
attempt_root="${4:-$repository/build/fg_math_ex2/macos/attempt-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ -e "$attempt_root" ]]; then
  echo "attempt exists: $attempt_root" >&2
  exit 2
fi
mkdir -p "$attempt_root/build"
source_root="$repository/src/fg_math_ex2"
xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math \
  -I "$source_root" -c "$source_root/sm120_ex2_probe.metal" \
  -o "$attempt_root/build/probe.air"
xcrun -sdk macosx metallib "$attempt_root/build/probe.air" \
  -o "$attempt_root/build/probe.metallib"
xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc -Wall -Wextra -Werror \
  -Wno-deprecated-declarations "$source_root/macos_probe.mm" \
  -framework Foundation -framework Metal -framework Security \
  -o "$attempt_root/build/host"

for poison in a5 5a; do
  for ordinal in a b; do
    run_id="ex2-poison-$poison-process-$ordinal"
    "$attempt_root/build/host" "$attempt_root/build/probe.metallib" \
      "$unit_table" "$fraction_table" "$identity_records" \
      "$attempt_root/$run_id" "$run_id" "$poison" \
      >"$attempt_root/$run_id.stdout.log" \
      2>"$attempt_root/$run_id.stderr.log"
  done
done

PYTHONDONTWRITEBYTECODE=1 python3 - "$attempt_root" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); runs=[]
for path in sorted(root.glob('ex2-*/run-receipt.json')):
    runs.append(json.loads(path.read_text()))
assert len(runs)==4
assert len({r['process_id'] for r in runs})==4
assert {r['poison_byte'] for r in runs}=={0x5a,0xa5}
assert all(r['result']=='pass' for r in runs)
assert len({r['output_sha256'] for r in runs})==1
result={
 'schema':'neuralstudio.fg.sm120-ex2-mac-backend.v1',
 'fresh_process_count':4,
 'poison_bytes':[0x5a,0xa5],
 'input_records':runs[0]['input_records'],
 'input_sha256':runs[0]['input_sha256'],
 'output_sha256':runs[0]['output_sha256'],
 'total_output_mismatched_bytes':sum(r['output_mismatched_bytes'] for r in runs),
 'total_input_mismatched_bytes':sum(r['input_mismatched_bytes'] for r in runs),
 'total_canary_mismatched_bytes':sum(r['canary_mismatched_bytes'] for r in runs),
 'tables_immutable':all(r['unit_table_mismatched']==0 and r['fraction_table_mismatched']==0 for r in runs),
 'expected_output_bound_to_gpu':False,
 'address_fields_recorded':False,
 'result':'pass'}
result['receipt_root_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest()
(root/'backend-receipt.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print('receipt_root_sha256='+result['receipt_root_sha256'])
print('result=pass')
PY
