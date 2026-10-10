#!/bin/bash
set -euo pipefail

repository="$(cd "$(dirname "$0")/.." && pwd)"
attempt_root="${1:-$repository/build/fg_math_ex2/fast-tables/attempt-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ -e "$attempt_root" ]]; then
  echo "attempt exists: $attempt_root" >&2
  exit 2
fi
mkdir -p "$attempt_root/build"
source_root="$repository/src/fg_math_ex2"
xcrun -sdk macosx metal -std=metal3.1 -c \
  "$source_root/fast_table_probe.metal" -o "$attempt_root/build/probe.air"
xcrun -sdk macosx metallib "$attempt_root/build/probe.air" \
  -o "$attempt_root/build/probe.metallib"
xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc -Wall -Wextra -Werror \
  -Wno-deprecated-declarations "$source_root/macos_fast_table_probe.mm" \
  -framework Foundation -framework Metal -framework Security \
  -o "$attempt_root/build/host"

for ordinal in a b; do
  run_id="process-$ordinal"
  "$attempt_root/build/host" "$attempt_root/build/probe.metallib" \
    "$attempt_root/$run_id" "$run_id" \
    >"$attempt_root/$run_id.stdout.log" \
    2>"$attempt_root/$run_id.stderr.log"
done

PYTHONDONTWRITEBYTECODE=1 python3 - "$attempt_root" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);runs=[]
for path in sorted(root.glob('process-*/run-receipt.json')):
    runs.append(json.loads(path.read_text()))
assert len(runs)==2 and len({r['process_id'] for r in runs})==2
assert all(r['result']=='pass' for r in runs)
assert len({r['unit_sha256'] for r in runs})==1
assert len({r['fraction_sha256'] for r in runs})==1
result={'schema':'neuralstudio.fg.sm120-ex2-fast-table-mac-backend.v1',
        'fresh_process_count':2,
        'unit_records':runs[0]['unit_records'],
        'unit_sha256':runs[0]['unit_sha256'],
        'fraction_records':runs[0]['fraction_records'],
        'fraction_sha256':runs[0]['fraction_sha256'],
        'canary_mismatched_bytes':sum(r['canary_mismatched_bytes'] for r in runs),
        'address_fields_recorded':False,'result':'pass'}
result['receipt_root_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest()
(root/'backend-receipt.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print('receipt_root_sha256='+result['receipt_root_sha256'])
print('result=pass')
PY
