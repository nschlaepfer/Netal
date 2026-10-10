#!/bin/bash
set -euo pipefail

repository="$(cd "$(dirname "$0")/.." && pwd)"
unit_table="${1:?unit table required}"
fraction_table="${2:?fraction table required}"
unit_corrections="${3:?unit corrections required}"
fraction_corrections="${4:?fraction corrections required}"
identity_records="${5:?identity records required}"
attempt_root="${6:-$repository/build/fg_math_ex2/benchmark/attempt-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ -e "$attempt_root" ]]; then
  echo "attempt exists: $attempt_root" >&2
  exit 2
fi
mkdir -p "$attempt_root/build"
source_root="$repository/src/fg_math_ex2"

xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math \
  -I "$source_root" -c "$source_root/sm120_ex2_benchmark.metal" \
  -o "$attempt_root/build/benchmark.air"
xcrun -sdk macosx metallib "$attempt_root/build/benchmark.air" \
  -o "$attempt_root/build/benchmark.metallib"
xcrun -sdk macosx clang++ -std=c++20 -O3 -fobjc-arc -Wall -Wextra \
  -Werror -Wno-deprecated-declarations \
  "$source_root/macos_ex2_benchmark.mm" \
  -framework Foundation -framework Metal -framework Security \
  -o "$attempt_root/build/benchmark"

for ordinal in a b c d; do
  run_id="process-$ordinal"
  "$attempt_root/build/benchmark" \
    "$attempt_root/build/benchmark.metallib" \
    "$unit_table" "$fraction_table" "$unit_corrections" \
    "$fraction_corrections" "$identity_records" \
    "$attempt_root/$run_id.json" "$run_id" \
    >"$attempt_root/$run_id.stdout.log" \
    2>"$attempt_root/$run_id.stderr.log"
done

PYTHONDONTWRITEBYTECODE=1 python3 - "$attempt_root" <<'PY'
import hashlib,json,statistics,sys
from pathlib import Path

root=Path(sys.argv[1])
runs=[json.loads(path.read_text()) for path in sorted(root.glob('process-?.json'))]
assert len(runs)==4 and len({r['process_id'] for r in runs})==4
assert all(r['result']=='pass' for r in runs)
assert len({r['device'] for r in runs})==1
assert len({r['reference_output_sha256'] for r in runs})==1
assert len({r['compact_output_sha256'] for r in runs})==1
assert runs[0]['reference_output_sha256']==runs[0]['compact_output_sha256']
for run in runs:
    assert run['reference_mismatched_bytes']==0
    assert run['compact_mismatched_bytes']==0
    assert run['cross_mismatched_bytes']==0
    assert run['input_mismatched_bytes']==0
    assert run['canary_mismatched_bytes']==0
    assert run['table_mismatches']==0
reference=[r['reference_median_gpu_ms_per_dispatch'] for r in runs]
compact=[r['compact_median_gpu_ms_per_dispatch'] for r in runs]
result={
 'schema':'neuralstudio.fg.sm120-ex2-microbenchmark.v1',
 'device':runs[0]['device'],
 'fresh_process_count':len(runs),
 'records_per_dispatch':runs[0]['records_per_dispatch'],
 'warmup_iterations_per_process':runs[0]['warmup_iterations'],
 'measured_iterations_per_block':runs[0]['measured_iterations_per_block'],
 'block_count_per_backend_per_process':runs[0]['block_count_per_backend'],
 'reference_table_bytes':runs[0]['reference_table_bytes'],
 'compact_correction_bytes':runs[0]['compact_correction_bytes'],
 'reference_process_median_gpu_ms':reference,
 'compact_process_median_gpu_ms':compact,
 'reference_median_gpu_ms':statistics.median(reference),
 'compact_median_gpu_ms':statistics.median(compact),
 'median_speedup':statistics.median(reference)/statistics.median(compact),
 'output_sha256':runs[0]['reference_output_sha256'],
 'total_output_mismatched_bytes':sum(
     r['reference_mismatched_bytes']+r['compact_mismatched_bytes']+
     r['cross_mismatched_bytes'] for r in runs),
 'total_input_mismatched_bytes':sum(r['input_mismatched_bytes'] for r in runs),
 'total_canary_mismatched_bytes':sum(r['canary_mismatched_bytes'] for r in runs),
 'tables_immutable':all(r['table_mismatches']==0 for r in runs),
 'expected_output_bound_to_gpu':False,
 'address_fields_recorded':False,
 'scope':'isolated 1048576-record scalar EX2 throughput; not connected FG or game FPS',
 'result':'pass'}
result['receipt_root_sha256']=hashlib.sha256(
    json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest()
(root/'benchmark-receipt.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print('reference_median_gpu_ms='+str(result['reference_median_gpu_ms']))
print('compact_median_gpu_ms='+str(result['compact_median_gpu_ms']))
print('median_speedup='+str(result['median_speedup']))
print('receipt_root_sha256='+result['receipt_root_sha256'])
print('result=pass')
PY
