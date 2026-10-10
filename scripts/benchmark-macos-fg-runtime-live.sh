#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
attempt="${1:-$repo/build/fg_runtime/macos/live-alias-v4-hmma-zero}"
output="${2:-$repo/build/fg_runtime/benchmark/live-hotpath-$(date -u +%Y%m%dT%H%M%SZ)}"
model="${3:-$repo/build/fg_default_readback/eval0-default-all-a5-a/runtime-default-buffers-post-evaluate-0}"
rom="${4:-$repo/build/fg_connected_tail/macos-l56-l87/attempt-exact-v1/sm120-rcp-correction-rom.bin}"
sigmoid="${5:-$repo/build/fg_postprocess_l84/rtx/oracle-006/sigmoid-process-a/sm120-half-logit-sigmoid-lut.bin}"
unit="${6:-$repo/build/fg_math_ex2/compact/attempt-v4/packed-a/unit-corrections.bin}"
fraction="${7:-$repo/build/fg_math_ex2/compact/attempt-v4/packed-a/fraction-corrections.bin}"
if [[ -e "$output" ]]; then
  echo "Benchmark output exists; preserve it: $output" >&2
  exit 2
fi
mkdir -p "$output"

for ordinal in 1 2 3; do
  "$attempt/fg-runtime-benchmark" "$attempt/libs" "$model" "$rom" \
    "$sigmoid" "$unit" "$fraction" "$output/run-$ordinal.json" \
    "fg-runtime-live-hotpath-$ordinal" a5 compact live \
    >"$output/run-$ordinal.stdout.log" \
    2>"$output/run-$ordinal.stderr.log"
done

python3 - "$repo" "$attempt" "$output" <<'PY'
import hashlib,json,math,pathlib,statistics,sys
repo,attempt,output=map(pathlib.Path,sys.argv[1:])
import os
expected=json.loads((repo/os.environ.get('FG_RUNTIME_EXPECTED','evidence/specs/fg-runtime-v1/live-alias/verification.json')).read_text())['output_sha256']
paths=sorted(output.glob('run-*.json'));runs=[json.loads(p.read_text()) for p in paths]
if len(runs)!=3 or len({r['process_id'] for r in runs})!=3:raise SystemExit('live benchmark process coverage')
values=[]
for run in runs:
    if run['result']!='pass' or run['auxiliary_mode']!='live' or run['ex2_mode']!='compact' or run['profiling_enabled']:
        raise SystemExit('live benchmark run contract')
    if not run['output_hashes_comparable'] or run['pre_output_sha256']!=expected or run['measured_output_sha256']!=expected or run['post_output_sha256']!=expected:
        raise SystemExit('live benchmark output drift')
    if any(run['measured_metal_resource_allocations']):raise SystemExit('live benchmark allocation drift')
    values.extend(run['measured_evaluate_wall_ms'])
def percentile(f):
    ordered=sorted(values);p=(len(ordered)-1)*f;lo=math.floor(p);hi=math.ceil(p)
    return ordered[lo] if lo==hi else ordered[lo]+(ordered[hi]-ordered[lo])*(p-lo)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
baseline=486.992
receipt={'schema':'neuralstudio.fg.runtime-live-hotpath-benchmark.v1',
         'profile':'640x360-compact-live-early-aliases',
         'diagnostic_component_only':True,'production_benchmark':False,
         'timing_class':'synchronous-evaluate-wall-not-input-latency',
         'profiling_instrumentation':False,
         'untimed_correctness_before_and_after':True,
         'output_hashes_comparable':True,
         'independent_process_count':3,'measured_frame_count':45,
         'evaluate_wall_ms':values,'p50_evaluate_wall_ms':percentile(.5),
         'p95_evaluate_wall_ms':percentile(.95),'p99_evaluate_wall_ms':percentile(.99),
         'max_evaluate_wall_ms':max(values),'mean_evaluate_wall_ms':statistics.mean(values),
         'stddev_evaluate_wall_ms':statistics.pstdev(values),
         'profiled_reference_p50_ms':baseline,
         'profiled_reference_instrumented':True,
         'diagnostic_ratio_vs_profiled_reference':baseline/percentile(.5),
         'raw_receipt_sha256':[sha(p) for p in paths],
         'binary_sha256':sha(attempt/'fg-runtime-benchmark'),
         'runtime_tail_sha256':sha(attempt/'build/runtime_tail.mm'),
         'accepted_output_sha256':expected,'result':'pass'}
normalized={k:v for k,v in receipt.items() if k not in ('evaluate_wall_ms','raw_receipt_sha256')}
receipt['normalized_root_sha256']=hashlib.sha256(json.dumps(normalized,sort_keys=True,separators=(',',':')).encode()).hexdigest()
(output/'benchmark.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print('p50_evaluate_wall_ms='+str(receipt['p50_evaluate_wall_ms']))
print('diagnostic_ratio_vs_profiled_reference='+str(receipt['diagnostic_ratio_vs_profiled_reference']))
print('benchmark_sha256='+sha(output/'benchmark.json'))
print('result=pass')
PY

echo "benchmark_output=$output"
