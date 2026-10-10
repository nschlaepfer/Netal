#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
clz_attempt="${1:-$repo/build/fg_runtime/macos/live-alias-clz-v1}"
u32_attempt="${2:-$repo/build/fg_runtime/macos/live-alias-u32-v1}"
output="${3:-$repo/build/fg_runtime/benchmark/u32-$(date -u +%Y%m%dT%H%M%SZ)}"
model="${4:-$repo/build/fg_default_readback/eval0-default-all-a5-a/runtime-default-buffers-post-evaluate-0}"
rom="${5:-$repo/build/fg_connected_tail/macos-l56-l87/attempt-exact-v1/sm120-rcp-correction-rom.bin}"
sigmoid="${6:-$repo/build/fg_postprocess_l84/rtx/oracle-006/sigmoid-process-a/sm120-half-logit-sigmoid-lut.bin}"
unit="${7:-$repo/build/fg_math_ex2/compact/attempt-v4/packed-a/unit-corrections.bin}"
fraction="${8:-$repo/build/fg_math_ex2/compact/attempt-v4/packed-a/fraction-corrections.bin}"
if [[ -e "$output" ]]; then
  echo "Benchmark output exists; preserve it: $output" >&2
  exit 2
fi
mkdir -p "$output/clz" "$output/u32"

run_one() {
  local mode="$1" ordinal="$2" attempt
  if [[ "$mode" == clz ]]; then attempt="$clz_attempt"; else attempt="$u32_attempt"; fi
  "$attempt/fg-runtime-benchmark" "$attempt/libs" "$model" "$rom" \
    "$sigmoid" "$unit" "$fraction" "$output/$mode/run-$ordinal.json" \
    "fg-runtime-u32-$mode-$ordinal" a5 compact live \
    >"$output/$mode/run-$ordinal.stdout.log" \
    2>"$output/$mode/run-$ordinal.stderr.log"
}

run_one clz 1
run_one u32 1
run_one u32 2
run_one clz 2
run_one clz 3
run_one u32 3

python3 - "$repo" "$clz_attempt" "$u32_attempt" "$output" <<'PY'
import hashlib,json,math,pathlib,statistics,sys
repo,clz_attempt,u32_attempt,output=map(pathlib.Path,sys.argv[1:])
expected=json.loads((repo/'evidence/specs/fg-runtime-v1/live-alias/verification.json').read_text())['output_sha256']
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
def percentile(values,fraction):
    ordered=sorted(values);position=(len(ordered)-1)*fraction
    lo=math.floor(position);hi=math.ceil(position)
    return ordered[lo] if lo==hi else ordered[lo]+(ordered[hi]-ordered[lo])*(position-lo)
def mode_receipt(mode,attempt):
    paths=sorted((output/mode).glob('run-*.json'));runs=[json.loads(path.read_text()) for path in paths]
    if len(paths)!=3 or len({run['process_id'] for run in runs})!=3:
        raise SystemExit(mode+': fresh-process coverage')
    values=[]
    for ordinal,run in enumerate(runs,1):
        if (run.get('run_id')!=f'fg-runtime-u32-{mode}-{ordinal}' or
            run.get('ex2_mode')!='compact' or run.get('auxiliary_mode')!='live' or
            run.get('poison_byte')!=165 or run.get('profiling_enabled') is not False or
            run.get('counter_attachment') is not False or run.get('result')!='pass' or
            run.get('pre_output_sha256')!=expected or
            run.get('measured_output_sha256')!=expected or
            run.get('post_output_sha256')!=expected or
            run.get('output_hashes_comparable') is not True or
            run.get('caller_owned_input_mismatches')!=0 or
            run.get('expected_or_archived_intermediate_ingress') is not False or
            run.get('address_fields_recorded') is not False or
            run.get('measured_metal_resource_allocations')!=[0]*15):
            raise SystemExit(mode+': raw run contract')
        values.extend(run['measured_evaluate_wall_ms'])
    backend=json.loads((attempt/'backend-receipt.json').read_text())
    if backend.get('hmma_mode')!=mode:raise SystemExit(mode+': HMMA mode')
    libs=('conv_3x3','conv_1x1','conv_pre','conv0','conv1','conv2','bot0','bot1',
          'block1_conv0','block1_conv1','block1_conv2','block1_bot0','block1_bot1')
    return {'hmma_mode':mode,'process_ids':[run['process_id'] for run in runs],
      'raw_receipt_sha256':[sha(path) for path in paths],'evaluate_wall_ms':values,
      'p50_evaluate_wall_ms':percentile(values,.5),'p95_evaluate_wall_ms':percentile(values,.95),
      'p99_evaluate_wall_ms':percentile(values,.99),'mean_evaluate_wall_ms':statistics.mean(values),
      'stddev_evaluate_wall_ms':statistics.pstdev(values),'max_evaluate_wall_ms':max(values),
      'session_create_wall_ms':[run['session_create_wall_ms'] for run in runs],
      'thermal_states_after':[run['thermal_state_after'] for run in runs],
      'binary_sha256':sha(attempt/'fg-runtime-benchmark'),
      'runtime_tail_sha256':sha(attempt/'build/runtime_tail.mm'),
      'hmma_metallib_sha256':{name:sha(attempt/'libs'/f'{name}.metallib') for name in libs}}
clz=mode_receipt('clz',clz_attempt);u32=mode_receipt('u32',u32_attempt)
if len(set(clz['process_ids']+u32['process_ids']))!=6:raise SystemExit('cross-mode freshness')
receipt={'schema':'neuralstudio.fg.runtime-u32-benchmark-comparison.v1',
  'profile':'640x360-compact-live-early-aliases','diagnostic_component_only':True,
  'production_benchmark':False,'timing_class':'synchronous-evaluate-wall-not-input-latency',
  'profiling_instrumentation':False,'counter_attachment':False,
  'execution_order':['clz-1','u32-1','u32-2','clz-2','clz-3','u32-3'],
  'independent_processes_per_mode':3,'measured_frames_per_process':15,
  'untimed_correctness_before_and_after':True,'output_hashes_comparable':True,
  'accepted_output_sha256':expected,
  'accepted_checkpoint_root_sha256':'2b13a70a5d44659a6912c872a01adfd0e9eceaa02b668743cef82a0ea088a156',
  'clz':clz,'u32':u32,
  'u32_vs_clz_p50_speedup':clz['p50_evaluate_wall_ms']/u32['p50_evaluate_wall_ms'],
  'expected_or_archived_intermediate_ingress':False,'address_fields_recorded':False,'result':'pass'}
normalized={key:value for key,value in receipt.items() if key not in ('clz','u32')}
receipt['normalized_root_sha256']=hashlib.sha256(
    json.dumps(normalized,sort_keys=True,separators=(',',':')).encode()).hexdigest()
(output/'comparison.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print('clz_p50_evaluate_wall_ms='+str(clz['p50_evaluate_wall_ms']))
print('u32_p50_evaluate_wall_ms='+str(u32['p50_evaluate_wall_ms']))
print('u32_vs_clz_p50_speedup='+str(receipt['u32_vs_clz_p50_speedup']))
print('comparison_sha256='+sha(output/'comparison.json'));print('result=pass')
PY

echo "benchmark_output=$output"
