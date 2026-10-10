#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
loop_attempt="${1:-$repo/build/fg_runtime/macos/live-alias-v4-hmma-zero}"
clz_attempt="${2:-$repo/build/fg_runtime/macos/live-alias-clz-v1}"
output="${3:-$repo/build/fg_runtime/benchmark/clz-$(date -u +%Y%m%dT%H%M%SZ)}"
model="${4:-$repo/build/fg_default_readback/eval0-default-all-a5-a/runtime-default-buffers-post-evaluate-0}"
rom="${5:-$repo/build/fg_connected_tail/macos-l56-l87/attempt-exact-v1/sm120-rcp-correction-rom.bin}"
sigmoid="${6:-$repo/build/fg_postprocess_l84/rtx/oracle-006/sigmoid-process-a/sm120-half-logit-sigmoid-lut.bin}"
unit="${7:-$repo/build/fg_math_ex2/compact/attempt-v4/packed-a/unit-corrections.bin}"
fraction="${8:-$repo/build/fg_math_ex2/compact/attempt-v4/packed-a/fraction-corrections.bin}"
if [[ -e "$output" ]]; then
  echo "Benchmark output exists; preserve it: $output" >&2
  exit 2
fi
mkdir -p "$output/loop" "$output/clz"

run_one() {
  local mode="$1" ordinal="$2" attempt
  if [[ "$mode" == loop ]]; then attempt="$loop_attempt"; else attempt="$clz_attempt"; fi
  "$attempt/fg-runtime-benchmark" "$attempt/libs" "$model" "$rom" \
    "$sigmoid" "$unit" "$fraction" "$output/$mode/run-$ordinal.json" \
    "fg-runtime-clz-$mode-$ordinal" a5 compact live \
    >"$output/$mode/run-$ordinal.stdout.log" \
    2>"$output/$mode/run-$ordinal.stderr.log"
}

# Interleave to reduce run-order and thermal bias.
run_one loop 1
run_one clz 1
run_one clz 2
run_one loop 2
run_one loop 3
run_one clz 3

python3 - "$repo" "$loop_attempt" "$clz_attempt" "$output" <<'PY'
import hashlib,json,math,pathlib,statistics,sys
repo,loop_attempt,clz_attempt,output=map(pathlib.Path,sys.argv[1:])
expected=json.loads((repo/'evidence/specs/fg-runtime-v1/live-alias/verification.json').read_text())['output_sha256']
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
def percentile(values,fraction):
    ordered=sorted(values);position=(len(ordered)-1)*fraction
    lo=math.floor(position);hi=math.ceil(position)
    return ordered[lo] if lo==hi else ordered[lo]+(ordered[hi]-ordered[lo])*(position-lo)
def mode_receipt(mode,attempt):
    paths=sorted((output/mode).glob('run-*.json'))
    runs=[json.loads(path.read_text()) for path in paths]
    if len(paths)!=3 or len({run['process_id'] for run in runs})!=3:
        raise SystemExit(mode+': fresh-process coverage')
    values=[]
    for ordinal,run in enumerate(runs,1):
        if (run.get('run_id')!=f'fg-runtime-clz-{mode}-{ordinal}' or
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
    hmma_mode=backend.get('hmma_mode','loop')
    if hmma_mode!=mode: raise SystemExit(mode+': HMMA mode')
    lib_names=('conv_3x3','conv_1x1','conv_pre','conv0','conv1','conv2','bot0','bot1',
               'block1_conv0','block1_conv1','block1_conv2','block1_bot0','block1_bot1')
    return {
      'hmma_mode':mode,
      'process_ids':[run['process_id'] for run in runs],
      'raw_receipt_sha256':[sha(path) for path in paths],
      'evaluate_wall_ms':values,
      'p50_evaluate_wall_ms':percentile(values,.5),
      'p95_evaluate_wall_ms':percentile(values,.95),
      'p99_evaluate_wall_ms':percentile(values,.99),
      'mean_evaluate_wall_ms':statistics.mean(values),
      'stddev_evaluate_wall_ms':statistics.pstdev(values),
      'max_evaluate_wall_ms':max(values),
      'session_create_wall_ms':[run['session_create_wall_ms'] for run in runs],
      'thermal_states_after':[run['thermal_state_after'] for run in runs],
      'binary_sha256':sha(attempt/'fg-runtime-benchmark'),
      'runtime_tail_sha256':sha(attempt/'build/runtime_tail.mm'),
      'hmma_metallib_sha256':{name:sha(attempt/'libs'/f'{name}.metallib') for name in lib_names},
    }
loop=mode_receipt('loop',loop_attempt);clz=mode_receipt('clz',clz_attempt)
if len(set(loop['process_ids']+clz['process_ids']))!=6:
    raise SystemExit('cross-mode process freshness')
receipt={
  'schema':'neuralstudio.fg.runtime-clz-benchmark-comparison.v1',
  'profile':'640x360-compact-live-early-aliases',
  'diagnostic_component_only':True,
  'production_benchmark':False,
  'timing_class':'synchronous-evaluate-wall-not-input-latency',
  'profiling_instrumentation':False,
  'counter_attachment':False,
  'execution_order':['loop-1','clz-1','clz-2','loop-2','loop-3','clz-3'],
  'independent_processes_per_mode':3,
  'measured_frames_per_process':15,
  'untimed_correctness_before_and_after':True,
  'output_hashes_comparable':True,
  'accepted_output_sha256':expected,
  'accepted_checkpoint_root_sha256':'2b13a70a5d44659a6912c872a01adfd0e9eceaa02b668743cef82a0ea088a156',
  'previous_live_hotpath_p50_ms':120.519042,
  'loop':loop,'clz':clz,
  'clz_vs_interleaved_loop_p50_speedup':loop['p50_evaluate_wall_ms']/clz['p50_evaluate_wall_ms'],
  'clz_vs_previous_live_hotpath_p50_speedup':120.519042/clz['p50_evaluate_wall_ms'],
  'expected_or_archived_intermediate_ingress':False,
  'address_fields_recorded':False,
  'result':'pass',
}
normalized={key:value for key,value in receipt.items() if key not in ('loop','clz')}
receipt['normalized_root_sha256']=hashlib.sha256(
    json.dumps(normalized,sort_keys=True,separators=(',',':')).encode()).hexdigest()
(output/'comparison.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print('loop_p50_evaluate_wall_ms='+str(loop['p50_evaluate_wall_ms']))
print('clz_p50_evaluate_wall_ms='+str(clz['p50_evaluate_wall_ms']))
print('clz_vs_loop_p50_speedup='+str(receipt['clz_vs_interleaved_loop_p50_speedup']))
print('comparison_sha256='+sha(output/'comparison.json'))
print('result=pass')
PY

echo "benchmark_output=$output"
