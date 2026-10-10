#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
loop_attempt="${1:-$repo/build/fg_runtime/macos/live-alias-v4-hmma-zero}"
clz_attempt="${2:-$repo/build/fg_runtime/macos/live-alias-clz-v1}"
output="${3:-$repo/build/fg_runtime/profile/clz-$(date -u +%Y%m%dT%H%M%SZ)}"
model="${4:-$repo/build/fg_default_readback/eval0-default-all-a5-a/runtime-default-buffers-post-evaluate-0}"
rom="${5:-$repo/build/fg_connected_tail/macos-l56-l87/attempt-exact-v1/sm120-rcp-correction-rom.bin}"
sigmoid="${6:-$repo/build/fg_postprocess_l84/rtx/oracle-006/sigmoid-process-a/sm120-half-logit-sigmoid-lut.bin}"
unit="${7:-$repo/build/fg_math_ex2/compact/attempt-v4/packed-a/unit-corrections.bin}"
fraction="${8:-$repo/build/fg_math_ex2/compact/attempt-v4/packed-a/fraction-corrections.bin}"
if [[ -e "$output" ]]; then
  echo "Profile output exists; preserve it: $output" >&2
  exit 2
fi
mkdir -p "$output/loop" "$output/clz"

run_one() {
  local mode="$1" ordinal="$2" attempt
  if [[ "$mode" == loop ]]; then attempt="$loop_attempt"; else attempt="$clz_attempt"; fi
  FG_RUNTIME_PROFILE_SPANS=1 \
    "$attempt/fg-runtime-benchmark" "$attempt/libs" "$model" "$rom" \
      "$sigmoid" "$unit" "$fraction" "$output/$mode/run-$ordinal.json" \
      "fg-runtime-clz-profile-$mode-$ordinal" a5 compact live \
      >"$output/$mode/run-$ordinal.stdout.log" \
      2>"$output/$mode/run-$ordinal.stderr.log"
}

run_one loop 1
run_one clz 1
run_one clz 2
run_one loop 2
run_one loop 3
run_one clz 3

python3 - "$repo" "$loop_attempt" "$clz_attempt" "$output" <<'PY'
import hashlib,json,pathlib,statistics,sys
repo,loop_attempt,clz_attempt,output=map(pathlib.Path,sys.argv[1:])
expected=json.loads((repo/'evidence/specs/fg-runtime-v1/live-alias/verification.json').read_text())['output_sha256']
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
labels=['status','L3','L4','L5-high','L5-low','L5-pack','L6',
        'L7-texture-to-buffer','L7-kernel','L7-buffer-to-texture','L8','L19',
        'L20-L51','L52','L53','L54','L55','L55-to-L84-blit','L56-L83',
        'L84','L85','L86','L87','caller-output-copy']
domains=[0]*12+[1]*11+[2]
def mode_receipt(mode,attempt):
    paths=sorted((output/mode).glob('run-*.json'))
    runs=[json.loads(path.read_text()) for path in paths]
    if len(paths)!=3 or len({run['process_id'] for run in runs})!=3:
        raise SystemExit(mode+': fresh-process coverage')
    cpu=[[] for _ in labels];gpu=[[] for _ in labels]
    evaluate=[];cpu_sum=[];gpu_sum=[];unspanned=[]
    for ordinal,run in enumerate(runs,1):
        if (run.get('run_id')!=f'fg-runtime-clz-profile-{mode}-{ordinal}' or
            run.get('ex2_mode')!='compact' or run.get('auxiliary_mode')!='live' or
            run.get('poison_byte')!=165 or run.get('profiling_enabled') is not True or
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
        for wall,spans in zip(run['measured_evaluate_wall_ms'],run['measured_command_spans']):
            if (len(spans)!=24 or [span['ordinal'] for span in spans]!=list(range(24)) or
                [span['domain'] for span in spans]!=domains):
                raise SystemExit(mode+': span identity')
            frame_cpu=sum(span['cpu_commit_wait_ms'] for span in spans)
            frame_gpu=sum(span['gpu_ms'] for span in spans)
            evaluate.append(wall);cpu_sum.append(frame_cpu);gpu_sum.append(frame_gpu)
            unspanned.append(wall-frame_cpu)
            for index,span in enumerate(spans):
                cpu[index].append(span['cpu_commit_wait_ms']);gpu[index].append(span['gpu_ms'])
    backend=json.loads((attempt/'backend-receipt.json').read_text())
    if backend.get('hmma_mode','loop')!=mode:raise SystemExit(mode+': HMMA mode')
    return {
      'hmma_mode':mode,'process_ids':[run['process_id'] for run in runs],
      'raw_receipt_sha256':[sha(path) for path in paths],
      'evaluate_wall_p50_ms':statistics.median(evaluate),
      'command_commit_wait_sum_p50_ms':statistics.median(cpu_sum),
      'gpu_span_sum_p50_ms':statistics.median(gpu_sum),
      'unspanned_host_work_p50_ms':statistics.median(unspanned),
      'spans':[{'ordinal':index,'label':label,'domain':domains[index],
                'cpu_commit_wait_p50_ms':statistics.median(cpu[index]),
                'gpu_p50_ms':statistics.median(gpu[index]),
                'gpu_max_ms':max(gpu[index])}
               for index,label in enumerate(labels)],
      'binary_sha256':sha(attempt/'fg-runtime-benchmark'),
      'runtime_tail_sha256':sha(attempt/'build/runtime_tail.mm'),
    }
loop=mode_receipt('loop',loop_attempt);clz=mode_receipt('clz',clz_attempt)
if len(set(loop['process_ids']+clz['process_ids']))!=6:
    raise SystemExit('cross-mode process freshness')
receipt={
  'schema':'neuralstudio.fg.runtime-clz-boundary-profile.v1',
  'profile':'640x360-compact-live-early-aliases',
  'diagnostic_instrumentation':True,'production_benchmark':False,
  'timing_class':'synchronous-evaluate-wall-and-command-buffer-spans-not-input-latency',
  'execution_order':['loop-1','clz-1','clz-2','loop-2','loop-3','clz-3'],
  'independent_processes_per_mode':3,'measured_frames_per_process':15,
  'command_spans_per_frame':24,'selected_wait_boundaries':23,
  'untimed_correctness_before_and_after':True,'output_hashes_comparable':True,
  'accepted_output_sha256':expected,
  'accepted_checkpoint_root_sha256':'2b13a70a5d44659a6912c872a01adfd0e9eceaa02b668743cef82a0ea088a156',
  'loop':loop,'clz':clz,
  'clz_vs_loop_evaluate_p50_speedup':loop['evaluate_wall_p50_ms']/clz['evaluate_wall_p50_ms'],
  'clz_vs_loop_gpu_span_p50_speedup':loop['gpu_span_sum_p50_ms']/clz['gpu_span_sum_p50_ms'],
  'generic_l20_l51_gpu_speedup':loop['spans'][12]['gpu_p50_ms']/clz['spans'][12]['gpu_p50_ms'],
  'custom_l56_l83_gpu_speedup':loop['spans'][18]['gpu_p50_ms']/clz['spans'][18]['gpu_p50_ms'],
  'expected_or_archived_intermediate_ingress':False,'address_fields_recorded':False,
  'result':'pass',
}
normalized={key:value for key,value in receipt.items() if key not in ('loop','clz')}
receipt['normalized_root_sha256']=hashlib.sha256(
    json.dumps(normalized,sort_keys=True,separators=(',',':')).encode()).hexdigest()
(output/'profile.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
for key in ('clz_vs_loop_evaluate_p50_speedup','clz_vs_loop_gpu_span_p50_speedup',
            'generic_l20_l51_gpu_speedup','custom_l56_l83_gpu_speedup'):
    print(key+'='+str(receipt[key]))
print('profile_sha256='+sha(output/'profile.json'))
print('result=pass')
PY

echo "profile_output=$output"
