#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
clz_attempt="${1:-$repo/build/fg_runtime/macos/live-alias-clz-v1}"
u32_attempt="${2:-$repo/build/fg_runtime/macos/live-alias-u32-v1}"
output="${3:-$repo/build/fg_runtime/profile/u32-$(date -u +%Y%m%dT%H%M%SZ)}"
model="${4:-$repo/build/fg_default_readback/eval0-default-all-a5-a/runtime-default-buffers-post-evaluate-0}"
rom="${5:-$repo/build/fg_connected_tail/macos-l56-l87/attempt-exact-v1/sm120-rcp-correction-rom.bin}"
sigmoid="${6:-$repo/build/fg_postprocess_l84/rtx/oracle-006/sigmoid-process-a/sm120-half-logit-sigmoid-lut.bin}"
unit="${7:-$repo/build/fg_math_ex2/compact/attempt-v4/packed-a/unit-corrections.bin}"
fraction="${8:-$repo/build/fg_math_ex2/compact/attempt-v4/packed-a/fraction-corrections.bin}"
if [[ -e "$output" ]]; then echo "Profile output exists; preserve it: $output" >&2; exit 2; fi
mkdir -p "$output/clz" "$output/u32"

run_one() {
  local mode="$1" ordinal="$2" attempt
  if [[ "$mode" == clz ]]; then attempt="$clz_attempt"; else attempt="$u32_attempt"; fi
  FG_RUNTIME_PROFILE_SPANS=1 \
    "$attempt/fg-runtime-benchmark" "$attempt/libs" "$model" "$rom" \
      "$sigmoid" "$unit" "$fraction" "$output/$mode/run-$ordinal.json" \
      "fg-runtime-u32-profile-$mode-$ordinal" a5 compact live \
      >"$output/$mode/run-$ordinal.stdout.log" 2>"$output/$mode/run-$ordinal.stderr.log"
}

run_one clz 1
run_one u32 1
run_one u32 2
run_one clz 2
run_one clz 3
run_one u32 3

python3 - "$repo" "$clz_attempt" "$u32_attempt" "$output" <<'PY'
import hashlib,json,pathlib,statistics,sys
repo,clz_attempt,u32_attempt,output=map(pathlib.Path,sys.argv[1:])
expected=json.loads((repo/'evidence/specs/fg-runtime-v1/live-alias/verification.json').read_text())['output_sha256']
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
labels=['status','L3','L4','L5-high','L5-low','L5-pack','L6','L7-texture-to-buffer',
        'L7-kernel','L7-buffer-to-texture','L8','L19','L20-L51','L52','L53','L54',
        'L55','L55-to-L84-blit','L56-L83','L84','L85','L86','L87','caller-output-copy']
domains=[0]*12+[1]*11+[2]
def mode_receipt(mode,attempt):
    paths=sorted((output/mode).glob('run-*.json'));runs=[json.loads(path.read_text()) for path in paths]
    if len(paths)!=3 or len({run['process_id'] for run in runs})!=3:
        raise SystemExit(mode+': fresh-process coverage')
    cpu=[[] for _ in labels];gpu=[[] for _ in labels];evaluate=[];cpu_sum=[];gpu_sum=[];unspanned=[]
    for ordinal,run in enumerate(runs,1):
        if (run.get('run_id')!=f'fg-runtime-u32-profile-{mode}-{ordinal}' or
            run.get('ex2_mode')!='compact' or run.get('auxiliary_mode')!='live' or
            run.get('poison_byte')!=165 or run.get('profiling_enabled') is not True or
            run.get('counter_attachment') is not False or run.get('result')!='pass' or
            run.get('pre_output_sha256')!=expected or run.get('measured_output_sha256')!=expected or
            run.get('post_output_sha256')!=expected or run.get('output_hashes_comparable') is not True or
            run.get('caller_owned_input_mismatches')!=0 or
            run.get('expected_or_archived_intermediate_ingress') is not False or
            run.get('address_fields_recorded') is not False or
            run.get('measured_metal_resource_allocations')!=[0]*15):
            raise SystemExit(mode+': raw run contract')
        for wall,spans in zip(run['measured_evaluate_wall_ms'],run['measured_command_spans']):
            if (len(spans)!=24 or [span['ordinal'] for span in spans]!=list(range(24)) or
                [span['domain'] for span in spans]!=domains):raise SystemExit(mode+': span identity')
            fc=sum(span['cpu_commit_wait_ms'] for span in spans);fg=sum(span['gpu_ms'] for span in spans)
            evaluate.append(wall);cpu_sum.append(fc);gpu_sum.append(fg);unspanned.append(wall-fc)
            for index,span in enumerate(spans):
                cpu[index].append(span['cpu_commit_wait_ms']);gpu[index].append(span['gpu_ms'])
    if json.loads((attempt/'backend-receipt.json').read_text()).get('hmma_mode')!=mode:
        raise SystemExit(mode+': HMMA mode')
    return {'hmma_mode':mode,'process_ids':[run['process_id'] for run in runs],
      'raw_receipt_sha256':[sha(path) for path in paths],
      'evaluate_wall_p50_ms':statistics.median(evaluate),
      'command_commit_wait_sum_p50_ms':statistics.median(cpu_sum),
      'gpu_span_sum_p50_ms':statistics.median(gpu_sum),
      'unspanned_host_work_p50_ms':statistics.median(unspanned),
      'spans':[{'ordinal':i,'label':label,'domain':domains[i],
                'cpu_commit_wait_p50_ms':statistics.median(cpu[i]),
                'gpu_p50_ms':statistics.median(gpu[i]),'gpu_max_ms':max(gpu[i])}
               for i,label in enumerate(labels)],
      'binary_sha256':sha(attempt/'fg-runtime-benchmark'),
      'runtime_tail_sha256':sha(attempt/'build/runtime_tail.mm')}
clz=mode_receipt('clz',clz_attempt);u32=mode_receipt('u32',u32_attempt)
if len(set(clz['process_ids']+u32['process_ids']))!=6:raise SystemExit('cross-mode freshness')
receipt={'schema':'neuralstudio.fg.runtime-u32-boundary-profile.v1',
  'profile':'640x360-compact-live-early-aliases','diagnostic_instrumentation':True,
  'production_benchmark':False,
  'timing_class':'synchronous-evaluate-wall-and-command-buffer-spans-not-input-latency',
  'execution_order':['clz-1','u32-1','u32-2','clz-2','clz-3','u32-3'],
  'independent_processes_per_mode':3,'measured_frames_per_process':15,
  'command_spans_per_frame':24,'selected_wait_boundaries':23,
  'untimed_correctness_before_and_after':True,'output_hashes_comparable':True,
  'accepted_output_sha256':expected,
  'accepted_checkpoint_root_sha256':'2b13a70a5d44659a6912c872a01adfd0e9eceaa02b668743cef82a0ea088a156',
  'clz':clz,'u32':u32,
  'u32_vs_clz_evaluate_p50_speedup':clz['evaluate_wall_p50_ms']/u32['evaluate_wall_p50_ms'],
  'u32_vs_clz_gpu_span_p50_speedup':clz['gpu_span_sum_p50_ms']/u32['gpu_span_sum_p50_ms'],
  'generic_l20_l51_gpu_speedup':clz['spans'][12]['gpu_p50_ms']/u32['spans'][12]['gpu_p50_ms'],
  'custom_l56_l83_gpu_speedup':clz['spans'][18]['gpu_p50_ms']/u32['spans'][18]['gpu_p50_ms'],
  'expected_or_archived_intermediate_ingress':False,'address_fields_recorded':False,'result':'pass'}
normalized={key:value for key,value in receipt.items() if key not in ('clz','u32')}
receipt['normalized_root_sha256']=hashlib.sha256(
    json.dumps(normalized,sort_keys=True,separators=(',',':')).encode()).hexdigest()
(output/'profile.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
for key in ('u32_vs_clz_evaluate_p50_speedup','u32_vs_clz_gpu_span_p50_speedup',
            'generic_l20_l51_gpu_speedup','custom_l56_l83_gpu_speedup'):
    print(key+'='+str(receipt[key]))
print('profile_sha256='+sha(output/'profile.json'));print('result=pass')
PY

echo "profile_output=$output"
