#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
attempt="${1:-$repo/build/fg_runtime/macos/compact-profile-baseline-v1}"
output="${2:-$repo/build/fg_runtime/profile/attempt-$(date -u +%Y%m%dT%H%M%SZ)}"
model="${3:-$repo/build/fg_default_readback/eval0-default-all-a5-a/runtime-default-buffers-post-evaluate-0}"
rom="${4:-$repo/build/fg_connected_tail/macos-l56-l87/attempt-exact-v1/sm120-rcp-correction-rom.bin}"
sigmoid="${5:-$repo/build/fg_postprocess_l84/rtx/oracle-006/sigmoid-process-a/sm120-half-logit-sigmoid-lut.bin}"
unit="${6:-$repo/build/fg_math_ex2/compact/attempt-v4/packed-a/unit-corrections.bin}"
fraction="${7:-$repo/build/fg_math_ex2/compact/attempt-v4/packed-a/fraction-corrections.bin}"
if [[ -e "$output" ]]; then
  echo "Profile output exists; preserve it: $output" >&2
  exit 2
fi
mkdir -p "$output"

for ordinal in 1 2 3; do
  FG_RUNTIME_PROFILE_SPANS=1 \
    "$attempt/fg-runtime-benchmark" "$attempt/libs" "$model" "$rom" \
      "$sigmoid" "$unit" "$fraction" "$output/run-$ordinal.json" \
      "fg-runtime-profile-$ordinal" a5 compact internal \
      >"$output/run-$ordinal.stdout.log" \
      2>"$output/run-$ordinal.stderr.log"
done

python3 - "$repo" "$attempt" "$output" <<'PY'
import hashlib,json,math,pathlib,statistics,sys
repo,attempt,output=map(pathlib.Path,sys.argv[1:])
expected=json.loads((repo/'evidence/specs/fg-production-contract-l3-l87-v1/verification.json').read_text())['final_output_sha256']
paths=sorted(output.glob('run-*.json'))
if len(paths)!=3: raise SystemExit('profile run count')
runs=[json.loads(path.read_text()) for path in paths]
if len({run['process_id'] for run in runs})!=3: raise SystemExit('profile fresh process IDs')
labels=['status','L3','L4','L5-high','L5-low','L5-pack','L6',
        'L7-texture-to-buffer','L7-kernel','L7-buffer-to-texture','L8','L19',
        'L20-L51','L52','L53','L54','L55','L55-to-L84-blit','L56-L83',
        'L84','L85','L86','L87','caller-output-copy']
expected_domains=[0]*12+[1]*11+[2]
cpu_by_span=[[] for _ in labels];gpu_by_span=[[] for _ in labels]
evaluate=[];cpu_sum=[];gpu_sum=[];unattributed=[]
for run in runs:
    if run['result']!='pass' or not run['profiling_enabled'] or not run['output_hashes_comparable']:
        raise SystemExit('profile run failed')
    if run['pre_output_sha256']!=expected or run['measured_output_sha256']!=expected or run['post_output_sha256']!=expected:
        raise SystemExit('profile output drift')
    if any(run['measured_metal_resource_allocations']): raise SystemExit('profile allocation drift')
    if len(run['measured_command_spans'])!=15: raise SystemExit('profile frame span coverage')
    for wall,spans in zip(run['measured_evaluate_wall_ms'],run['measured_command_spans']):
        if len(spans)!=24 or [span['ordinal'] for span in spans]!=list(range(24)) or [span['domain'] for span in spans]!=expected_domains:
            raise SystemExit('profile span identity drift')
        frame_cpu=sum(span['cpu_commit_wait_ms'] for span in spans)
        frame_gpu=sum(span['gpu_ms'] for span in spans)
        evaluate.append(wall);cpu_sum.append(frame_cpu);gpu_sum.append(frame_gpu)
        unattributed.append(wall-frame_cpu)
        for index,span in enumerate(spans):
            cpu_by_span[index].append(span['cpu_commit_wait_ms'])
            gpu_by_span[index].append(span['gpu_ms'])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
spans=[]
for index,label in enumerate(labels):
    spans.append({'ordinal':index,'label':label,'domain':expected_domains[index],
                  'cpu_commit_wait_p50_ms':statistics.median(cpu_by_span[index]),
                  'gpu_p50_ms':statistics.median(gpu_by_span[index]),
                  'cpu_commit_wait_max_ms':max(cpu_by_span[index]),
                  'gpu_max_ms':max(gpu_by_span[index])})
receipt={
 'schema':'neuralstudio.fg.runtime-boundary-profile.v1',
 'profile':'640x360-compact-accepted-internal-small-flow',
 'diagnostic_instrumentation':True,
 'production_benchmark':False,
 'timing_class':'synchronous-evaluate-wall-and-command-buffer-spans-not-input-latency',
 'untimed_correctness_before_and_after':True,
 'output_hashes_comparable':True,
 'independent_process_count':3,
 'measured_frame_count':45,
 'command_spans_per_frame':24,
 'static_selected_wait_boundaries':23,
 'additional_caller_output_copy_boundary':1,
 'evaluate_wall_p50_ms':statistics.median(evaluate),
 'command_commit_wait_sum_p50_ms':statistics.median(cpu_sum),
 'gpu_span_sum_p50_ms':statistics.median(gpu_sum),
 'unattributed_host_work_p50_ms':statistics.median(unattributed),
 'spans':spans,
 'raw_receipt_sha256':[sha(path) for path in paths],
 'runtime_binary_sha256':sha(attempt/'fg-runtime-benchmark'),
 'runtime_tail_source_sha256':sha(attempt/'build/runtime_tail.mm'),
 'result':'pass'}
normalized={key:value for key,value in receipt.items() if key not in ('raw_receipt_sha256',)}
receipt['normalized_root_sha256']=hashlib.sha256(json.dumps(normalized,sort_keys=True,separators=(',',':')).encode()).hexdigest()
(output/'profile.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print('evaluate_wall_p50_ms='+str(receipt['evaluate_wall_p50_ms']))
print('command_commit_wait_sum_p50_ms='+str(receipt['command_commit_wait_sum_p50_ms']))
print('gpu_span_sum_p50_ms='+str(receipt['gpu_span_sum_p50_ms']))
print('unattributed_host_work_p50_ms='+str(receipt['unattributed_host_work_p50_ms']))
print('profile_sha256='+sha(output/'profile.json'))
print('result=pass')
PY

echo "profile_output=$output"
