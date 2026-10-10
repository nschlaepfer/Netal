#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
reference_attempt="${1:-$repo/build/fg_runtime/macos/reference-v2-checkpoints}"
compact_attempt="${2:-$repo/build/fg_runtime/macos/compact-reference-v2}"
output="${3:-$repo/build/fg_runtime/benchmark/attempt-$(date -u +%Y%m%dT%H%M%SZ)}"
model="${4:-$repo/build/fg_default_readback/eval0-default-all-a5-a/runtime-default-buffers-post-evaluate-0}"
rom="${5:-$repo/build/fg_connected_tail/macos-l56-l87/attempt-exact-v1/sm120-rcp-correction-rom.bin}"
sigmoid="${6:-$repo/build/fg_postprocess_l84/rtx/oracle-006/sigmoid-process-a/sm120-half-logit-sigmoid-lut.bin}"
reference_unit="${7:-$repo/build/fg_math_ex2/unit-v1/unit-minus1-to-zero-u32.bin}"
reference_fraction="${8:-$repo/build/fg_math_ex2/rtx-v3/fraction-minus2-to-minus1-u32.bin}"
compact_unit="${9:-$repo/build/fg_math_ex2/compact/attempt-v4/packed-a/unit-corrections.bin}"
compact_fraction="${10:-$repo/build/fg_math_ex2/compact/attempt-v4/packed-a/fraction-corrections.bin}"
if [[ -e "$output" ]]; then
  echo "Benchmark output exists; preserve it: $output" >&2
  exit 2
fi
mkdir -p "$output/build" "$output/reference" "$output/compact"

xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc -ffp-contract=off \
  -Wall -Wextra -Werror -Wno-unused-function -Wno-deprecated-declarations \
  -I "$repo/src/fg_runtime" -I "$repo/src/fg_production_contract_l3_l87" \
  -I "$repo/src/fg_connected_l20_l87" \
  "$repo/src/fg_runtime/fg_runtime.mm" \
  "$repo/src/fg_runtime/span_recorder.mm" \
  "$repo/src/fg_runtime/tail_output_bridge.mm" \
  "$repo/src/fg_runtime/benchmark_host.mm" \
  "$repo/src/fg_production_contract_l3_l87/macos_live_handoff.mm" \
  "$reference_attempt/build/runtime_tail.mm" \
  -framework Foundation -framework Metal -o "$output/build/fg-runtime-benchmark"

run_one() {
  local mode="$1" ordinal="$2" attempt unit fraction
  if [[ "$mode" == reference ]]; then
    attempt="$reference_attempt"; unit="$reference_unit"; fraction="$reference_fraction"
  else
    attempt="$compact_attempt"; unit="$compact_unit"; fraction="$compact_fraction"
  fi
  local receipt="$output/$mode/run-$ordinal.json"
  "$output/build/fg-runtime-benchmark" "$attempt/libs" "$model" "$rom" \
    "$sigmoid" "$unit" "$fraction" "$receipt" \
    "fg-runtime-benchmark-$mode-$ordinal" a5 "$mode" internal \
    >"$output/$mode/run-$ordinal.stdout.log" \
    2>"$output/$mode/run-$ordinal.stderr.log"
}

run_one reference 1
run_one compact 1
run_one compact 2
run_one reference 2
run_one reference 3
run_one compact 3

python3 - "$repo" "$output" "$reference_attempt" "$compact_attempt" \
  "$reference_unit" "$reference_fraction" "$compact_unit" \
  "$compact_fraction" <<'PY'
import hashlib, json, math, pathlib, statistics, sys

(repo, output, reference_attempt, compact_attempt, reference_unit,
 reference_fraction, compact_unit, compact_fraction) = map(pathlib.Path,
                                                             sys.argv[1:])
expected = json.loads((repo/'evidence/specs/fg-production-contract-l3-l87-v1/verification.json').read_text())['final_output_sha256']
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()

def percentile(values, fraction):
    ordered=sorted(values)
    position=(len(ordered)-1)*fraction
    lower=math.floor(position); upper=math.ceil(position)
    if lower==upper: return ordered[lower]
    return ordered[lower]+(ordered[upper]-ordered[lower])*(position-lower)

def load_mode(mode):
    paths=sorted((output/mode).glob('run-*.json'))
    if len(paths)!=3: raise SystemExit(f'{mode}: independent run count')
    runs=[json.loads(path.read_text()) for path in paths]
    if len({run['process_id'] for run in runs})!=3:
        raise SystemExit(f'{mode}: fresh process IDs')
    for run in runs:
        if run['result']!='pass' or run['ex2_mode']!=mode:
            raise SystemExit(f'{mode}: failed run')
        if not run['output_hashes_comparable']:
            raise SystemExit(f'{mode}: timing perturbed output')
        if run['pre_output_sha256']!=expected or run['measured_output_sha256']!=expected or run['post_output_sha256']!=expected:
            raise SystemExit(f'{mode}: accepted output mismatch')
        if any(run['measured_metal_resource_allocations']):
            raise SystemExit(f'{mode}: measured resource allocation')
    per_run=[statistics.median(run['measured_evaluate_wall_ms']) for run in runs]
    all_frames=[value for run in runs for value in run['measured_evaluate_wall_ms']]
    mean=statistics.mean(all_frames)
    std=statistics.pstdev(all_frames)
    return {
        'raw_receipt_sha256':[sha(path) for path in paths],
        'process_ids':[run['process_id'] for run in runs],
        'thermal_states_after':[run['thermal_state_after'] for run in runs],
        'session_create_wall_ms':[run['session_create_wall_ms'] for run in runs],
        'warmup_sequence_wall_ms':[run['warmup_sequence_wall_ms'] for run in runs],
        'per_run_median_evaluate_wall_ms':per_run,
        'evaluate_wall_ms':all_frames,
        'p50_evaluate_wall_ms':percentile(all_frames,0.50),
        'p95_evaluate_wall_ms':percentile(all_frames,0.95),
        'p99_evaluate_wall_ms':percentile(all_frames,0.99),
        'max_evaluate_wall_ms':max(all_frames),
        'mean_evaluate_wall_ms':mean,
        'stddev_evaluate_wall_ms':std,
        'coefficient_of_variation':std/mean,
    }

reference=load_mode('reference')
compact=load_mode('compact')
binary=output/'build/fg-runtime-benchmark'
configuration_common={
    'runtime_source_sha256':sha(repo/'src/fg_runtime/fg_runtime.mm'),
    'benchmark_source_sha256':sha(repo/'src/fg_runtime/benchmark_host.mm'),
    'tail_source_sha256':sha(reference_attempt/'build/runtime_tail.mm'),
    'binary_sha256':sha(binary),
    'model_directory':str(pathlib.Path('build/fg_default_readback/eval0-default-all-a5-a/runtime-default-buffers-post-evaluate-0')),
    'profile':'640x360-accepted-internal-small-flow',
    'measured_generated_frames_per_run':15,
    'independent_run_count':3,
}
def config_root(mode,unit,fraction,attempt):
    value=dict(configuration_common)
    value.update({'ex2_mode':mode,'unit_sha256':sha(unit),
                  'fraction_sha256':sha(fraction),
                  'metallib_l52_sha256':sha(attempt/'libs/mid_l52.metallib'),
                  'metallib_l54_sha256':sha(attempt/'libs/mid_l54.metallib')})
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest(),value
reference_root,reference_config=config_root('reference',reference_unit,reference_fraction,reference_attempt)
compact_root,compact_config=config_root('compact',compact_unit,compact_fraction,compact_attempt)
receipt={
    'schema':'neuralstudio.fg.runtime-benchmark-comparison.v1',
    'diagnostic_component_only':True,
    'production_benchmark':False,
    'timing_class':'synchronous-evaluate-wall-not-input-latency',
    'counter_attachment':False,
    'execution_order':['reference-1','compact-1','compact-2',
                       'reference-2','reference-3','compact-3'],
    'untimed_correctness_before_and_after':True,
    'output_hashes_comparable':True,
    'accepted_output_sha256':expected,
    'reference_configuration_root_sha256':reference_root,
    'compact_configuration_root_sha256':compact_root,
    'reference_configuration':reference_config,
    'compact_configuration':compact_config,
    'reference':reference,
    'compact':compact,
    'compact_vs_reference_p50_speedup':reference['p50_evaluate_wall_ms']/compact['p50_evaluate_wall_ms'],
    'reference_ex2_bytes':reference_unit.stat().st_size+reference_fraction.stat().st_size,
    'compact_ex2_bytes':compact_unit.stat().st_size+compact_fraction.stat().st_size,
    'ex2_saved_bytes':(reference_unit.stat().st_size+reference_fraction.stat().st_size)-(compact_unit.stat().st_size+compact_fraction.stat().st_size),
    'result':'pass',
}
normalized={key:value for key,value in receipt.items() if key not in ('reference','compact')}
receipt['normalized_root_sha256']=hashlib.sha256(json.dumps(normalized,sort_keys=True,separators=(',',':')).encode()).hexdigest()
(output/'comparison.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print('reference_p50_evaluate_wall_ms='+str(reference['p50_evaluate_wall_ms']))
print('compact_p50_evaluate_wall_ms='+str(compact['p50_evaluate_wall_ms']))
print('compact_vs_reference_p50_speedup='+str(receipt['compact_vs_reference_p50_speedup']))
print('comparison_sha256='+sha(output/'comparison.json'))
print('result=pass')
PY

echo "benchmark_output=$output"
