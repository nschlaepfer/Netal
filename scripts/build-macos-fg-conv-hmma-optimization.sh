#!/bin/bash
set -euo pipefail

repository="$(cd "$(dirname "$0")/.." && pwd)"
attempt="${1:-$repository/build/fg_conv_hmma_optimization/attempt-$(date -u +%Y%m%dT%H%M%SZ)}"
model="${2:-$repository/build/fg_default_readback/eval0-default-all-a5-a/runtime-default-buffers-post-evaluate-0}"
if [[ -e "$attempt" ]]; then
  echo "attempt exists; preserve it: $attempt" >&2
  exit 2
fi
mkdir -p "$attempt/build" "$attempt/clz" "$attempt/counterexamples"

compile_metal() {
  local source="$1" name="$2"
  xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math \
    -ffp-contract=off -I "$repository/src/fg_conv" \
    -I "$repository/src/fg_conv_simd_l20" -c "$repository/$source" \
    -o "$attempt/build/$name.air"
  xcrun -sdk macosx metallib "$attempt/build/$name.air" \
    -o "$attempt/build/$name.metallib"
}

compile_metal src/fg_conv/conv_3x3.metal scalar
compile_metal src/fg_conv_simd_l20/l20_exact_clz.metal clz
compile_metal src/fg_conv_simd_l20/l20_simd.metal simd
compile_metal src/fg_conv_simd_l20/l20_dual_order.metal dual
compile_metal src/fg_conv_simd_l20/l20_multi_order.metal multi

xcrun -sdk macosx clang++ -std=c++20 -O3 -fobjc-arc -Wall -Wextra \
  -Werror -Wno-unused-function -Wno-deprecated-declarations \
  "$repository/src/fg_conv/macos_3x3_probe.mm" \
  -framework Foundation -framework Metal -framework Security \
  -o "$attempt/build/probe"
xcrun -sdk macosx clang++ -std=c++20 -O3 -fobjc-arc -Wall -Wextra \
  -Werror -Wno-unused-function -Wno-deprecated-declarations \
  "$repository/src/fg_conv_simd_l20/benchmark.mm" \
  -framework Foundation -framework Metal -framework Security \
  -o "$attempt/build/benchmark"

for spec in l20:63 l24:67 l26:69 l28:71; do
  layer="${spec%%:*}"
  token="${spec##*:}"
  weights="$model/default-token-0000$token.bin"
  for poison in a5 5a; do
    for ordinal in a b; do
      run="$attempt/clz/$layer-$poison-$ordinal"
      "$attempt/build/probe" "$layer" "$attempt/build/clz.metallib" \
        "$weights" "$run" "$layer-$poison-$ordinal" "$poison" \
        >"$run.stdout.log" 2>"$run.stderr.log"
    done
  done
  "$attempt/build/benchmark" "$attempt/build/scalar.metallib" \
    "$attempt/build/clz.metallib" "$weights" \
    "$attempt/clz/$layer-benchmark.json" a5 fg_conv_3x3 "$layer" \
    >"$attempt/clz/$layer-benchmark.stdout.log" \
    2>"$attempt/clz/$layer-benchmark.stderr.log"
done

for candidate in simd dual multi; do
  function=fg_conv_3x3
  [[ "$candidate" == simd ]] && function=fg_conv_3x3_simd
  "$attempt/build/benchmark" "$attempt/build/scalar.metallib" \
    "$attempt/build/$candidate.metallib" \
    "$model/default-token-000063.bin" \
    "$attempt/counterexamples/$candidate-benchmark.json" a5 "$function" l20 \
    >"$attempt/counterexamples/$candidate-benchmark.stdout.log" \
    2>"$attempt/counterexamples/$candidate-benchmark.stderr.log"
done

PYTHONDONTWRITEBYTECODE=1 python3 - "$repository" "$attempt" <<'PY'
import hashlib,json,sys
from pathlib import Path

repository,attempt=map(Path,sys.argv[1:])
layers=('l20','l24','l26','l28')
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
expected={layer:json.loads((repository/f'evidence/specs/fg-conv-3x3-family-v1/layers/{layer}-verification.json').read_text())['output_sha256'] for layer in layers}
clz={}
process_ids=[]
for layer in layers:
    paths=sorted((attempt/'clz').glob(f'{layer}-*/run-receipt.json'))
    assert len(paths)==4
    runs=[json.loads(path.read_text()) for path in paths]
    assert len({run['process_id'] for run in runs})==4
    assert {run['poison_byte'] for run in runs}=={90,165}
    assert all(run['result']=='pass' for run in runs)
    assert {run['output_sha256'] for run in runs}=={expected[layer]}
    process_ids.extend(run['process_id'] for run in runs)
    benchmark=json.loads((attempt/'clz'/f'{layer}-benchmark.json').read_text())
    assert benchmark['correctness_result']=='pass'
    assert benchmark['scalar_candidate_mismatched_bytes']==0
    clz[layer]={
      'output_sha256':expected[layer],
      'fresh_process_count':4,
      'poison_bytes':[90,165],
      'compared_bytes':runs[0]['output_bytes']*4,
      'mismatched_bytes':0,
      'scalar_median_gpu_ms':benchmark['scalar_median_gpu_ms'],
      'candidate_median_gpu_ms':benchmark['candidate_median_gpu_ms'],
      'candidate_gpu_speedup':benchmark['candidate_gpu_speedup'],
    }
assert len(set(process_ids))==16
counterexamples={}
for name in ('simd','dual','multi'):
    benchmark=json.loads((attempt/'counterexamples'/f'{name}-benchmark.json').read_text())
    assert benchmark['result']=='diagnostic' and benchmark['correctness_result']=='fail'
    assert benchmark['scalar_candidate_mismatched_bytes']>0
    assert benchmark['immutable_input_mismatched_bytes']==0
    assert benchmark['canary_mismatched_bytes']==0
    counterexamples[name]={
      'output_sha256':benchmark['candidate_output_sha256'],
      'mismatched_bytes':benchmark['scalar_candidate_mismatched_bytes'],
      'scalar_median_gpu_ms':benchmark['scalar_median_gpu_ms'],
      'candidate_median_gpu_ms':benchmark['candidate_median_gpu_ms'],
      'candidate_gpu_speedup':benchmark['candidate_gpu_speedup'],
      'result':'rejected',
    }
sources={relative:sha(repository/relative) for relative in (
  'src/fg_conv_simd_l20/hmma_exact_clz.metalh',
  'src/fg_conv_simd_l20/l20_exact_clz.metal',
  'src/fg_conv_simd_l20/l20_simd.metal',
  'src/fg_conv_simd_l20/l20_dual_order.metal',
  'src/fg_conv_simd_l20/l20_multi_order.metal',
  'src/fg_conv_simd_l20/benchmark.mm',
  'scripts/build-macos-fg-conv-hmma-optimization.sh')}
result={
 'schema':'neuralstudio.fg.conv-hmma-optimization-mac.v1',
 'device':'Apple M3 Max',
 'accepted_parent_root_sha256':'972a6888b346a87423949b6cfa33c6dc91e3f93c303ebfc9e420a355c14550b3',
 'exact_clz_family':clz,
 'exact_clz_fresh_process_count':16,
 'exact_clz_two_poison':True,
 'exact_clz_total_mismatched_bytes':0,
 'counterexamples':counterexamples,
 'source_sha256':sources,
 'expected_output_bound_to_gpu':False,
 'address_fields_recorded':False,
 'claims':{
   'clz_leading_bit_substitution_exact_for_3x3_family':True,
   'raw_simd_exact':False,
   'dual_or_multi_order_detector_exact':False,
   'complete_runtime_optimized':False,
   'netal_integration':False},
 'result':'pass'}
result['receipt_root_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest()
(attempt/'accepted-receipt.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print('receipt_root_sha256='+result['receipt_root_sha256'])
print('result=pass')
PY

echo "attempt=$attempt"
