#!/usr/bin/env bash
# Complete-generated-frame A/B for NSFG_FUSED_UPSAMPLE with the benchmark host: alternating fresh processes
# (15 warm-up + 15 measured generated frames each, 640x360, compact EX2, live auxiliaries, deferred waits), baseline
# first on even pairs and fused first on odd pairs. Every process is gated on its pre/measured/post output hashes
# against the expected receipt, on zero measured allocations, and on the fused-path log marker matching its variant.
#   scripts/fg-fused-frame-ab.sh <attempt> <output dir> [pairs, default 6] [NSFG_FUSED_UPSAMPLE value, default 1]
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
attempt="$1"; output="$2"; pairs="${3:-6}"; launches="${4:-1}"
expected="${FG_RUNTIME_EXPECTED:-$repo/evidence/specs/fg-runtime-v1/live-zero-mask/verification.json}"
model="$repo/build/fg_default_readback/eval0-default-all-a5-a/runtime-default-buffers-post-evaluate-0"
rom="$repo/build/fg_connected_tail/macos-l56-l87/attempt-exact-v1/sm120-rcp-correction-rom.bin"
sigmoid="$repo/build/fg_postprocess_l84/rtx/oracle-006/sigmoid-process-a/sm120-half-logit-sigmoid-lut.bin"
unit="$repo/build/fg_math_ex2/compact/attempt-v4/packed-a/unit-corrections.bin"
fraction="$repo/build/fg_math_ex2/compact/attempt-v4/packed-a/fraction-corrections.bin"
if [[ -e "$output" ]]; then echo "output exists; preserve it: $output" >&2; exit 2; fi
mkdir -p "$output"
run_one() {
  local variant="$1" index="$2"; local tag="$variant-$index"
  local -a args=("$attempt/libs" "$model" "$rom" "$sigmoid" "$unit" "$fraction" "$output/run-$tag.json" "fg-fused-ab-$tag" a5 compact live)
  if [[ "$variant" == fused ]]; then
    env NSFG_FUSED_UPSAMPLE="$launches" "$attempt/fg-runtime-benchmark" "${args[@]}" >"$output/run-$tag.stdout.log" 2>"$output/run-$tag.stderr.log" || true
  else
    env -u NSFG_FUSED_UPSAMPLE "$attempt/fg-runtime-benchmark" "${args[@]}" >"$output/run-$tag.stdout.log" 2>"$output/run-$tag.stderr.log" || true
  fi
}
for ((i = 0; i < pairs; ++i)); do
  if (( i % 2 == 0 )); then run_one baseline "$i"; run_one fused "$i"; else run_one fused "$i"; run_one baseline "$i"; fi
done
python3 - "$output" "$expected" "$pairs" "$launches" <<'PY'
import json, math, pathlib, statistics, sys
out, expected_path, pairs, launches = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
expected = json.loads(expected_path.read_text())['output_sha256']
def pct(v, f):
    o = sorted(v); p = (len(o) - 1) * f; lo, hi = math.floor(p), math.ceil(p)
    return o[lo] if lo == hi else o[lo] + (o[hi] - o[lo]) * (p - lo)
variants, problems = {}, []
for variant in ('baseline', 'fused'):
    frames, per_process = [], []
    for i in range(pairs):
        path = out / f'run-{variant}-{i}.json'
        if not path.exists():
            problems.append(f'{variant} {i}: no receipt'); continue
        r = json.loads(path.read_text())
        marker = 'fused bilinear fetch on launch' in (out / f'run-{variant}-{i}.stderr.log').read_text()
        if marker != (variant == 'fused'): problems.append(f'{variant} {i}: fused marker {marker}')
        if r['result'] != 'pass' or r['profiling_enabled'] or not r['output_hashes_comparable']: problems.append(f'{variant} {i}: run contract')
        if r['pre_output_sha256'] != expected or r['measured_output_sha256'] != expected or r['post_output_sha256'] != expected:
            problems.append(f'{variant} {i}: output hashes differ from expected')
        if any(r['measured_metal_resource_allocations']): problems.append(f'{variant} {i}: measured allocations')
        v = r['measured_evaluate_wall_ms']; frames += v; per_process.append(statistics.median(v))
    variants[variant] = {'frames': len(frames), 'p50_ms': pct(frames, .5) if frames else None, 'p95_ms': pct(frames, .95) if frames else None,
                         'mean_ms': statistics.mean(frames) if frames else None, 'min_ms': min(frames) if frames else None,
                         'per_process_p50_ms': per_process, 'evaluate_wall_ms': frames}
ok = all(variants[k]['frames'] for k in variants)
delta = variants['fused']['p50_ms'] - variants['baseline']['p50_ms'] if ok else None
summary = {'schema': 'neuralstudio.fg.fused-upsample-frame-ab.v1', 'pairs': pairs, 'nsfg_fused_upsample': launches,
           'expected': str(expected_path), 'order': 'baseline first on even pairs, fused first on odd pairs',
           'variants': variants, 'delta_p50_ms': delta, 'problems': problems, 'result': 'pass' if not problems else 'FAIL'}
(out / 'ab.json').write_text(json.dumps(summary, indent=2) + '\n')
for k in ('baseline', 'fused'):
    r = variants[k]
    if r['frames']:
        print(f"{k:8s}: p50 {r['p50_ms']:.3f} ms  p95 {r['p95_ms']:.3f}  mean {r['mean_ms']:.3f}  min {r['min_ms']:.3f}  frames {r['frames']}  per-process p50 " + ' '.join(f'{x:.2f}' for x in r['per_process_p50_ms']))
print(f"delta p50 (fused - baseline): {delta:+.3f} ms" if delta is not None else 'delta: n/a')
print(f"result={summary['result']} problems={problems}")
sys.exit(0 if not problems else 3)
PY
