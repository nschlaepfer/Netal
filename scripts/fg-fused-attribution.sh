#!/usr/bin/env bash
# Per-dispatch GPU attribution of NSFG_FUSED_UPSAMPLE inside the runtime (diagnostic; the frame measure is
# fg-fused-frame-ab.sh). NSFG_BLOCK_TIMING=1 puts stage-boundary timestamps on every block dispatch and needs
# NSFG_SYNC_WAITS=1. Baseline and fused processes alternate; per-kernel medians are taken over every reported frame
# after the first three of each process. Output hashes are still checked against the expected receipt.
#   scripts/fg-fused-attribution.sh <attempt> <output dir> [processes per variant, default 2] [NSFG_FUSED_UPSAMPLE value, default 1]
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
attempt="$1"; output="$2"; count="${3:-2}"; launches="${4:-1}"
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
  local -a args=("$attempt/libs" "$model" "$rom" "$sigmoid" "$unit" "$fraction" "$output/attr-$tag.json" "fg-fused-attr-$tag" a5 compact live)
  if [[ "$variant" == fused ]]; then
    env NSFG_BLOCK_TIMING=1 NSFG_SYNC_WAITS=1 NSFG_FUSED_UPSAMPLE="$launches" "$attempt/fg-runtime-benchmark" "${args[@]}" >"$output/attr-$tag.stdout.log" 2>"$output/attr-$tag.stderr.log" || true
  else
    env -u NSFG_FUSED_UPSAMPLE NSFG_BLOCK_TIMING=1 NSFG_SYNC_WAITS=1 "$attempt/fg-runtime-benchmark" "${args[@]}" >"$output/attr-$tag.stdout.log" 2>"$output/attr-$tag.stderr.log" || true
  fi
}
for ((i = 0; i < count; ++i)); do
  if (( i % 2 == 0 )); then run_one baseline "$i"; run_one fused "$i"; else run_one fused "$i"; run_one baseline "$i"; fi
done
python3 - "$output" "$count" "$expected" <<'PY'
import json, pathlib, re, statistics, sys
out, count = pathlib.Path(sys.argv[1]), int(sys.argv[2])
expected = json.loads(pathlib.Path(sys.argv[3]).read_text())['output_sha256']
medians, frames, problems = {}, {}, []
for variant in ('baseline', 'fused'):
    per = {}
    n = 0
    for i in range(count):
        tag = f'{variant}-{i}'
        path = out / f'attr-{tag}.json'
        if not path.exists():
            problems.append(f'{tag}: no receipt'); continue
        r = json.loads(path.read_text())
        if r['result'] != 'pass' or r['measured_output_sha256'] != expected:
            problems.append(f'{tag}: result or output hashes')
        lines = [l for l in (out / f'attr-{tag}.stderr.log').read_text().splitlines() if l.startswith('block_timing ')]
        for line in lines[3:]:
            n += 1
            for m in re.finditer(r'(\S+)=([0-9.]+)', line):
                per.setdefault(m.group(1), []).append(float(m.group(2)))
    medians[variant] = {k: statistics.median(v) for k, v in per.items()}
    frames[variant] = n
keys = sorted({k for v in medians.values() for k in v if 'bot' in k or k in ('total_ms', 'dispatches')},
              key=lambda k: (k not in ('total_ms', 'dispatches'), k))
print(f"frames with block timing: baseline {frames['baseline']}, fused {frames['fused']}")
print(f"{'kernel (median ms per frame)':58s} {'baseline':>9s} {'fused':>9s}")
for k in keys:
    b, f = medians['baseline'].get(k), medians['fused'].get(k)
    print(f"{k:58s} {'' if b is None else f'{b:9.3f}':>9s} {'' if f is None else f'{f:9.3f}':>9s}")
def bottom(v):
    return sum(x for k, x in medians[v].items() if 'bot' in k)
print(f"{'sum of bottom-family dispatches':58s} {bottom('baseline'):9.3f} {bottom('fused'):9.3f}")
summary = {'schema': 'neuralstudio.fg.fused-upsample-attribution.v1', 'processes_per_variant': count, 'frames': frames,
           'median_ms': medians, 'problems': problems, 'result': 'pass' if not problems else 'FAIL'}
(out / 'attribution.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
print(f"result={summary['result']} problems={problems}")
sys.exit(0 if not problems else 3)
PY
