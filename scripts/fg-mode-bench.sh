#!/bin/zsh
# Time the FG runtime in a given mode (env NSFG_BLOCK_MODE / NSFG_PREFIX_MODE) with the benchmark host,
# one process, and print the p50 evaluate wall. Output hashes are not gated here: hardware modes are
# PSNR-gated by scripts/fg-mode-psnr.py against the exact replay instead.
#   scripts/fg-mode-bench.sh <attempt> <tag>
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"; attempt="$1"; tag="$2"
model="$repo/build/fg_default_readback/eval0-default-all-a5-a/runtime-default-buffers-post-evaluate-0"
out="$repo/build/fg_runtime/benchmark/mode-$tag"; mkdir -p "$out"
"$attempt/fg-runtime-benchmark" "$attempt/libs" "$model" \
  "$repo/build/fg_connected_tail/macos-l56-l87/attempt-exact-v1/sm120-rcp-correction-rom.bin" \
  "$repo/build/fg_postprocess_l84/rtx/oracle-006/sigmoid-process-a/sm120-half-logit-sigmoid-lut.bin" \
  "$repo/build/fg_math_ex2/compact/attempt-v4/packed-a/unit-corrections.bin" \
  "$repo/build/fg_math_ex2/compact/attempt-v4/packed-a/fraction-corrections.bin" \
  "$out/run.json" "mode-$tag" a5 compact live > "$out/stdout.log" 2>&1 || true
python3 - "$out/run.json" "$tag" <<'PY'
import json, statistics, sys
j = json.load(open(sys.argv[1])); v = sorted(j["measured_evaluate_wall_ms"])
print(f"{sys.argv[2]}: p50 {statistics.median(v):.2f} ms  min {v[0]:.2f}  p95 {v[int(len(v)*0.95)-1]:.2f}  frames {len(v)}  result {j['result']}")
PY
