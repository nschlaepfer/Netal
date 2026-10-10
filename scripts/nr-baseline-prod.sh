#!/bin/zsh
# Matched PRODUCTION baseline (prepared 2026-09-07; DO NOT RUN before a separate review and an explicit quiet-window ACK).
# ONE reservation for the whole set, balanced arm order p1 v11->v17, p2 v17->v11, p3 v11->v17, p4 v17->v11.
# Codex 02:53 amendments: a REAL second `top` sample (the first sample of `top -l 1` is since-boot and meaningless),
# sample identity and any error/unknown value retained verbatim in confounds.log; the outer storage admission uses the
# runner's own rule (free - 3 GiB pilot reserve >= 12 GiB floor); the output root reaches the nested shell through argv,
# never string interpolation; python3.13 is named explicitly. Full stdout/stderr of every process is kept.
# Usage: nr-baseline-prod.sh OUT_ROOT
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"; cd "$repo"
out="${1:?output root}"
PY="${NR_PY:-/opt/homebrew/bin/python3.13}"
[[ -x "$PY" ]] || { echo "python3.13 not found at $PY (override NR_PY)" >&2; exit 2; }
[[ ! -e "$out" ]] || { echo "preserve existing $out" >&2; exit 2; }
mkdir -p "$out"
for i in 1 2 3 4; do for arm in v11 v17; do
  "$PY" scripts/nr-run-recipe.py preflight "spec/nr/recipes/baseline-prod-v1/baseline-prod-$arm-p$i.json" >/dev/null || { echo "preflight failed $arm p$i" >&2; exit 3; }
done; done
echo "all eight production recipes preflight ok"
"$PY" scripts/nr-gpu-lock.py run --owner Fable --note "QUIET WINDOW matched PRODUCTION baseline: 8 fresh processes interleaved v11/v17, telemetry on, production body mode; outputs retained; runner pilot reserve 3 GiB, floor 12 GiB" -- \
  zsh -c '
  set -euo pipefail
  out="$1"; PY="$2"
  for pair in "1 v11 v17" "2 v17 v11" "3 v11 v17" "4 v17 v11"; do
    set -- ${=pair}; i=$1
    for arm in $2 $3; do
      {
        echo "=== $(date -u +%FT%TZ) before $arm p$i"
        echo "-- pmset -g therm"; pmset -g therm 2>&1 | tr -s " " | head -6
        echo "-- vm.loadavg";     sysctl -n vm.loadavg 2>&1
        echo "-- memory_pressure"; memory_pressure -Q 2>&1 | head -2
        # a real measured sample: top -l 2 -s 1 discards nothing - the FIRST sample is since-boot, the SECOND is a 1 s
        # measurement; both are kept here with their identity, and any error text or "unknown" value is retained verbatim
        echo "-- top -l 2 -s 1 (sample 1 = since boot, sample 2 = measured over 1 s; both retained)"
        top -l 2 -s 1 -n 6 -stats pid,cpu,command 2>&1 | awk "/^Processes:/{s++} {print \"[sample \" (s==0?1:s) \"] \" \$0}" | tail -30
      } >> "$out/confounds.log" 2>&1
      free=$(df -g . | awk "NR==2{print \$4}")
      echo "$(date -u +%FT%TZ) admission $arm p$i: free ${free} GiB, runner pilot reserve 3 GiB, floor 12 GiB" | tee -a "$out/admission.log"
      "$PY" -c "import sys; sys.exit(0 if $free - 3.0 >= 12 else 1)" || { echo "STORAGE ADMISSION REFUSED before $arm p$i (free ${free} GiB)" | tee -a "$out/admission.log"; exit 5; }
      "$PY" scripts/nr-run-recipe.py run "spec/nr/recipes/baseline-prod-v1/baseline-prod-$arm-p$i.json" --out "$out/$arm-p$i" > "$out/$arm-p$i.log" 2>&1 || {
        echo "PROCESS $arm p$i DID NOT PASS; stopping, evidence preserved"; tail -3 "$out/$arm-p$i.log"; exit 1; }
      tail -1 "$out/$arm-p$i.log"
    done
  done
  echo "matched production baseline: all eight processes passed"' nr-baseline-prod "$out" "$PY"
