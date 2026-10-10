#!/bin/zsh
# Matched checked baseline-0: ONE reservation (the quiet window), eight fresh processes interleaved with BALANCED arm order (Codex 01:38): p1 v11->v17, p2 v17->v11, p3 v11->v17, p4 v17->v11;
# before every process the thermal pressure, load average, memory pressure and top CPU consumers are logged (confounds.log),
# preflight of all eight recipes BEFORE the lock, storage admission per process (free - 1.0 GiB estimated peak >= 12 GiB; v11 checked writes pool_post.bin
# 3 x 212 MB, v17 writes the same on desc-v30t), fail-stop on the first process that does not pass. Usage: nr-b0-interleaved.sh OUT_ROOT
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"; cd "$repo"; out="${1:?output root}"; [[ ! -e "$out" ]] || { echo "preserve existing $out" >&2; exit 2; }; mkdir -p "$out"
for i in 1 2 3 4; do for arm in v11 v17; do python3 scripts/nr-run-recipe.py preflight "spec/nr/recipes/b0-checked/b0-checked-$arm-p$i.json" >/dev/null || { echo "preflight failed $arm p$i" >&2; exit 3; }; done; done; echo "all eight recipes preflight ok"
python3 scripts/nr-gpu-lock.py run --owner Fable --note "QUIET WINDOW matched checked baseline-0: 8 fresh processes interleaved v11/v17 (checked body, zero-v3 telemetry on); outputs retained; est peak 1.0 GiB per process" -- zsh -c '
  set -euo pipefail; out="'"$out"'"
  for pair in "1 v11 v17" "2 v17 v11" "3 v11 v17" "4 v17 v11"; do set -- ${=pair}; i=$1; for arm in $2 $3; do
    { echo "=== $(date -u +%FT%TZ) before $arm p$i"; pmset -g therm 2>/dev/null | tr -s " " | head -6; sysctl -n vm.loadavg 2>/dev/null; memory_pressure -Q 2>/dev/null | head -2; top -l 1 -n 6 -stats pid,cpu,command 2>/dev/null | tail -8; } >> "$out/confounds.log" 2>&1
    free=$(df -g . | awk "NR==2{print \$4}"); echo "$(date -u +%FT%TZ) admission $arm p$i: free ${free} GiB, est peak 1.0 GiB, floor 12" | tee -a "$out/admission.log"
    python3 -c "import sys; sys.exit(0 if $free - 1.0 >= 12 else 1)" || { echo "STORAGE ADMISSION REFUSED before $arm p$i" | tee -a "$out/admission.log"; exit 5; }
    python3 scripts/nr-run-recipe.py run "spec/nr/recipes/b0-checked/b0-checked-$arm-p$i.json" --out "$out/$arm-p$i" > "$out/$arm-p$i.log" 2>&1 || { echo "PROCESS $arm p$i DID NOT PASS; stopping, evidence preserved"; tail -3 "$out/$arm-p$i.log"; exit 1; }
    tail -1 "$out/$arm-p$i.log"
  done; done; echo "baseline-0 matched checked: all eight processes passed"'
