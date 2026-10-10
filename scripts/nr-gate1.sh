#!/bin/zsh
# Gate 1 of the resident contract: run the eight pinned recipes (NR_GATE1_VER, default v27t), ONE reservation per recipe (the lock is released between recipes so the
# other agent can interleave), preflight BEFORE each lock, storage admission (measured free minus the estimated run peak >= 12 GiB) journaled
# per run, fail-stop on the first recipe that does not pass. Usage: nr-gate1-v26t.sh OUT_ROOT [recipe-name ...]
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"; cd "$repo"; out="${1:?output root}"; shift || true
mkdir -p "$out"; names=("$@"); [[ ${#names[@]} -gt 0 ]] || names=(black3 checker3 dense-case3-reset images3 midreset8 uniform3 spatial3 combined3-v20t-v1)
for n in $names; do
  ver="${NR_GATE1_VER:-v27t}"; rec="spec/nr/recipes/$ver/$n-$ver.json"; [[ -f "$rec" ]] || { echo "missing recipe $rec" >&2; exit 2; }
  python3 scripts/nr-run-recipe.py preflight "$rec" || { echo "preflight failed for $n" >&2; exit 3; }
  free=$(df -g "$repo" | awk 'NR==2{print $4}'); frames=$(python3 -c "import json;r=json.load(open('$rec'));print(len(r['frames'])*len(r['replicas']))"); peak_gib=$(python3 -c "print(round($frames*0.045+0.2,2))")   # per frame: final 16.6 MB + tex_out 155/156 33 MB + view/receipts
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) storage admission $n: free ${free} GiB, estimated peak ${peak_gib} GiB (${frames} frames), floor 12 GiB" | tee -a "$out/admission.log"
  python3 -c "import sys; sys.exit(0 if $free - $peak_gib >= 12 else 1)" || { echo "STORAGE ADMISSION REFUSED for $n" | tee -a "$out/admission.log"; exit 5; }
  [[ ! -e "$out/$n" ]] || { echo "preserve existing $out/$n" >&2; exit 2; }
  python3 scripts/nr-gpu-lock.py run --owner Fable --note "gate 1 (resident v12) recipe $n-$ver: four fresh processes, receipts must equal accepted; storage free ${free} GiB peak ${peak_gib} GiB" -- python3 scripts/nr-run-recipe.py run "$rec" --out "$out/$n" > "$out/$n.log" 2>&1 || { echo "RECIPE $n DID NOT PASS (see $out/$n.log); stopping, evidence preserved"; tail -5 "$out/$n.log"; exit 1; }
  tail -2 "$out/$n.log"
done
echo "gate 1 recipes all passed: $names"
