#!/bin/zsh
# Run ONE pinned recipe under its own GPU reservation with preflight first and storage admission (free - PEAK_GIB >= 12) journaled: nr-recipe-under-lock.sh RECIPE.json OUT_DIR PEAK_GIB NOTE
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"; cd "$repo"; rec="$1"; out="$2"; peak="$3"; note="$4"
python3 scripts/nr-run-recipe.py preflight "$rec"
free=$(df -g "$repo" | awk 'NR==2{print $4}'); echo "$(date -u +%FT%TZ) storage admission $(basename $rec): free ${free} GiB, estimated peak ${peak} GiB, floor 12 GiB" | tee -a "$(dirname "$out")/admission.log"
python3 -c "import sys; sys.exit(0 if $free - $peak >= 12 else 1)" || { echo "STORAGE ADMISSION REFUSED"; exit 5; }
[[ ! -e "$out" ]] || { echo "preserve existing $out" >&2; exit 2; }
python3 scripts/nr-gpu-lock.py run --owner Fable --note "$note; storage free ${free} GiB peak ${peak} GiB" -- python3 scripts/nr-run-recipe.py run "$rec" --out "$out"
