#!/bin/zsh
# Acceptance run for the NR ViT-1D family: builds into a fresh attempt directory, then runs every ViT block
# (31..38) connected - each block's tokens are the previous block's live projection output, block 31 starts
# from the recorded repack output - for two poisons and two ordinals (four fresh processes), comparing every
# stage with the RTX recording. The verifier seals the receipt.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
attempt="${1:?Provide a fresh output directory}"
fixtures="${2:-$repo/build/nr_vit/fixtures}"
zsh "$repo/scripts/build-nr-vit.sh" "$attempt"
for poison in a5 5a; do
  for ordinal in a b; do
    prev=""
    for block in $(seq 31 38); do
      run="$attempt/$poison-$ordinal/block$block"; mkdir -p "$attempt/$poison-$ordinal"
      if [[ -n "$prev" ]]; then extra=(--input-tokens "$prev/projection.bin"); else extra=(); fi
      "$attempt/vit_probe" "$attempt/vit.metallib" "$fixtures" "$block" "$poison" "$run" "${extra[@]}" > "$run.stdout.log" 2> "$run.stderr.log" || { echo "block $block $poison-$ordinal FAILED"; cat "$run.stderr.log"; exit 1; }
      tail -1 "$run.stdout.log" | sed "s/^/$poison-$ordinal block $block: /"
      prev="$run"
    done
  done
done
/opt/homebrew/bin/python3.13 "$repo/scripts/verify-nr-vit.py" --attempt "$attempt" --fixtures "$fixtures" --output "$attempt/verification.json"
