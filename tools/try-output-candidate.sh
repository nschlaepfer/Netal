#!/bin/zsh
# Try a candidate hiluma *output* library: exactness first, speed second.
#
#   tools/try-output-candidate.sh <candidate.metal|candidate.metallib> <label>
#
# Why this exists next to try-hiluma-candidate.sh: that one copies its candidate over
# metallib/hiluma_input_depthinv.metallib, so pointing it at an output candidate swaps a library the
# output pipeline never loads and reports the *baseline* as the candidate's number. This one targets
# hiluma_output_depthinv_exact.metallib, which is what ExactLibrary() actually selects.
#
# Always run the null control first (--null): an early-returning kernel must CHANGE the receipts. If
# it does not, this harness is measuring something else and every number after it is fiction.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
W=/private/tmp/claude-501/-Users-nico/deb7f1d0-b1d5-4951-9b30-84ff130544fc/scratchpad/hil
mkdir -p "$W"
candidate="$1"; label="$2"
target="metallib/hiluma_output_depthinv_exact.metallib"

if [[ "$candidate" == *.metal ]]; then
  xcrun -sdk macosx metal -fmetal-math-mode=safe -c "$candidate" -o "$W/$label.air"
  xcrun -sdk macosx metallib "$W/$label.air" -o "$W/$label.metallib"
  candidate="$W/$label.metallib"
fi

bundle="$W/bundle-$label"
rm -rf "$bundle"; cp -R "$repo/work/re9-sr-runtime-bundle" "$bundle"
cp "$candidate" "$bundle/$target"
# The runtime hash-checks every file in the manifest, which is the point of the manifest. A
# candidate is a legitimately different artifact, so its row is restated rather than bypassed - the
# check stays on for every other file.
python3 - "$bundle" "$target" <<'PYEOF'
import hashlib, json, pathlib, sys
bundle, target = pathlib.Path(sys.argv[1]), sys.argv[2]
path = bundle / "manifest.json"
manifest = json.loads(path.read_text())
data = (bundle / target).read_bytes()
for row in manifest["files"]:
    if row.get("path") == target:
        row["sha256"] = hashlib.sha256(data).hexdigest()
        row["bytes"] = len(data)
        break
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PYEOF

expected=(a15dd978 e337b2ba a4925154 56478d20)
receipt="$W/receipt-$label.txt"
env NEURALSTUDIO_SR_PWIN_PIPELINE=1 NEURALSTUDIO_SR_PWIN_SUFFIX=_hw7 \
    NEURALSTUDIO_SR_DEPTH_INPUT_LIBRARY=hiluma_input_depthinv \
    NEURALSTUDIO_SR_TEMPORAL_FRAME_COUNT=4 \
    "$repo/build/macos_host/dlss-sr-session-temporal-probe" \
    "$bundle" "$repo/work/rtx-dlss-sr-temporal-4frame-20260901-v1/capture" "$receipt" \
    > "$W/probe-$label.log" 2>&1 || true

if [[ ! -s "$receipt" ]]; then print "$label: NO RECEIPT"; tail -3 "$W/probe-$label.log"; exit 1; fi
got=("${(@f)$(grep output_sha "$receipt" | cut -d= -f2 | cut -c1-8)}")
ok=1
for i in {1..4}; do [[ "${got[$i]:-}" == "${expected[$i]}" ]] || ok=0; done
ms=$(grep gpu_ms "$receipt" | cut -d= -f2 | sort -n | awk '{a[NR]=$1} END {printf "min=%.2f med=%.2f", a[1], a[int((NR+1)/2)]}')
if (( ok )); then print "$label: EXACT    $ms"; else print "$label: DIVERGED $ms  ${got[*]}"; fi
