#!/bin/zsh
# Generate the batched pipeline set for one output resolution, and verify it is complete.
#
#   scripts/netal-complete-profile.sh 1728x972
#
# Why this exists: a resolution with no profiles/<WxH>/ directory gets a synthesized pipeline set,
# and a layer with no batched pipeline silently falls back to the transpiled per-window kernel -
# correct, roughly twice the GPU time. 1728x972 shipped with 6 of 11 layers and cost Resident Evil
# Requiem about 17 ms a frame with nothing in any log to say so. The failure is invisible because
# the fallback is a legitimate code path.
#
# The two steps are the same ones the session and the generator already implement; this only puts
# them together and refuses to report success on a set that is still short.
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"; repository="$(cd "$script_dir/.." && pwd)"
bundle="$repository/work/re9-sr-runtime-bundle"
probe="$repository/build/macos_host/dlss-sr-session-temporal-probe"
capture="$repository/work/rtx-dlss-sr-temporal-4frame-20260901-v1/capture"

fail() { print -u2 "\n  $1\n"; exit 1 }

[[ $# -eq 1 ]] || fail "usage: netal-complete-profile.sh <WxH>   (e.g. 1728x972)"
resolution="$1"
[[ "$resolution" =~ '^[0-9]+x[0-9]+$' ]] || fail "expected a resolution like 1728x972, got: $resolution"
[[ -x "$probe" ]] || fail "probe not built: run scripts/build-macos-dlss-sr-session-temporal-probe.sh"
[[ -d "$bundle" ]] || fail "no runtime bundle at $bundle"

# A shipped profile already has its own tuned pipelines; synthesising over it would be wrong.
if [[ -d "$bundle/profiles/$resolution" ]]; then
  print "result=pass  $resolution is a shipped profile — nothing to synthesise."
  exit 0
fi

target="$bundle/pipeline-synth/$resolution"
before=0
[[ -d "$target" ]] && before=$(ls "$target" | grep -oE '(enc|dec)[0-9]' | sort -u | wc -l | tr -d ' ')
print "profile   $resolution"
print "  before  $before of 11 layers"

# 1. the session owns the synthesis rules, so it exports the artefacts; asking it for this geometry
#    is enough, no game required.
export_dir="$(mktemp -d)"; trap 'rm -rf "$export_dir"' EXIT
env NEURALSTUDIO_SR_PWIN_PIPELINE=1 NEURALSTUDIO_SR_PWIN_SUFFIX=_hw7 \
    NEURALSTUDIO_SR_DEPTH_INPUT_LIBRARY=hiluma_input_depthinv \
    NEURALSTUDIO_SR_TEMPORAL_FRAME_COUNT=4 \
    NEURALSTUDIO_SR_PROFILE="$resolution" NEURALSTUDIO_SR_EXPORT_SYNTH_DIR="$export_dir" \
    "$probe" "$bundle" "$capture" /dev/null >/dev/null 2>&1 || true
exported=$(find "$export_dir" -type f 2>/dev/null | wc -l | tr -d ' ')
(( exported > 0 )) || fail "the session exported no synthesis artefacts for $resolution"
print "  exported $exported artefacts"

# 2. generate the pipelines from them
python3 "$repository/tools/pwin_pipeline/build-profile-pipelines.py" --synth "$export_dir" "$resolution" \
  | sed 's/^/    /' || fail "pipeline generation failed for $resolution"

after=$(ls "$target" 2>/dev/null | grep -oE '(enc|dec)[0-9]' | sort -u | wc -l | tr -d ' ')
print "  after   $after of 11 layers"

# The whole point: a short set is a silent 2x slowdown, so a partial result is a failure, not a
# warning. Report it as one.
if (( after != 11 )); then
  missing=$(python3 - "$target" <<'PY'
import sys, os, re
want = {f"{p}{i}" for p, n in (("enc", 5), ("dec", 6)) for i in range(n)}
have = {m.group(0) for f in os.listdir(sys.argv[1]) if (m := re.search(r"(enc|dec)[0-9]", f))}
print(",".join(sorted(want - have)) or "-")
PY
)
  fail "still incomplete: $after of 11 layers, missing $missing"
fi

print "\nresult=pass  $resolution complete at 11 of 11 layers"
