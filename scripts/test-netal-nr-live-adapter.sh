#!/bin/zsh
# Actual model adapter test: synthetic GPU source or explicit external-input
# offline replay. Neither mode establishes live-game/visual/parity acceptance.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
archive="${1:?runtime archive required}"; bundle="${2:?immutable staged bundle required}"; out="${3:?new output directory required}"
cb_args=()
[[ -z "${4:-}" || "${4:-}" == --unretained ]] || exit 2
[[ "${4:-}" == --unretained ]] && cb_args+=(--unretained)
validation="${NR_ADAPTER_DEBUG_LAYER:-1}"
[[ "$validation" == 0 || "$validation" == 1 ]] || exit 2
[[ -f "$archive" && -f "$bundle/manifest.json" && ! -e "$out" ]] || exit 2
/opt/homebrew/bin/python3.13 - "$repo/../NeuralStudio-nr-fable/scripts/nr-run-recipe.py" <<'PY'
import importlib.util,sys
s=importlib.util.spec_from_file_location('gate',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
ok,why=m.held_reservation('Codex');print('reservation:',why)
if not ok:raise SystemExit('coordinated Codex GPU reservation required')
PY
mkdir -p "$out/source/src/macos_host" "$out/source/src/nr_experimental"
# Snapshot only the external NR-boundary game input, after manifest verification.
# Recorded model inputs/outputs remain comparison-only and are not loaded here.
unset NR_ADAPTER_EXTERNAL_INPUT
if [[ -n "${NR_ADAPTER_REPLAY_CAPTURE:-}" ]]; then
  /opt/homebrew/bin/python3.13 - "$NR_ADAPTER_REPLAY_CAPTURE" "${NR_ADAPTER_GAME_HEIGHT:-1080}" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); height=int(sys.argv[2])
manifest=json.loads((root/'capture.json').read_text())
entries=[e for e in manifest['files'] if e['file']=='game-input.bin']
assert len(entries)==1, 'exactly one external game input required'
e=entries[0]; raw=(root/'game-input.bin').read_bytes()
assert (e['width'],e['height'],e['metal_pixel_format'],e['bytes_per_pixel'])==(1920,height,92,4), 'external input geometry/format mismatch'
assert len(raw)==e['bytes']==1920*height*4, 'external input size mismatch'
assert hashlib.sha256(raw).hexdigest()==e['sha256'], 'external input hash mismatch'
print('verified_external_input_sha256='+e['sha256'])
PY
  cp -p "$NR_ADAPTER_REPLAY_CAPTURE/game-input.bin" "$out/source/external-game-input.rg11b10f"
  cp -p "$NR_ADAPTER_REPLAY_CAPTURE/capture.json" "$out/source/external-provenance.json"
  cmp "$NR_ADAPTER_REPLAY_CAPTURE/game-input.bin" "$out/source/external-game-input.rg11b10f"
  export NR_ADAPTER_EXTERNAL_INPUT="$(cd "$out/source" && pwd)/external-game-input.rg11b10f"
  /opt/homebrew/bin/python3.13 - "$NR_ADAPTER_EXTERNAL_INPUT" "$out/source/external-provenance.json" <<'PY'
import hashlib,json,pathlib,sys
raw=pathlib.Path(sys.argv[1]).read_bytes()
e=next(e for e in json.loads(pathlib.Path(sys.argv[2]).read_text())['files'] if e['file']=='game-input.bin')
assert len(raw)==e['bytes'] and hashlib.sha256(raw).hexdigest()==e['sha256'], 'snapshot changed during transfer'
PY
  shasum -a 256 "$NR_ADAPTER_EXTERNAL_INPUT" "$out/source/external-provenance.json" > "$out/external-input-sha256.txt"
fi
cp -p "$repo/src/macos_host/netal_nr_live_adapter_smoke.mm" "$repo/src/macos_host/netal_nr_experimental_bridge.mm" \
  "$repo/src/macos_host/netal_nr_publication_guard.metal" "$repo/src/macos_host/netal_texture_import_ext.metal" \
  "$repo/src/macos_host/netal_nr_aspect.h" "$repo/src/macos_host/netal_nr_aspect.metal" \
  "$repo/src/macos_host/netal_nr_color_transfer.metal" \
  "$repo/src/macos_host/netal_nr_frame_capture.h" "$repo/src/macos_host/netal_nr_present_observer.h" "$out/source/src/macos_host/"
cp -p "$repo/src/nr_experimental/nr_experimental_api.h" "$repo/src/nr_experimental/nr_encoder_scope.h" "$out/source/src/nr_experimental/"
cp -p "$archive" "$out/runtime.a"
cp -p "$0" "$out/source/test-as-run.sh"
cp -p "$repo/scripts/compare-nr-frame-captures.py" "$out/source/compare-nr-frame-captures.py"
shasum -a 256 "$out"/source/src/macos_host/* "$out"/source/src/nr_experimental/* "$out/source/compare-nr-frame-captures.py" "$out/runtime.a" "$bundle/manifest.json" > "$out/source-sha256.txt"
xcrun --sdk macosx metal -c -O2 -std=metal3.1 "$out/source/src/macos_host/netal_texture_import_ext.metal" -o "$out/formats.air"
xcrun --sdk macosx metal -c -O2 -std=metal3.1 "$out/source/src/macos_host/netal_nr_publication_guard.metal" -o "$out/guard.air"
xcrun --sdk macosx metal -c -O2 -std=metal3.1 "$out/source/src/macos_host/netal_nr_aspect.metal" -o "$out/aspect.air"
xcrun --sdk macosx metal -c -O2 -fno-fast-math -std=metal3.1 "$out/source/src/macos_host/netal_nr_color_transfer.metal" -o "$out/color.air"
xcrun --sdk macosx metallib "$out/formats.air" "$out/guard.air" "$out/aspect.air" "$out/color.air" -o "$out/formats.metallib"
xcrun --sdk macosx clang++ -arch arm64 -fobjc-arc -std=c++20 -O2 -Wall -Wextra -Werror -mmacosx-version-min=26.0 \
  -framework Foundation -framework Metal -framework MetalFX "$out/source/src/macos_host/netal_nr_live_adapter_smoke.mm" \
  "$out/runtime.a" -o "$out/live-adapter-smoke"
shasum -a 256 "$out/live-adapter-smoke" "$out/formats.metallib" > "$out/build-sha256.txt"
for poison in a5 5a; do
  set +e
  MTL_DEBUG_LAYER="$validation" "$out/live-adapter-smoke" "$bundle" "$out/formats.metallib" "$poison" "$out/output-$poison.rg11b10f" "${cb_args[@]}" > "$out/process-$poison.log" 2>&1
  run_status=$?
  set -e
  print "poison=$poison exit=$run_status" >> "$out/run-status.txt"
  tail -n 8 "$out/process-$poison.log"
  [[ "$run_status" == 0 ]] || exit "$run_status"
done
cmp "$out/output-a5.rg11b10f" "$out/output-5a.rg11b10f"
cmp "$out/output-a5.rg11b10f.producer" "$out/output-5a.rg11b10f.producer"
shasum -a 256 "$out"/output-*.rg11b10f "$out"/output-*.producer "$out"/process-*.log "$out/run-status.txt" > "$out/result-sha256.txt"
if [[ -n "${NR_ADAPTER_REPLAY_CAPTURE:-}" && -n "${NR_ADAPTER_CAPTURE_DIR:-}" ]]; then
  # Only AFTER both executions are complete may archived output enter comparison.
  for frame in 123456 0; do
    /opt/homebrew/bin/python3.13 "$out/source/compare-nr-frame-captures.py" \
      "$NR_ADAPTER_REPLAY_CAPTURE/capture.json" "$NR_ADAPTER_CAPTURE_DIR/frame-$frame-stream-7/capture.json" \
      > "$out/replay-comparison-$frame.json"
  done
  shasum -a 256 "$out"/replay-comparison-*.json > "$out/comparison-sha256.txt"
fi
if [[ -n "${NR_ADAPTER_SOURCE_FAULT:-}" ]]; then
  print 'both fresh processes preserved the original frame after injected source failure; no model output published; standalone safety test only'
else
  print 'both fresh processes published identical actual-model output; standalone synthetic/offline adapter only, not game/visual/temporal/parity/performance acceptance'
fi
