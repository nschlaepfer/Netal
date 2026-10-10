#!/bin/zsh
# Actual delayed reset worker with a SIMULATED renderer. Never launches a game.
# Invoke only inside a Codex GPU reservation, with Opus's worker READY archive.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
archive="${1:?actual worker archive required}"
bundle="${2:?closed runtime bundle required}"
out="${3:?new output directory required}"
runtime_headers="${4:?matching frozen runtime header directory required}"
[[ $# == 4 && -f "$archive" && -f "$bundle/manifest.json" && ! -e "$out" ]] || exit 2
[[ -f "$runtime_headers/nr_experimental_api.h" && -f "$runtime_headers/nr_experimental_reset_worker.h" &&
   -f "$runtime_headers/runtime/nr_encoder_scope.h" ]] || exit 2
validation="${NR_ADAPTER_DEBUG_LAYER:-1}"
[[ "$validation" == 0 || "$validation" == 1 ]] || exit 2
phase_gates="${NR_ADAPTER_PHASE_GATES:-0}"
[[ "$phase_gates" == 0 || "$phase_gates" == 1 ]] || exit 2
export NR_ADAPTER_PHASE_GATES="$phase_gates"
python=/opt/homebrew/bin/python3.13
"$python" -I - "$repo/../NeuralStudio-nr-fable/scripts/nr-run-recipe.py" <<'PY'
import importlib.util,sys
s=importlib.util.spec_from_file_location('gate',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
ok,why=m.held_reservation('Codex');print('reservation:',why)
if not ok:raise SystemExit('coordinated Codex GPU reservation required')
PY
mkdir -p "$out/source/src/macos_host" "$out/source/src/nr_experimental"
out="$(cd "$out" && pwd)"
bundle="$(cd "$bundle" && pwd)"
# This is a local child test, not an opt-in injected game launch. Prevent
# inherited inline/approximate/fault modes or an unverified external-input path.
unset NETAL_NR_EXPERIMENTAL NETAL_NR_MODE NETAL_NR_ARITHMETIC NETAL_NR_ARM_FILE NETAL_NR_CAPTURE_DIR
unset NETAL_NR_DELAYED_PHASE_GATES
unset NR_ADAPTER_EXTERNAL_INPUT NR_ADAPTER_SOURCE_FAULT
if [[ -n "${NR_ADAPTER_REPLAY_CAPTURE:-}" ]]; then
  "$python" -I - "$NR_ADAPTER_REPLAY_CAPTURE" "$out/source" <<'PY'
import hashlib,json,pathlib,shutil,sys
root=pathlib.Path(sys.argv[1]);dst=pathlib.Path(sys.argv[2])
m=json.loads((root/'capture.json').read_text());entries=[e for e in m['files'] if e['file']=='game-input.bin']
assert len(entries)==1,'exactly one external game input required'
e=entries[0];raw=(root/'game-input.bin').read_bytes()
assert (e['width'],e['height'],e['metal_pixel_format'],e['bytes_per_pixel'])==(1920,1200,92,4)
assert len(raw)==e['bytes']==1920*1200*4 and hashlib.sha256(raw).hexdigest()==e['sha256']
shutil.copy2(root/'game-input.bin',dst/'external-game-input.rg11b10f')
shutil.copy2(root/'capture.json',dst/'external-provenance.json')
assert hashlib.sha256((dst/'external-game-input.rg11b10f').read_bytes()).hexdigest()==e['sha256']
print('external input verified:',e['sha256'])
PY
  export NR_ADAPTER_EXTERNAL_INPUT="$out/source/external-game-input.rg11b10f"
fi
for name in netal_nr_delayed_adapter_smoke.mm netal_nr_delayed_callback_lifetime_test.mm netal_nr_delayed_once_state_test.cpp netal_nr_experimental_bridge.mm \
  netal_nr_delayed_once_bridge.h netal_nr_delayed_once_state.h netal_nr_frame_capture.h \
  netal_nr_present_observer.h netal_nr_aspect.h netal_nr_aspect.metal \
  netal_nr_color_transfer.metal netal_nr_publication_guard.metal netal_texture_import_ext.metal; do
  cp -p "$repo/src/macos_host/$name" "$out/source/src/macos_host/$name"
done
for name in nr_experimental_api.h nr_experimental_reset_worker.h; do
  cp -p "$runtime_headers/$name" "$out/source/src/nr_experimental/$name"
done
cp -p "$runtime_headers/runtime/nr_encoder_scope.h" "$out/source/src/nr_experimental/nr_encoder_scope.h"
cp -p "$archive" "$out/runtime.a"
cp -p "$0" "$out/source/test-as-run.sh"
cp -p "$bundle/manifest.json" "$out/source/runtime-manifest.json"
# Verify the full declared bundle inventory, not only its manifest filename.
"$python" -I - "$bundle" "$out" <<'PY'
import hashlib,json,pathlib,re,sys
bundle=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((bundle/'manifest.json').read_text());files=m['files'];assert isinstance(files,dict) and files
for name,h in files.items():
    p=pathlib.PurePosixPath(name)
    assert not p.is_absolute() and '..' not in p.parts and re.fullmatch('[0-9a-f]{64}',h),'unsafe inventory'
    assert '/fixtures/launch-' not in '/'+name,'research fixtures are not a closed game bundle'
    target=bundle/name;assert target.resolve().is_relative_to(bundle.resolve()),'bundle escape'
    assert sha(target)==h,'bundle identity drift: '+name
record={'schema':'neuralstudio.nr.delayed-smoke-source.v1','bundle':str(bundle),'bundle_manifest_sha256':sha(bundle/'manifest.json'),
        'bundle_file_count':len(files),'archive_sha256':sha(out/'runtime.a'),
        'sources':{str(p.relative_to(out)):sha(p) for p in sorted((out/'source').rglob('*')) if p.is_file()},
        'actual_model_required':True,'live_game':False,'internal_activation_ingress':False,'expected_output_ingress':False}
with (out/'source-receipt.json').open('x') as f:json.dump(record,f,indent=2)
PY
for spec in 'netal_texture_import_ext:formats' 'netal_nr_publication_guard:guard' 'netal_nr_aspect:aspect'; do
  xcrun --sdk macosx metal -c -O2 -std=metal3.1 "$out/source/src/macos_host/${spec%%:*}.metal" -o "$out/${spec##*:}.air"
done
xcrun --sdk macosx metal -c -O2 -fno-fast-math -std=metal3.1 "$out/source/src/macos_host/netal_nr_color_transfer.metal" -o "$out/color.air"
xcrun --sdk macosx metallib "$out/formats.air" "$out/guard.air" "$out/aspect.air" "$out/color.air" -o "$out/formats.metallib"
counter_flag=-DNR_DELAYED_HAS_DROPPED_COUNTER=0
labels_flag=-DNR_DELAYED_HAS_SUBMISSION_LABELS=0
if rg -q 'reset_worker_dropped_duration_samples' "$out/source/src/nr_experimental/nr_experimental_reset_worker.h"; then
  counter_flag=-DNR_DELAYED_HAS_DROPPED_COUNTER=1
fi
if rg -q 'reset_worker_submissions' "$out/source/src/nr_experimental/nr_experimental_reset_worker.h"; then
  labels_flag=-DNR_DELAYED_HAS_SUBMISSION_LABELS=1
fi
xcrun --sdk macosx clang++ -arch arm64 -fobjc-arc -std=c++20 -O2 -Wall -Wextra -Werror -mmacosx-version-min=26.0 "$counter_flag" "$labels_flag" \
  -framework Foundation -framework Metal -framework MetalFX "$out/source/src/macos_host/netal_nr_delayed_adapter_smoke.mm" \
  "$out/runtime.a" -o "$out/delayed-adapter-smoke"
xcrun --sdk macosx clang++ -arch arm64 -fobjc-arc -std=c++20 -O1 -g -fsanitize=address \
  -fsanitize-address-use-after-scope -Wall -Wextra -Werror -mmacosx-version-min=26.0 "$labels_flag" \
  -framework Foundation -framework Metal -framework MetalFX \
  "$out/source/src/macos_host/netal_nr_delayed_callback_lifetime_test.mm" "$out/runtime.a" -o "$out/callback-lifetime-test"
"$out/callback-lifetime-test" > "$out/callback-lifetime.log" 2>&1
tail -n 2 "$out/callback-lifetime.log"
xcrun --sdk macosx clang++ -arch arm64 -std=c++20 -O1 -g -fsanitize=address -Wall -Wextra -Werror \
  "$out/source/src/macos_host/netal_nr_delayed_once_state_test.cpp" -o "$out/delayed-state-test"
"$out/delayed-state-test" > "$out/delayed-state.log" 2>&1
tail -n 1 "$out/delayed-state.log"
shasum -a 256 "$out/delayed-adapter-smoke" "$out/formats.metallib" "$out/source-receipt.json" > "$out/build-sha256.txt"
for poison in a5 5a; do
  set +e
  MTL_DEBUG_LAYER="$validation" "$out/delayed-adapter-smoke" "$bundle" "$out/formats.metallib" "$poison" "$out/process-$poison" > "$out/process-$poison.log" 2>&1
  result=$?
  set -e
  print "poison=$poison exit=$result" >> "$out/run-status.txt"
  tail -n 8 "$out/process-$poison.log"
  [[ "$result" == 0 ]] || exit "$result"
done
"$python" -I - "$out" "$bundle" <<'PY'
import hashlib,json,pathlib,sys,os
out,bundle=map(pathlib.Path,sys.argv[1:]);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
initial=json.loads((out/'source-receipt.json').read_text())
assert sha(bundle/'manifest.json')==initial['bundle_manifest_sha256'],'manifest changed'
for name,h in json.loads((bundle/'manifest.json').read_text())['files'].items():assert sha(bundle/name)==h,'bundle changed'
for name,h in initial['sources'].items():assert sha(out/name)==h,'source changed'
assert sha(out/'runtime.a')==initial['archive_sha256'],'archive changed'
receipts=[];planes=[]
for poison in ('a5','5a'):
    root=out/f'process-{poison}';r=json.loads((root/'receipt.json').read_text());receipts.append(r)
    assert r['result']=='pass' and r['poison']==poison and not r['caller_retained_references']
    assert r['phase_gates']==(os.environ['NR_ADAPTER_PHASE_GATES']=='1')
    if r['phase_gates']:assert r['compute_gate_original_frames']>=4 and r['ready_gate_original_frames']==4
    assert r['publication_count']==1 and r['publication_frame']>r['source_frame'] and r['original_recovered']
    assert r['subsequent_original_frames']>=3 and r['original_frames_during_compute']>=2
    assert not r['game_connected'] and not r['screen_presented'] and not r['exact_rtx_parity']
    assert sum(f['published'] for f in r['frames'])==1
    assert all(f['original_bytes_equal'] for f in r['frames'] if not f['published'])
    stages={}
    for stage,count in [('source',1),('input',2),('computed',3),('published',4)]:
        d=root/'captures'/stage/f"frame-{r['source_frame']}-stream-7"
        m=json.loads((d/'capture.json').read_text());assert len(m['files'])==count
        assert m['execution_mode']=='delayed-reset-once' and m['gpu_completed'] and not m['presentation_verified']
        assert bool(m['nr_export_completed'])==(stage=='published')
        values={}
        for e in m['files']:
            p=d/e['file'];assert p.stat().st_size==e['bytes'] and sha(p)==e['sha256'];values[e['file']]=e['sha256']
        stages[stage]=values
    for name,h in stages['published'].items():
        for values in stages.values():
            if name in values:assert values[name]==h,'phase buffer changed after completion'
    assert sha(root/'source.rg11b10f')==stages['published']['game-input.bin']
    assert sha(root/'published.rg11b10f')==stages['published']['game-output.bin']
    planes.append(stages['published'])
assert receipts[0]['pid']!=receipts[1]['pid'],'fresh native processes required'
assert receipts[0]['model']==receipts[1]['model'] and receipts[0]['weights']==receipts[1]['weights']
assert planes[0]==planes[1],'external-poison/process invariance failed'
summary={'schema':'neuralstudio.nr.delayed-adapter-repeats.v1','result':'pass','processes':[r['pid'] for r in receipts],
         'model':receipts[0]['model'],'weights':receipts[0]['weights'],'four_planes_equal':True,'planes':planes[0],
         'single_delayed_publication_and_original_recovery':True,'actual_model':True,'live_game':False,
         'internal_write_coverage_proven':False,'visual_accepted':False,'rtx_parity':False,'playable':False,
         'files':{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and 'source' not in p.relative_to(out).parts}}
with (out/'summary.json').open('x') as f:json.dump(summary,f,indent=2)
print('Two fresh unretained processes passed one delayed actual-model publication and original-frame recovery. Simulated renderer only; no live/visual/playability/parity acceptance.')
PY
