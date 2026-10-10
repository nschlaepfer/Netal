#!/bin/zsh
# Offline texture-copy integration gate. No NR library, shader, or game launch.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
out="${1:?new evidence directory required}"
[[ ! -e "$out" ]] || exit 2
/opt/homebrew/bin/python3.13 -I - "$repo/../NeuralStudio-nr-fable/scripts/nr-run-recipe.py" <<'PY'
import importlib.util,sys
s=importlib.util.spec_from_file_location('gate',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
ok,why=m.held_reservation('Codex');print(why)
if not ok:raise SystemExit('Codex GPU reservation required')
PY
mkdir -p "$out/source/macos_host" "$out/source/nr_experimental"
out="$(cd "$out" && pwd)"
for name in netal_nr_motion_pair_capture.h netal_nr_motion_pair_capture_test.mm netal_nr_frame_capture.h netal_nr_aspect.h; do
  cp -p "$repo/src/macos_host/$name" "$out/source/macos_host/$name"
done
cp -p "$repo/src/nr_experimental/nr_encoder_scope.h" "$out/source/nr_experimental/"
cp -p "$0" "$out/test-as-run.sh"
xcrun --sdk macosx clang++ -fobjc-arc -fobjc-arc-exceptions -std=c++20 -O1 -g -Wall -Wextra -Werror \
  -fsanitize=address,undefined -fno-omit-frame-pointer -framework Foundation -framework Metal \
  "$out/source/macos_host/netal_nr_motion_pair_capture_test.mm" -o "$out/test" > "$out/build.log" 2>&1
for run in 1 2; do
  MTL_DEBUG_LAYER=1 UBSAN_OPTIONS=halt_on_error=1 "$out/test" "$out/process-$run" > "$out/process-$run.log" 2>&1
  tail -n 4 "$out/process-$run.log"
done
/opt/homebrew/bin/python3.13 -I - "$out" <<'PY'
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);pids=[];hashes=[]
for run in (1,2):
 records=[json.loads((p/f'process-{run}/pair-{i}-frame-{100+i}/capture.json').read_text()) for i in (0,1)]
 assert records[0]['pid']==records[1]['pid'];pids.append(records[0]['pid'])
 assert records[1]['source_frame']==records[0]['source_frame']+1
 assert records[0]['stream']==records[1]['stream']
 for i,r in enumerate(records):
  assert r['gpu_completed'] and r['encoded'] and not r['actual_nr'] and not r['temporal_admitted']
  assert len(r['files'])==2
  for f in r['files']:
   digest=hashlib.sha256((p/f'process-{run}/pair-{i}-frame-{100+i}'/f['file']).read_bytes()).hexdigest()
   assert digest==f['sha256'];hashes.append(digest)
assert len(set(pids))==2 and hashes[:4]==hashes[4:]
receipt={'schema':'netal.nr.motion-pair-offline.v1','pass':True,'pids':pids,'actual_game':False,'temporal_admission':False,
 'files':{str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*') if f.is_file()}}
with (p/'summary.json').open('x') as f:json.dump(receipt,f,indent=2)
print('PASS repeat process/capture checks; synthetic input only')
PY
