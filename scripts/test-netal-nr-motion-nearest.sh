#!/bin/zsh
# Offline primitive only. Must run under the coordinated Codex GPU lock.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
out="${1:?new evidence directory required}"
[[ ! -e "$out" ]] || exit 2
python=/opt/homebrew/bin/python3.13
"$python" -I - "$repo/../NeuralStudio-nr-fable/scripts/nr-run-recipe.py" <<'PY'
import importlib.util,sys
s=importlib.util.spec_from_file_location('gate',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
ok,why=m.held_reservation('Codex');print(why)
if not ok:raise SystemExit('Codex GPU reservation required')
PY
mkdir -p "$out/source";out="$(cd "$out" && pwd)"
for name in netal_nr_motion_nearest_candidate.metal netal_nr_motion_nearest_test.mm netal_nr_motion_geometry.h netal_nr_aspect.h; do
  cp -p "$repo/src/macos_host/$name" "$out/source/$name"
done
cp -p "$0" "$out/test-as-run.sh"
xcrun --sdk macosx metal --version > "$out/compiler.txt"
xcrun --sdk macosx metal -c -O2 -fno-fast-math -ffp-contract=off -std=metal3.1 \
  "$out/source/netal_nr_motion_nearest_candidate.metal" -o "$out/motion.air"
xcrun --sdk macosx metallib "$out/motion.air" -o "$out/motion.metallib"
xcrun --sdk macosx clang++ -fobjc-arc -std=c++20 -O1 -g -Wall -Wextra -Werror \
  -fsanitize=address,undefined -fno-omit-frame-pointer -framework Foundation -framework Metal \
  "$out/source/netal_nr_motion_nearest_test.mm" -o "$out/motion-test"
failed=0
for poison in a5 5a; do
  set +e
  MTL_DEBUG_LAYER=1 UBSAN_OPTIONS=halt_on_error=1 "$out/motion-test" "$out/motion.metallib" "$poison" "$out/process-$poison" > "$out/process-$poison.log" 2>&1
  rc=$?;set -e
  print "poison=$poison exit=$rc" >> "$out/status.txt"
  tail -n 8 "$out/process-$poison.log"
  [[ "$rc" == 0 ]] || failed=1
done
[[ "$failed" == 0 ]] || exit 1
"$python" -I - "$out" <<'PY'
import json,pathlib,sys,hashlib
p=pathlib.Path(sys.argv[1]);r=[json.loads((p/f'process-{v}/receipt.json').read_text()) for v in ('a5','5a')]
assert r[0]['pid']!=r[1]['pid'] and all(x['pass'] for x in r)
for x in r:
 assert not x['game_connected'] and not x['temporal_admission'] and len(x['cases'])==6
 for c in x['cases']:
  f=p/f"process-{x['poison']}"/(c['case']+'.rg16f')
  assert hashlib.sha256(f.read_bytes()).hexdigest()==c['output_sha256']
assert r[0]['cases']==r[1]['cases'],'poison/process variance'
files={str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*') if f.is_file()}
summary={'schema':'netal.nr.motion-nearest-repeat.v1','pass':True,'processes':[x['pid'] for x in r],
 'case_count':6,'cases':r[0]['cases'],'game_convention_verified':False,'temporal_enabled':False,'files_sha256':files}
with (p/'summary.json').open('x') as f:json.dump(summary,f,indent=2)
print('PASS six synthetic cases in two fresh poisoned processes; no temporal/game admission')
PY
