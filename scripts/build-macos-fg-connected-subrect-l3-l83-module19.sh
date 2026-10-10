#!/usr/bin/env bash
set -euo pipefail

repository="$(cd "$(dirname "$0")/.." && pwd)"
attempt="${1:-$repository/build/fg_connected_subrect_l3_l83/macos/module19-$(date -u +%Y%m%dT%H%M%SZ)}"
rom="${2:-$repository/build/fg_connected_tail/macos-l56-l87/attempt-exact-v1/sm120-rcp-correction-rom.bin}"
if [[ -e "$attempt" ]]; then
  echo "Attempt exists; preserve it: $attempt" >&2
  exit 2
fi
build="$attempt/build"
mkdir -p "$build"

python3 -B "$repository/scripts/generate-fg-connected-subrect-l3-l83-module19.py" \
  "$repository/src/fg_postprocess_l86/postprocess_l86_mode0.metal" \
  "$build/postprocess_l86_mode1_subrect.metal"
xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off \
  -c "$build/postprocess_l86_mode1_subrect.metal" \
  -o "$build/postprocess_l86_mode1_subrect.air" \
  >"$build/metal-compile.log" 2>&1
xcrun -sdk macosx metallib "$build/postprocess_l86_mode1_subrect.air" \
  -o "$build/postprocess_l86_mode1_subrect.metallib" \
  >"$build/metallib-link.log" 2>&1
xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc -Wall -Wextra -Werror \
  -Wno-deprecated-declarations \
  -I "$repository/src/fg_connected_subrect_l3_l83" \
  "$repository/src/fg_connected_subrect_l3_l83/macos_module19_mode1.mm" \
  -framework Foundation -framework Metal -o "$build/module19-mode1" \
  >"$build/host-compile.log" 2>&1

for fixture in mode1-a mode1-b; do
  for poison in a5 5a; do
    for process in a b; do
      run_id="$fixture-poison-$poison-process-$process"
      run="$attempt/$run_id"
      mkdir -p "$run"
      "$build/module19-mode1" \
        "$build/postprocess_l86_mode1_subrect.metallib" "$rom" "$run" \
        "$run_id" "$fixture" "$poison" 1 \
        >"$run/stdout.log" 2>"$run/stderr.log"
    done
  done
done

negative="$attempt/reject-mode0"
mkdir -p "$negative"
set +e
"$build/module19-mode1" \
  "$build/postprocess_l86_mode1_subrect.metallib" "$rom" "$negative" \
  reject-mode0 mode1-a a5 0 \
  >"$negative/stdout.log" 2>"$negative/stderr.log"
negative_exit=$?
set -e
if [[ "$negative_exit" -ne 4 || -e "$negative/run-receipt.json" ]]; then
  echo "Mode rejection gate failed: exit=$negative_exit" >&2
  exit 4
fi

python3 -B - "$attempt" "$repository" <<'PY'
import hashlib, json, pathlib, platform, subprocess, sys
root, repository = map(pathlib.Path, sys.argv[1:])
runs = sorted(path for path in root.glob('mode1-*-process-*') if path.is_dir())
if len(runs) != 8:
    raise SystemExit('eight fresh run directories required')
rows=[]
for run in runs:
    receipt=json.loads((run/'run-receipt.json').read_text())
    if receipt['result']!='pass': raise SystemExit(f'failed receipt: {run.name}')
    rows.append({
        'run_id':run.name,'process_id':receipt['process_id'],
        'fixture':receipt['fixture'],'poison_byte':receipt['poison_byte'],
        'output_full_sha256':receipt['output_full_sha256'],
        'output_region_sha256':receipt['output_region_sha256']})
if len({row['process_id'] for row in rows}) != 8:
    raise SystemExit('fresh process ids failed')
for fixture in ('mode1-a','mode1-b'):
    selected=[row for row in rows if row['fixture']==fixture]
    if len({row['output_region_sha256'] for row in selected})!=1:
        raise SystemExit(f'region invariance failed: {fixture}')
    for poison in (90,165):
        repeat=[row for row in selected if row['poison_byte']==poison]
        if len(repeat)!=2 or len({row['output_full_sha256'] for row in repeat})!=1:
            raise SystemExit(f'process repeat failed: {fixture}/{poison}')
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
system=subprocess.run(['system_profiler','SPDisplaysDataType'],check=True,
                      capture_output=True,text=True).stdout.splitlines()
gpu=next((line.split(':',1)[1].strip() for line in system
          if 'Chipset Model:' in line),'unknown')
receipt={
 'schema':'neuralstudio.fg.subrect-module19-mode1-mac-backend.v1',
 'host':{'machine':platform.machine(),'macos':platform.mac_ver()[0],'gpu':gpu},
 'source_sha256':sha(repository/'src/fg_connected_subrect_l3_l83/macos_module19_mode1.mm'),
 'generator_sha256':sha(repository/'scripts/generate-fg-connected-subrect-l3-l83-module19.py'),
 'generated_metal_sha256':sha(root/'build/postprocess_l86_mode1_subrect.metal'),
 'metallib_sha256':sha(root/'build/postprocess_l86_mode1_subrect.metallib'),
 'host_binary_sha256':sha(root/'build/module19-mode1'),
 'processes':rows,'fresh_process_count':8,'fixture_count':2,
 'poison_bytes':[90,165],'two_process_repeat_per_fixture_poison':True,
 'two_poison_region_invariance':True,'region_extent':[32,24,512,288],
 'output_region_bytes':589824,'mode_0_rejected':True,
 'input_immutability':True,'canary_integrity':True,
 'complete_declared_write_coverage':True,
 'expected_or_captured_output_ingress':False,
 'archived_intermediate_ingress':False,
 'address_fields_recorded':False,'result':'pass'}
(root/'backend-receipt.json').write_text(
    json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print('result=pass')
PY

echo "attempt_root=$attempt"
echo "result=pass"
