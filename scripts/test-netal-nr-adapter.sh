#!/bin/zsh
# Caller must acquire the shared Mac GPU reservation; retains every run/failure.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
out="${1:?new output directory required}"
[[ ! -e "$out" ]] || { print -u2 'refusing to overwrite evidence'; exit 2; }
/opt/homebrew/bin/python3.13 - "$repo/../NeuralStudio-nr-fable/scripts/nr-run-recipe.py" <<'PY'
import importlib.util,sys
s=importlib.util.spec_from_file_location('gate',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
ok,why=m.held_reservation('Codex')
print('reservation:',why)
if not ok:raise SystemExit('coordinated Codex GPU reservation required')
PY
mkdir -p "$out/source"
cp -p "$repo/src/macos_host/netal_nr_adapter_smoke.mm" "$repo/src/macos_host/netal_nr_publication_guard.metal" \
  "$repo/src/macos_host/netal_texture_import_ext.metal" "$repo/src/macos_host/netal_nr_color_transfer.metal" "$out/source/"
cp -p "$0" "$out/source/test-as-run.sh"
shasum -a 256 "$out"/source/* > "$out/source-sha256.txt"
xcrun --sdk macosx metal -c -O2 -std=metal3.1 "$out/source/netal_texture_import_ext.metal" -o "$out/formats.air"
xcrun --sdk macosx metal -c -O2 -std=metal3.1 "$out/source/netal_nr_publication_guard.metal" -o "$out/guard.air"
xcrun --sdk macosx metal -c -O2 -fno-fast-math -std=metal3.1 "$out/source/netal_nr_color_transfer.metal" -o "$out/color.air"
xcrun --sdk macosx metallib "$out/formats.air" "$out/guard.air" "$out/color.air" -o "$out/formats.metallib"
xcrun --sdk macosx clang++ -arch arm64 -fobjc-arc -std=c++20 -O2 -Wall -Wextra -Werror \
  -framework Foundation -framework Metal "$out/source/netal_nr_adapter_smoke.mm" -o "$out/adapter-smoke"
shasum -a 256 "$out/adapter-smoke" "$out/formats.metallib" > "$out/build-sha256.txt"
for trial in a b; do
  set +e
  MTL_DEBUG_LAYER=1 "$out/adapter-smoke" "$out/formats.metallib" > "$out/process-$trial.log" 2>&1
  run_status=$?
  set -e
  print "process=$trial exit=$run_status" >> "$out/run-status.txt"
  if [[ "$run_status" != 0 ]]; then tail -n 20 "$out/process-$trial.log"; exit "$run_status"; fi
  tail -n 1 "$out/process-$trial.log"
done
shasum -a 256 "$out"/process-*.log "$out/run-status.txt" > "$out/result-sha256.txt"
