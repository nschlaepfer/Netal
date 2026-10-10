#!/bin/zsh
# Attach the Apple-native DLSS SR bridge to a CrossOver/GPTK game that is ALREADY running,
# or wait for one to appear. Launch the game yourself however you like (Steam, CrossOver, a
# shortcut); this only injects.
#
#   scripts/mac-dlss-attach.sh --exe re9.exe       # wait for one particular executable
#   scripts/mac-dlss-attach.sh --exe 'Test Game.exe' --bottle 'Other Bottle'
#   scripts/mac-dlss-attach.sh --exe re9.exe --bottle Steam --timeout 600
#
# The target must have started with Netal's helper search path. Model/input compatibility
# and visual quality still need validation per game; discovery is not a support guarantee.
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
source "$script_dir/netal-wine-output.sh"
repository="$(cd "$script_dir/.." && pwd)"
config_tool="$repository/build/netal-tools/netal-launch-config"
[[ -x "$config_tool" ]] || { print -u2 "Build the runtime helpers before attaching."; exit 2; }
cx_root="${NETAL_CROSSOVER_ROOT:-/Applications/CrossOver.app/Contents/SharedSupport/CrossOver}"
bottle="Steam"; exe=""; timeout=900; suffix="${PWIN_SUFFIX:-_hw7}"; extra=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bottle) bottle="$2"; shift 2 ;;
    --exe) exe="$2"; shift 2 ;;
    --process) print -u2 'Process patterns are no longer supported; select the executable and bottle explicitly.'; exit 64 ;;
    --timeout) timeout="$2"; shift 2 ;;
    --pwin-suffix) suffix="$2"; shift 2 ;;
    --jitter-sign=*|--mv-sign=*|--dump=*|--force-vsync) extra+=("$1"); shift ;;
    *) print -u2 "usage: mac-dlss-attach.sh --exe game.exe [--bottle B] [--timeout S] [--pwin-suffix _hw7] [--force-vsync]"; exit 1 ;;
  esac
done
[[ -n "$exe" && "$exe" != */* && "$exe" != *\\* ]] || { print -u2 -- '--exe <bare game executable name> is required; process command lines cannot reliably identify names containing spaces'; exit 2; }
controller="$repository/build/re9_crossover/neuralstudio_re9_inject.exe"
hook="$repository/build/re9_crossover/netal_wine_host_v1.dll"
[[ -f "$controller" && -f "$hook" ]] || { print -u2 'Build the Wine helpers before attaching'; exit 2; }
[[ "$timeout" == <-> && "$timeout" -gt 0 ]] || { print -u2 'Timeout must be a positive integer'; exit 2; }
controller_win="Z:${controller//\//\\}"
hook_win="Z:${hook//\//\\}"
print "waiting for $exe in bottle $bottle (up to ${timeout}s)"
pid=""
for _ in $(seq 1 $(( (timeout + 2) / 3 ))); do
  if found="$(netal_wine_capture "$cx_root/bin/wine" --bottle "$bottle" --no-update "$controller_win" --find "$exe")"; then
    if ! queried="$(netal_wine_capture "$cx_root/bin/wine" --bottle "$bottle" --no-update "$controller_win" --query "$exe" "$hook_win")"; then
      print -u2 "Quit $exe and Steam in bottle $bottle, then start through Netal before attaching."
      exit 4
    fi
    pid="$(print -r -- "$queried" | sed -n 's/^mac_pid=//p' | tr -d '\r')"
    [[ "$pid" == <-> && "$pid" -gt 0 ]] || { print -u2 'Cannot identify the selected game'; exit 4; }
    break
  else
    code=$?
    (( code == 2 )) || { print -u2 "Cannot uniquely select $exe in bottle $bottle"; exit "$code"; }
  fi
  sleep 3
done
if [[ -z "$pid" ]]; then print -u2 'No matching game process appeared in the selected bottle'; exit 2; fi
name="$exe"
print "found $name pid=$pid; giving the renderer 20s before injecting"
sleep 20
info="$("$config_tool" info "$pid")"
log="$(print -r -- "$info" | sed -n 's/^log=//p')"
[[ -n "$log" ]] || { print -u2 "No process session log path"; exit 2; }
if ! zsh "$script_dir/inject-re9-metalfx-noop.sh" --arm-exact-sr --sustained \
    "--pid=$pid" "--target-exe=$name" "--pwin-suffix=$suffix" "--bottle=$bottle" "${extra[@]}"; then
  print -u2 "Netal could not start the native bridge; retry from a fresh game process"
  exit 5
fi
ready=0
for _ in {1..45}; do grep -q "event=bridge_ready" "$log" 2>/dev/null && { ready=1; break; }; sleep 2; done
current_info="$("$config_tool" info "$pid")" || { print -u2 'Target process exited during attachment'; exit 5; }
[[ "$current_info" == "$info" ]] || { print -u2 'Target process changed during attachment; start a new attach'; exit 5; }
mapped="$(vmmap "$pid" 2>/dev/null | grep -c libNeuralStudioRe9MetalFXNoop || true)"
if (( ready )) && (( mapped > 0 )); then
  print "bridge=OK pid=$pid ($name); SR frames appear in the log as sr_exact_scheduled once gameplay starts"
  print "log: $log"
else
  print -u2 "bridge=FAILED pid=$pid ready=$ready mapped=$mapped -- our SR is NOT active in this process"
  exit 5
fi
