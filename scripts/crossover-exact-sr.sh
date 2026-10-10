#!/bin/zsh
# Portable SR/FG launch for a game in the selected CrossOver bottle.
#
#   scripts/crossover-exact-sr.sh --bottle Steam --appid 3764200 [--exe re9.exe] [--sustained] [--relaunch] [--mode exact|hw]
#
# Launches the game through the bottle's Steam (or attaches to a running one),
# prepares the helper search path, queries the selected game through Wine,
# starts the Netal bridge and prints the live receipt path. A rebuilt bridge needs a
# fresh game process; --relaunch kills and restarts it.
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
source "$script_dir/netal-wine-output.sh"
repository="$(cd "$script_dir/.." && pwd)"
cx_root="${NETAL_CROSSOVER_ROOT:-/Applications/CrossOver.app/Contents/SharedSupport/CrossOver}"
config_tool="$repository/build/netal-tools/netal-launch-config"
bottle="Steam"; appid=""; exe="re9.exe"; sustained=false; relaunch=false; fg=false; settle=75; mode="exact"; extra=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bottle) bottle="$2"; shift 2 ;;
    --appid) appid="$2"; shift 2 ;;
    --exe) exe="$2"; shift 2 ;;
    --sustained) sustained=true; shift ;;
    --fg) fg=true; shift ;;
    --mode) mode="$2"; shift 2 ;;
    --dump=*|--jitter-sign=*|--mv-sign=*|--force-vsync) extra+=("$1"); shift ;;
    --relaunch) relaunch=true; shift ;;
    --settle) settle="$2"; shift 2 ;;
    *) print -u2 "unknown argument: $1"; exit 1 ;;
  esac
done
[[ -n "$appid" ]] || { print -u2 -- "--appid <steam app id> is required"; exit 1; }
[[ -x "$config_tool" ]] || { print -u2 "Runtime helpers are not built. Run scripts/build-macos-re9-wine-injector.sh before launching."; exit 2; }
for required in neuralstudio_re9_inject.exe netal_wine_host_v1.dll netal_wine_host_v1.so libNeuralStudioRe9MetalFXNoop.dylib; do
  [[ -f "$repository/build/re9_crossover/$required" ]] || { print -u2 "Runtime helper missing: $required"; exit 2; }
done
[[ -f "$repository/work/re9-sr-runtime-bundle/manifest.json" ]] || { print -u2 "SR runtime bundle is missing"; exit 2; }
if $fg; then [[ -f "$repository/work/re9-fg-runtime-bundle/manifest.json" ]] || { print -u2 "FG runtime bundle is missing"; exit 2; }; fi
# Resolve a target through the selected Wine server, not a global macOS name search.
controller="$repository/build/re9_crossover/neuralstudio_re9_inject.exe"
hook="$repository/build/re9_crossover/netal_wine_host_v1.dll"
controller_win="Z:${controller//\//\\}"
hook_win="Z:${hook//\//\\}"
[[ "$mode" == exact || "$mode" == hw ]] || { print -u2 'Mode must be exact or hw'; exit 1; }
[[ "$settle" == <-> ]] || { print -u2 'Settle time must be a nonnegative integer'; exit 1; }
# The wrapper returns the selected bottle's normal paths, including custom configuration.
base_path="$(netal_wine_capture "$cx_root/bin/wine" --bottle "$bottle" --no-update --ux-app /usr/bin/printenv WINEDLLPATH)"
[[ -n "$base_path" && "$base_path" != *$'\n'* && "$repository" != *:* ]] || { print -u2 'Cannot prepare the runtime search path'; exit 2; }
bottle_root="$(netal_wine_capture "$cx_root/bin/wine" --bottle "$bottle" --no-update --ux-app /usr/bin/printenv WINEPREFIX)"
[[ "$bottle_root" == /* && "$bottle_root" != *$'\n'* ]] || { print -u2 'Cannot locate the selected bottle'; exit 2; }
for existing in "$cx_root/lib/wine/x86_64-windows/netal_wine_host_v1.dll" "$cx_root/lib/wine/x86_64-unix/netal_wine_host_v1.so" "$bottle_root/drive_c/windows/system32/netal_wine_host_v1.dll"; do
  [[ ! -e "$existing" && ! -L "$existing" ]] || { print -u2 "An older shared Netal hook conflicts with portable startup: $existing"; exit 3; }
done
helper_path="$repository/build/re9_crossover:$base_path"
launch_environment="WINEDLLPATH=${(q)helper_path}"

find_windows() {
  local output code
  if output="$(netal_wine_capture "$cx_root/bin/wine" --bottle "$bottle" --no-update "$controller_win" --find "$1")"; then
    print -r -- "$output"
  else
    code=$?
    if (( code == 2 )); then return 0; fi
    print -u2 "Cannot uniquely identify $1 in bottle $bottle. Close duplicate instances and retry."
    return "$code"
  fi
}
query_mac_pid() {
  local output selected_pid
  if ! output="$(netal_wine_capture "$cx_root/bin/wine" --bottle "$bottle" --no-update "$controller_win" --query "$1" "$hook_win")"; then
    if [[ "$1" == steam.exe ]]; then
      print -u2 "Quit Steam in bottle $bottle, then launch through Netal so it receives the runtime settings."
    else
      print -u2 "Quit $1 and Steam in bottle $bottle, then launch through Netal so they receive its runtime settings."
    fi
    return 4
  fi
  selected_pid="$(print -r -- "$output" | sed -n 's/^mac_pid=//p' | tr -d '\r')"
  [[ "$selected_pid" == <-> && "$selected_pid" -gt 0 ]] || { print -u2 'The runtime could not identify the selected process'; return 4; }
  print -r -- "$selected_pid"
}
game_pid() {
  local found
  found="$(find_windows "$exe")" || return $?
  [[ -n "$found" ]] || return 0
  query_mac_pid "$exe"
}
# Check existing Steam before terminating any game: a running client keeps its old environment.
steam_found="$(find_windows steam.exe)"
if [[ -n "$steam_found" ]]; then query_mac_pid steam.exe >/dev/null; fi
pid="$(game_pid)"
if [[ -n "$pid" ]] && $relaunch; then
  identity="$("$config_tool" info "$pid")"
  [[ "$("$config_tool" info "$pid")" == "$identity" ]] || { print -u2 'Game changed before relaunch; retry'; exit 4; }
  print "relaunch: terminating $exe pid=$pid in bottle $bottle"
  kill "$pid" || true
  for _ in {1..30}; do [[ -n "$(find_windows "$exe")" ]] || break; sleep 2; done
  [[ -z "$(find_windows "$exe")" ]] || { print -u2 "$exe is still running; save your progress and quit it before relaunching"; exit 4; }
  pid=""
fi
if [[ -n "$pid" ]] && lsof -p "$pid" 2>/dev/null | grep -q libNeuralStudioRe9MetalFXNoop; then
  print -u2 "bridge already mapped in pid $pid; use --relaunch for a rebuilt bridge"
  exit 2
fi
if [[ -z "$pid" ]]; then
  print "launching steam app $appid in bottle $bottle"
  "$cx_root/bin/cxstart" --bottle "$bottle" --no-update --env "$launch_environment" 'C:\Program Files (x86)\Steam\steam.exe' -applaunch "$appid" >/dev/null 2>&1 &
  for _ in {1..80}; do pid="$(game_pid)"; [[ -n "$pid" ]] && break; sleep 3; done
  [[ -n "$pid" ]] || { print -u2 "$exe did not start"; exit 3; }
  print "$exe pid=$pid; settling ${settle}s for the renderer"
  sleep "$settle"
fi
args=(--arm-exact-sr --portable)
${fg:-false} && args+=(--arm-fg)
$sustained && args+=(--sustained)
args+=("${extra[@]}")
case "$mode" in
  exact) ;;
  hw) args+=(--pwin-suffix=${PWIN_SUFFIX:-_hw7}) ;;   # Fast mode; see the documented numerical limits.
  *) print -u2 "unknown --mode $mode (exact|hw)"; exit 1 ;;
esac
info="$("$config_tool" info "$pid")"
log="$(print -r -- "$info" | sed -n 's/^log=//p')"
[[ -n "$log" ]] || { print -u2 "Cannot resolve this process's runtime log"; exit 2; }
# Point the injector at whichever game this launch targets; without these it silently
# defaults to RE9 and reports target_not_found for anything else.
args+=(--pid="$pid" --target-exe="$exe" --bottle="$bottle")
[[ -n "${NEURALSTUDIO_SR_METAL4_COMMIT_HOOK:-}" ]] && args+=(--commit-hook="$NEURALSTUDIO_SR_METAL4_COMMIT_HOOK")
if ! zsh "$script_dir/inject-re9-metalfx-noop.sh" "${args[@]}"; then
  print -u2 "Netal could not start the native bridge; inspect the session log and retry from a fresh game process"
  exit 5
fi
# Verify the same process lifetime; a restarted game must not reuse an old ready log.
ready=0
for _ in {1..45}; do grep -q "event=bridge_ready" "$log" 2>/dev/null && { ready=1; break; }; sleep 2; done
current_info="$("$config_tool" info "$pid")" || { print -u2 'Target process exited during attachment'; exit 5; }
[[ "$current_info" == "$info" ]] || { print -u2 'Target process changed during attachment; start a new launch'; exit 5; }
mapped="$(vmmap "$pid" 2>/dev/null | grep -c libNeuralStudioRe9MetalFXNoop || true)"
if [[ $ready -eq 1 && "$mapped" -gt 0 ]]; then
  print "bridge=OK pid=$pid (bridge_ready logged, dylib mapped x$mapped)"
else
  print -u2 "bridge=FAILED pid=$pid ready=$ready mapped=$mapped -- our SR is NOT active in this process"
  exit 5
fi
print "live log: $log"
print -r -- "summary: python3 ${(q)script_dir}/summarize-re9-live-sustained-sr.py --log ${(q)log}"
