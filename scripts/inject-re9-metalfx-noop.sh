#!/bin/zsh
# Use prebuilt, relocatable helpers with a private per-process launch configuration.
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
source "$script_dir/netal-wine-output.sh"
repository="$(cd "$script_dir/.." && pwd)"
build_dir="$repository/build/re9_crossover"
config_tool="$repository/build/netal-tools/netal-launch-config"
cx_root="${NETAL_CROSSOVER_ROOT:-/Applications/CrossOver.app/Contents/SharedSupport/CrossOver}"
bottle="${NEURALSTUDIO_BOTTLE:-Steam}"
pid=""; target_exe="re9.exe"
arm_sr=0; arm_fg=0; sustained=0; suffix=""; commit_hook=""; dump_spec=""; jitter_sign=""; mv_sign=""; vsync=0
for argument in "$@"; do
  case "$argument" in
    --pid=*) pid="${argument#*=}" ;;
    --arm-exact-sr) arm_sr=1 ;;
    --sustained) sustained=1 ;;
    --arm-fg) arm_fg=1 ;;
    --pwin-suffix=*) suffix="${argument#*=}" ;;
    --process=*) print -u2 'Process patterns are no longer supported; select the executable and bottle explicitly.'; exit 64 ;;
    --target-exe=*) target_exe="${argument#*=}" ;;
    --bottle=*) bottle="${argument#*=}" ;;
    --commit-hook=*) commit_hook="${argument#*=}" ;;
    --dump=*) dump_spec="${argument#*=}" ;;
    --jitter-sign=*) jitter_sign="${argument#*=}" ;;
    --mv-sign=*) mv_sign="${argument#*=}" ;;
    --force-vsync) vsync=1 ;;
    --portable) ;; # retained for callers; attachment is always portable
    *) print -u2 "unknown injector argument: $argument"; exit 64 ;;
  esac
done
[[ "$pid" == <-> ]] || { print -u2 -- '--pid=<macOS game PID> is required'; exit 64; }
(( ! sustained || arm_sr )) || { print -u2 -- '--sustained requires --arm-exact-sr'; exit 64; }
for file in "$config_tool" "$build_dir/neuralstudio_re9_inject.exe" "$build_dir/netal_wine_host_v1.dll" "$build_dir/netal_wine_host_v1.so" "$build_dir/libNeuralStudioRe9MetalFXNoop.dylib"; do
  [[ -f "$file" ]] || { print -u2 "Runtime helper missing: $file. Build the runtime before launching."; exit 2; }
done
kill -0 "$pid" 2>/dev/null || { print -u2 'Target process has exited'; exit 2; }
# Selection is performed inside the chosen Wine bottle.
# Only load from Netal's own directory. Never install hooks into CrossOver or a bottle.
hook_dll="$build_dir/netal_wine_host_v1.dll"
bottle_root="$(netal_wine_capture "$cx_root/bin/wine" --bottle "$bottle" --no-update --ux-app /usr/bin/printenv WINEPREFIX)"
[[ "$bottle_root" == /* && "$bottle_root" != *$'\n'* ]] || { print -u2 'Cannot locate the selected bottle'; exit 2; }
for existing in "$cx_root/lib/wine/x86_64-windows/netal_wine_host_v1.dll" "$cx_root/lib/wine/x86_64-unix/netal_wine_host_v1.so" "$bottle_root/drive_c/windows/system32/netal_wine_host_v1.dll"; do
  [[ ! -e "$existing" && ! -L "$existing" ]] || { print -u2 "Refusing existing hook: $existing"; exit 3; }
done
hook_argument=("Z:${hook_dll//\//\\}")
controller="$build_dir/neuralstudio_re9_inject.exe"
controller_win="Z:${controller//\//\\}"
queried="$(netal_wine_capture "$cx_root/bin/wine" --bottle "$bottle" --no-update "$controller_win" --query "$target_exe" "${hook_argument[1]}")"
selected_pid="$(print -r -- "$queried" | sed -n 's/^mac_pid=//p' | tr -d '\r')"
[[ "$selected_pid" == "$pid" ]] || { print -u2 'The selected bottle process does not match the requested Mac PID; retry from a fresh launch'; exit 4; }
info="$("$config_tool" info "$pid")"
log="$(print -r -- "$info" | sed -n 's/^log=//p')"
session_dir="$(print -r -- "$info" | sed -n 's/^session_dir=//p')"
[[ -n "$log" && -n "$session_dir" ]] || { print -u2 'Runtime session location unavailable'; exit 2; }
pipeline=0; [[ "$suffix" == _hw* ]] && pipeline=1
"$config_tool" prepare "$pid" --bridge "$build_dir/libNeuralStudioRe9MetalFXNoop.dylib" \
  --sr_bundle "$repository/work/re9-sr-runtime-bundle" --fg_bundle "$repository/work/re9-fg-runtime-bundle" \
  --target_exe "$target_exe" --arm_sr "$arm_sr" --arm_fg "$arm_fg" --sustained "$sustained" \
  --pwin_suffix "$suffix" --pipeline "$pipeline" --commit_hook "$commit_hook" --dump_spec "$dump_spec" \
  --jitter_sign "$jitter_sign" --mv_sign "$mv_sign" --force_vsync "$vsync"
unix_log="$session_dir/re9-live-unixlib.log"
: > "$log"; : > "$unix_log"
defaults write com.nschlaepfer.netal.menubar NetalLogPath "$log" 2>/dev/null || true

netal_wine_capture "$cx_root/bin/cxstart" --bottle "$bottle" --no-update --wait "$controller_win" "$target_exe" "${hook_argument[@]}"
for attempt in {1..40}; do
  if grep -q '^event=bridge_loaded ' "$unix_log" && grep -q '^event=bridge_ready ' "$log"; then
    print "unix_log=$unix_log"; print "metalfx_log=$log"; print 'result=pass'; exit 0
  fi
  sleep 0.25
done
print -u2 'Native bridge did not report ready; inspect the per-process session logs'
print -u2 'Start the game through Netal with the helper search path; an existing Steam process may retain its old environment.'
exit 4
