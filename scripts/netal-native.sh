#!/bin/zsh
# Launch an eligible native game. The read-only --check path runs before any quit or staging.
# --observe never arms a model; --no-sr / --no-fg preserve each user's feature choice.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
script_dir="$(cd "$(dirname "$0")" && pwd)"
repository="$(cd "$script_dir/.." && pwd)"
app=""; observe=0; stage=1; sr=1; fg=1; check_only=0; want_output=""; steam_id=""
nr_exp=0; nr_bridge_dir=""; nr_model=""; nr_scale=""; nr_interval_ms=""; nr_async=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --observe) observe=1; shift ;;
    --no-sr) sr=0; shift ;;
    --no-fg) fg=0; shift ;;
    --no-stage) stage=0; shift ;;
    --check) check_only=1; shift ;;
    --nr-experimental) nr_exp=1; shift ;;
    --nr-inline) nr_async=0; shift ;;
    --nr-bridge-dir|--nr-model|--nr-scale|--nr-worker-interval-ms)
      [[ $# -ge 2 && -n "$2" ]] || { print -u2 "Missing value for $1"; exit 64; }
      case "$1" in
        --nr-bridge-dir) nr_bridge_dir="$2" ;;
        --nr-model) nr_model="$2" ;;
        --nr-scale) nr_scale="$2" ;;
        --nr-worker-interval-ms) nr_interval_ms="$2" ;;
      esac
      shift 2 ;;
    --output|--steam-app-id)
      [[ $# -ge 2 && -n "$2" ]] || { print -u2 "Missing value for $1"; exit 64; }
      if [[ "$1" == --output ]]; then want_output="$2"; else steam_id="$2"; fi
      shift 2 ;;
    -*) print -u2 "Unknown option: $1"; exit 64 ;;
    *) [[ -z "$app" ]] || { print -u2 'Choose one game at a time.'; exit 64; }; app="$1"; shift ;;
  esac
done
[[ -n "$app" ]] || { print -u2 "Usage: $0 Game.app [--observe] [--no-sr] [--no-fg] [--check]"; exit 64; }
[[ -z "$want_output" || "$want_output" == <->x<-> ]] || { print -u2 'Output must be WIDTHxHEIGHT.'; exit 64; }
[[ -z "$steam_id" || "$steam_id" == <-> ]] || { print -u2 'Steam app ID must contain digits only.'; exit 64; }
(( observe || sr || fg || nr_exp )) || { print -u2 'Choose Super Resolution, Frame Generation, experimental NR, or an observation run.'; exit 64; }
if (( nr_exp )); then
  [[ -n "$nr_bridge_dir" && -n "$nr_model" ]] || { print -u2 -- '--nr-experimental requires --nr-bridge-dir DIR and --nr-model PATH.'; exit 64; }
fi

# --observe here checks injection eligibility separately from static MetalFX evidence.
inspection="$(zsh "$script_dir/netal-native-preflight.sh" "$app" --json --observe)" || {
  print -u2 -- "$(python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["summary"]+": "+d["detail"])' <<< "$inspection" 2>/dev/null || print 'Compatibility inspection failed. Check the runtime folder and Python 3 installation.')"
  exit 1
}
inspection_values=("${(@f)$(python3 -c 'import json,sys; d=json.load(sys.stdin); exes=d.get("targetExecutables") or [d["executable"]]; print(d["bundleID"]); print(d["executable"]); print(int(d["sandboxed"])); print(int(d["canLaunchModels"])); print(",".join(exes))' <<< "$inspection")}")
bundle_id="$inspection_values[1]"
executable="$inspection_values[2]"
sandboxed="$inspection_values[3]"
target_executables_csv="$inspection_values[5]"
if (( ! observe && (sr || fg) && ! inspection_values[4] )); then
  print -u2 'No native MetalFX integration has been verified for this game. Use Window Capture, or run an observation check first.'
  exit 1
fi

dylib="$repository/build/netal-native/libNetalNative.dylib"
if (( observe || sr || fg )); then
  [[ -f "$dylib" ]] || { print -u2 'The native bridge is missing. Choose a complete runtime folder in Netal Settings.'; exit 1; }
  [[ "$(/usr/bin/lipo -archs "$dylib" 2>/dev/null)" == *arm64* ]] || { print -u2 'The native bridge has no Apple silicon version. Rebuild or choose a complete runtime folder.'; exit 1; }
fi
if (( nr_exp )); then
  nr_dylib="$nr_bridge_dir/libNetalNRFastBridge.dylib"
  [[ -f "$nr_dylib" && -f "$nr_bridge_dir/nr_fast_kernels.metallib" && -f "$nr_bridge_dir/netal_nr_optical_flow.metallib" ]] || {
    print -u2 'Experimental NR bridge artifacts are incomplete. Build with scripts/build-netal-nr-fast-bridge.py first.'
    exit 1
  }
  [[ "$(/usr/bin/lipo -archs "$nr_dylib" 2>/dev/null)" == *arm64* ]] || { print -u2 'The experimental NR bridge has no Apple silicon slice.'; exit 1; }
  [[ -f "$nr_model" || -d "$nr_model" ]] || { print -u2 'Experimental NR model path was not found.'; exit 1; }
fi
container="$HOME/Library/Containers/$bundle_id/Data"
if (( sandboxed )); then
  [[ -d "$container" ]] || { print -u2 'Open this App Store game normally once, then quit it and try Netal again.'; exit 1; }
  netal_dir="$container/netal"
else
  netal_dir="$HOME/Library/Application Support/Netal/NativeGames/$bundle_id"
fi
source_sr="$repository/work/re9-sr-runtime-bundle"
source_fg="$repository/work/re9-fg-runtime-bundle"
# A skipped-stage run validates the staged folders it will actually use.
sr_bundle="$source_sr"; fg_bundle="$source_fg"
if (( sandboxed )); then
  sr_bundle="$netal_dir/re9-sr-runtime-bundle"; fg_bundle="$netal_dir/re9-fg-runtime-bundle"
fi
validate_bundle() {
  local selected="$1" label="$2"
  python3 - "$selected" "$label" <<'VERIFY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
label = sys.argv[2]
try:
    manifest = json.loads((root / 'manifest.json').read_text())
    entries = manifest['files']
    if isinstance(entries, dict):
        entries = [dict(value, path=key) for key, value in entries.items()]
    if not isinstance(entries, list) or not entries:
        raise ValueError('manifest has no runtime files')
    for item in entries:
        path = root / item['path']
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError('a required runtime file is missing')
        if not isinstance(item.get('bytes'), int) or path.stat().st_size != item['bytes']:
            raise ValueError('a required runtime file is incomplete')
except (OSError, ValueError, TypeError, KeyError) as error:
    print(f'{label} runtime is not ready: {error}. Choose a complete runtime folder in Settings.', file=sys.stderr)
    sys.exit(1)
VERIFY
}
if (( ! observe )); then
  verify_sr="$sr_bundle"; verify_fg="$fg_bundle"
  if (( stage && sandboxed )); then verify_sr="$source_sr"; verify_fg="$source_fg"; fi
  (( ! sr )) || validate_bundle "$verify_sr" 'Super Resolution'
  (( ! fg )) || validate_bundle "$verify_fg" 'Frame Generation'
fi
if (( check_only )); then
  print 'result=pass Native launch requirements are ready. Gameplay remains unverified.'
  exit 0
fi
# Refuse before touching session logs or staging. Match the process command as a literal path,
# rather than using a regex made from a path that can contain brackets or parentheses.
target_paths=()
for exe in "${(@s:,:)target_executables_csv}"; do
  [[ -n "$exe" ]] && target_paths+=("$app/Contents/MacOS/$exe")
done
if /bin/ps -ww -axo comm= | python3 -c 'import sys; from pathlib import Path; commands={c for raw in sys.stdin.read().splitlines() if raw for c in (raw, str(Path(raw).resolve())) if c}; targets={t for arg in sys.argv[1:] for t in (arg, str(Path(arg).resolve()))}; sys.exit(0 if commands & targets else 1)' "${target_paths[@]}"; then
  print -u2 'The game is still running. Save and quit it, then start it through Netal.'
  exit 1
fi

ensure_pipelines() {
  local geometry="$1"
  [[ "$geometry" == <->x<-> ]] || return 0
  local bundle_root="$source_sr"
  # A resolution the bundle ships a real profile for already carries its own batched pipelines, and
  # nothing needs synthesizing - 3840x2160 is one of seven such profiles. Checking only the
  # synthesized directory made those look missing and sent the generator off to fail on them.
  local shipped="$bundle_root/profiles/$geometry/pipeline"
  if [[ -d "$shipped" && -n "$(print -rn -- "$shipped"/*.json(N))" ]]; then
    print "pipelines      $geometry shipped with the bundle"
    return 0
  fi
  local target="$bundle_root/pipeline-synth/$geometry"
  if [[ -d "$target" && -n "$(print -rn -- "$target"/*.json(N))" ]]; then
    print "pipelines      $geometry present"
    return 0
  fi
  print "pipelines      $geometry missing - generating (one time, ~2s)"
  zsh "$script_dir/netal-make-pipelines.sh" "$geometry" ||     print -u2 "  generation failed; the transpiled kernels will run instead (slower, still correct)"
}
# A game's resolution menu is the display's mode list, so generate a set for every mode up front
# instead of making the first run at each one slow. On this 16-inch MacBook Pro that is ten modes -
# the six the game offers plus the notch-inclusive variants - at about two seconds and 256K each,
# so the whole sweep costs under twenty seconds once and nothing on every later launch.
if (( sr && ! observe )); then
for mode in ${(f)"$(python3 "$repository/tools/netal-display-modes.py" 2>/dev/null || true)"}; do
  ensure_pipelines "$mode"
done

if [[ -n "$want_output" ]]; then
  ensure_pipelines "$want_output"
elif [[ -f "$netal_dir/netal-live.log" ]]; then
  # scaler_create as well as sr_profile_selected, and this is the point of it: the session logs its
  # selected profile only once it has finished arming, so a run whose pipelines were missing - the
  # exact run worth learning from - spends its life compiling and teaches the next launch nothing.
  # The scaler carries the output size from its first frame, seconds in, whether or not SR ever ran.
  # Restricted to those two events on purpose: an encode line carries output=<pointer>, and a
  # pointer's leading hex digits parse as a resolution if you let them.
  for seen in ${(f)"$(grep -E 'event=(sr_profile_selected|scaler_create) ' "$netal_dir/netal-live.log" 2>/dev/null | grep -oE ' output=[0-9]+x[0-9]+' | grep -oE '[0-9]+x[0-9]+' | sort -u)"}; do
    ensure_pipelines "$seen"
  done
fi

fi

# Sandboxed games need a private runtime copy. Steam/native games can read the prepared runtime
# directly and keep only their log/controls in Netal's own application-support folder.
mkdir -p "$netal_dir"
if (( sandboxed && stage && ! observe )); then
  if (( sr )); then
    print 'Preparing Super Resolution…'
    rsync -a --delete "$source_sr/" "$sr_bundle/"
  fi
  if (( fg )); then
    print 'Preparing Frame Generation…'
    rsync -a --delete "$source_fg/" "$fg_bundle/"
  fi
fi
log="$netal_dir/netal-live.log"
: > "$log"
rm -f "$netal_dir/re9-sr-pause" "$netal_dir/re9-sr-mark"
# Honor the launch checkbox even if a previous session paused FG; keep the user's SR arithmetic.
python3 - "$netal_dir/netal-control" "$(( fg && ! observe ))" <<'CONTROL'
import os, sys
from pathlib import Path
path = Path(sys.argv[1])
lines = path.read_text().splitlines() if path.exists() else []
lines = [line for line in lines if not line.startswith('fg=')]
temporary = path.with_suffix('.launching')
temporary.write_text('\n'.join(lines + ['fg=' + sys.argv[2]]) + '\n')
os.replace(temporary, path)
CONTROL
sr_arm=$(( sr && ! observe )); fg_arm=$(( fg && ! observe ))
dyld_insert="$dylib"
if (( nr_exp && ! observe )); then
  if (( observe || sr || fg )); then
    dyld_insert="$dylib:$nr_bridge_dir/libNetalNRFastBridge.dylib"
  else
    dyld_insert="$nr_bridge_dir/libNetalNRFastBridge.dylib"
  fi
fi
env_args=(
  --env "DYLD_INSERT_LIBRARIES=$dyld_insert"
  --env "NEURALSTUDIO_RE9_METALFX_NOOP=1"
  --env "NEURALSTUDIO_RE9_METALFX_LOG=$log"
  --env "NEURALSTUDIO_TARGET_EXE=$target_executables_csv"
  --env "NEURALSTUDIO_RE9_SR_ARM=$sr_arm"
  --env "NEURALSTUDIO_RE9_FG_ARM=$fg_arm"
)
if (( nr_exp && ! observe )); then
  env_args+=(
    --env "NETAL_NR_ADAPTER_MODE=continuous"
    --env "NETAL_NR_MODEL=$nr_model"
    --env "NETAL_NR_METALLIB=$nr_bridge_dir/nr_fast_kernels.metallib"
    --env "NETAL_NR_FLOW_METALLIB=$nr_bridge_dir/netal_nr_optical_flow.metallib"
    --env "NETAL_NR_LOG=$netal_dir/netal-nr-live.log"
    --env "NETAL_NR_HOOK_SPATIAL=1"
    --env "NETAL_NR_HOOK_DRAWABLE=1"
    --env "NETAL_NR_ASYNC=$nr_async"
  )
  [[ -z "$nr_scale" ]] || env_args+=(--env "NETAL_NR_SCALE=$nr_scale")
  [[ -z "$nr_interval_ms" ]] || env_args+=(--env "NETAL_NR_WORKER_INTERVAL_MS=$nr_interval_ms")
fi
if (( ! observe && sr )); then
  env_args+=(
    --env "NEURALSTUDIO_RE9_SR_BUNDLE=$sr_bundle"
    --env "NEURALSTUDIO_RE9_SR_SUSTAINED=1"
    --env "NEURALSTUDIO_SR_PWIN_SUFFIX=${PWIN_SUFFIX:-_hw7}"
    --env "NEURALSTUDIO_SR_ALLOW_ANY_MOTION_SCALE=1"
    --env "NEURALSTUDIO_SR_ALLOW_ANY_JITTER=1"
    --env "NEURALSTUDIO_SR_PWIN_PIPELINE=1"
    --env "NEURALSTUDIO_SR_DEPTH_INPUT_LIBRARY=hiluma_input_depthinv"
  )
fi
if (( ! observe && fg )); then
  env_args+=(--env "NEURALSTUDIO_RE9_FG_BUNDLE=$fg_bundle")
fi
if [[ -n "$steam_id" ]]; then
  env_args+=(--env "SteamAppId=$steam_id" --env "SteamGameId=$steam_id")
fi
# Launch Services preserves App Store receipt behavior. Exit zero means a launch was requested,
# never that the bridge loaded or that a model has served a frame.
/usr/bin/open --arch arm64 -a "$app" "${env_args[@]}"
defaults write com.nschlaepfer.netal.menubar NetalLogPath "$log" 2>/dev/null || true
print "Launch requested. Waiting for the game to report frames."
print "log=$log"
