#!/bin/zsh
# Launch an eligible native game. The read-only --check path runs before any quit or staging.
# --observe never arms a model; --no-sr / --no-fg preserve each user's feature choice.
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
repository="$(cd "$script_dir/.." && pwd)"
app=""; observe=0; stage=1; sr=1; fg=1; check_only=0; want_output=""; steam_id=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --observe) observe=1; shift ;;
    --no-sr) sr=0; shift ;;
    --no-fg) fg=0; shift ;;
    --no-stage) stage=0; shift ;;
    --check) check_only=1; shift ;;
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
(( observe || sr || fg )) || { print -u2 'Choose Super Resolution, Frame Generation, or an observation run.'; exit 64; }

# --observe here checks injection eligibility separately from static MetalFX evidence.
inspection="$(zsh "$script_dir/netal-native-preflight.sh" "$app" --json --observe)" || {
  print -u2 -- "$(python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["summary"]+": "+d["detail"])' <<< "$inspection" 2>/dev/null || print 'Compatibility inspection failed. Check the runtime folder and Python 3 installation.')"
  exit 1
}
inspection_values=("${(@f)$(python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["bundleID"]); print(d["executable"]); print(int(d["sandboxed"])); print(int(d["canLaunchModels"]))' <<< "$inspection")}")
bundle_id="$inspection_values[1]"
executable="$inspection_values[2]"
sandboxed="$inspection_values[3]"
if (( ! observe && ! inspection_values[4] )); then
  print -u2 'No native MetalFX integration has been verified for this game. Use Window Capture, or run an observation check first.'
  exit 1
fi

dylib="$repository/build/netal-native/libNetalNative.dylib"
[[ -f "$dylib" ]] || { print -u2 'The native bridge is missing. Choose a complete runtime folder in Netal Settings.'; exit 1; }
[[ "$(/usr/bin/lipo -archs "$dylib" 2>/dev/null)" == *arm64* ]] || { print -u2 'The native bridge has no Apple silicon version. Rebuild or choose a complete runtime folder.'; exit 1; }
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
if /bin/ps -ww -axo comm= | python3 -c 'import sys; commands=sys.stdin.read().splitlines(); sys.exit(0 if sys.argv[1] in commands else 1)' "$app/Contents/MacOS/$executable"; then
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
env_args=(
  --env "DYLD_INSERT_LIBRARIES=$dylib"
  --env "NEURALSTUDIO_RE9_METALFX_NOOP=1"
  --env "NEURALSTUDIO_RE9_METALFX_LOG=$log"
  --env "NEURALSTUDIO_TARGET_EXE=$executable"
  --env "NEURALSTUDIO_RE9_SR_ARM=$sr_arm"
  --env "NEURALSTUDIO_RE9_FG_ARM=$fg_arm"
)
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
