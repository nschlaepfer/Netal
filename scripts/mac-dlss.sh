#!/bin/zsh
# mac-dlss-v1 entry point: inject (or relaunch and inject) the Apple-native DLSS SR
# bridge into a CrossOver game. Defaults: hardware mode, sustained, RE9 (appid 3764200).
#   scripts/mac-dlss.sh --bottle Steam --appid 3764200 [--exe re9.exe] [--relaunch] [--mode hw|exact]
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
args=(--sustained --mode hw)
bottle="Steam"; appid="3764200"; exe="re9.exe"; gptk=""; fallback=1; relaunch=0; allow_gptk3=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bottle) bottle="$2"; shift 2 ;;
    --appid) appid="$2"; shift 2 ;;
    --exe) exe="$2"; shift 2 ;;
    --relaunch) relaunch=1; shift ;;
    --fg) args+=(--fg); shift ;;   # arm Netal frame generation on the MetalFX interpolator seam
    --gptk) gptk="$2"; relaunch=1; shift 2 ;;
    --allow-gptk3) allow_gptk3=1; shift ;;
    --no-fallback) fallback=0; shift ;;
    --dump=*|--jitter-sign=*|--mv-sign=*|--force-vsync) args+=("$1"); shift ;;
    --mode) args=(${args:#--mode} ); args=(${args:#hw}); args+=(--mode "$2"); shift 2 ;;
    *) print -u2 "unknown argument $1"; exit 1 ;;
  esac
done
# Toolkit selection. GPTK 4 (D3DMetal 4.0b2) is the default: the game renders
# much faster on it (user report 2026-09-01). Under GPTK 4 the game creates its
# MetalFX scaler through the Metal 4 API, which the bridge currently observes and
# passes through (Apple's MetalFX does the upscale) until the Metal 4 encode path
# lands; --gptk 3 gives the Apple-native SR path today. The scaler is created
# lazily when gameplay starts, so its absence is never a failure signal.
# Fallback to GPTK 3 (unless --no-fallback) only on hard failures: the game
# process dies within the settle window, the bridge never reports ready, or the
# bridge disarms on a GPU error.
log="$script_dir/../build/re9_crossover/re9-live-metalfx.log"
fallback_log="$script_dir/../build/re9_crossover/gptk-fallback.log"
[[ -z "$gptk" ]] && gptk=4
# GPTK 4 is the supported route. GPTK 3 renders the game far slower and is kept only as a
# debugging escape hatch behind --allow-gptk3; nothing selects it automatically any more.
if [[ "$gptk" != "4" ]] && (( ! allow_gptk3 )); then
  print -u2 "GPTK $gptk is not a supported route (GPTK 4 only). Pass --allow-gptk3 to override for debugging."
  exit 1
fi
current="$(zsh "$script_dir/gptk-select.sh" status | sed -n 's/^active=\([^ ]*\).*/\1/p')"
want="3.0"; [[ "$gptk" == "4" ]] && want="4.0b2"
if [[ "$current" != "$want" ]]; then zsh "$script_dir/gptk-select.sh" "$gptk" --bottle "$bottle"; relaunch=1; fi
# Reap stranded GPTK wine trees before launching. Every toolkit swap used to leave a full prefix
# server behind (services.exe, winedevice.exe, plugplay.exe, svchost.exe, rpcss.exe reparented to
# launchd); 18 of them accumulated over one day of testing, burning ~12% CPU between them. Only
# runs when no game .exe is live under GPTK, so a running session is never touched.
if ! pgrep -f 'Game Porting Toolkit.*\.exe' >/dev/null 2>&1; then
  # `|| true`: pgrep exits 1 when nothing matches, and under `set -e -o pipefail` that aborts
  # the launcher before it ever starts the game.
  stranded=$(pgrep -f 'Game Porting Toolkit.app' 2>/dev/null | wc -l | tr -d ' ' || true)
  if (( stranded > 0 )); then
    print "reaping $stranded stranded GPTK wine process(es) from earlier launches"
    pgrep -f 'Game Porting Toolkit.app' 2>/dev/null | while read -r stale; do kill -TERM "$stale" 2>/dev/null; done
    sleep 2
    pgrep -f 'Game Porting Toolkit.app' 2>/dev/null | while read -r stale; do kill -KILL "$stale" 2>/dev/null; done
  fi
fi
run_args=("${args[@]}"); (( relaunch )) && run_args+=(--relaunch)
zsh "$script_dir/crossover-exact-sr.sh" --bottle "$bottle" --appid "$appid" --exe "$exe" "${run_args[@]}"
hard_failure() {
  pgrep -f "requiem.*${exe//./\\.}" >/dev/null || { print "game process not running"; return 0; }
  grep -q 'event=bridge_ready' "$log" 2>/dev/null || { print "bridge never reported ready"; return 0; }
  grep -q 'sr_disarmed reason=gpu_error' "$log" 2>/dev/null && { print "bridge disarmed on a GPU error"; return 0; }
  return 1
}
if (( fallback )) && [[ "$want" == "4.0b2" ]]; then
  sleep 45
  if reason="$(hard_failure)"; then
    # Report it; do not swap the toolkit. Dropping to GPTK 3 traded a visible failure for a
    # much slower game, and it silently moved the session onto an unsupported route.
    print "$(date '+%F %T') gptk4 hard failure ($reason) -- staying on GPTK 4; rerun with --allow-gptk3 to test the old route" | tee -a "$fallback_log"
  else
    print "gptk4: healthy (the MetalFX scaler appears when gameplay starts; Metal 4 scalers are served by the SR bridge: look for sr_exact_scheduled api=metal4)"
  fi
fi
