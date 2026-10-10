#!/bin/zsh
# Select the D3DMetal (Game Porting Toolkit) version CrossOver uses.
#   scripts/gptk-select.sh status
#   scripts/gptk-select.sh 3|4 [--bottle Steam]     # swaps the framework, stopping the bottle first
# Versions live in ~/Library/Application Support/NeuralStudio/gptk/{3.0,4.0b2}/external/
# (3.0 = CrossOver 26.3's bundled D3DMetal, 4.0b2 = GPTK 4 evaluation environment).
set -euo pipefail
store="$HOME/Library/Application Support/NeuralStudio/gptk"
cx="/Applications/CrossOver.app/Contents/SharedSupport/CrossOver"
external="$cx/lib64/apple_gptk/external"
bottle="Steam"
request="${1:-status}"; shift || true
while [[ $# -gt 0 ]]; do case "$1" in --bottle) bottle="$2"; shift 2 ;; *) print -u2 "unknown argument $1"; exit 1 ;; esac; done
installed_version() { defaults read "$external/D3DMetal.framework/Versions/A/Resources/Info.plist" CFBundleShortVersionString 2>/dev/null || print unknown; }
case "$request" in
  status) print "active=$(installed_version) store=$store"; ls "$store" | tr '\n' ' '; print; exit 0 ;;
  3) version="3.0" ;;
  4) version="4.0b2" ;;
  *) print -u2 "usage: gptk-select.sh status|3|4 [--bottle NAME]"; exit 1 ;;
esac
[[ -d "$store/$version/external/D3DMetal.framework" ]] || { print -u2 "missing $store/$version"; exit 1; }
if [[ "$(installed_version)" == "$version" ]]; then print "active=$version (unchanged)"; exit 0; fi
# Stop everything in the bottle so the new framework loads cleanly.
pkill -f 'steamwebhelper\.exe' 2>/dev/null || true
pkill -f 'requiem.*re9\.exe' 2>/dev/null || true
pkill -f 'steam\.exe' 2>/dev/null || true
sleep 3
WINEPREFIX="$HOME/Library/Application Support/CrossOver/Bottles/$bottle" "$cx/bin/wineserver" -k 2>/dev/null || true
sleep 3
rm -rf "$external/D3DMetal.framework"
ditto "$store/$version/external/D3DMetal.framework" "$external/D3DMetal.framework"
cp -p "$store/$version/external/libd3dshared.dylib" "$external/libd3dshared.dylib"
xattr -dr com.apple.quarantine "$external" 2>/dev/null || true
print "$version" > "$store/active"
print "active=$(installed_version) (swapped, bottle stopped)"
