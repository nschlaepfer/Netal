#!/bin/zsh
# Package an already-built app. Signing/notarization remain explicit release steps.
# Usage: NETAL_CODESIGN_IDENTITY=<identity> scripts/package-netal-dmg.sh <Netal.app> <new.dmg>
set -euo pipefail
[[ $# == 2 ]] || { print -u2 'Usage: package-netal-dmg.sh <Netal.app> <new.dmg>'; exit 1; }
repository="$(cd "$(dirname "$0")/.." && pwd)"
app="${1:A}"
output="${2:A}"
[[ -d "$app" && -f "$app/Contents/Info.plist" ]] || { print -u2 'Missing app bundle'; exit 1; }
[[ ! -e "$output" && "$output" == *.dmg ]] || { print -u2 'Use a new .dmg output path'; exit 1; }
[[ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$app/Contents/Info.plist")" == com.nschlaepfer.netal.menubar ]] || { print -u2 'Unexpected app identity'; exit 1; }
version="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$app/Contents/Info.plist")"
[[ "$version" =~ '^[0-9]+\.[0-9]+\.[0-9]+$' ]] || { print -u2 'Invalid app version'; exit 1; }
codesign --verify --deep --strict "$app"
stage="$(mktemp -d "${TMPDIR:-/tmp}/netal-dmg.XXXXXX")"
trap 'rm -rf "$stage"' EXIT
ditto --noextattr --norsrc "$app" "$stage/Netal.app"
ln -s /Applications "$stage/Applications"
cp "$repository/LICENSE" "$stage/LICENSE.txt"
cp "$repository/THIRD_PARTY_NOTICES.md" "$stage/THIRD_PARTY_NOTICES.txt"
cat > "$stage/Read Me.txt" <<TEXT
NETAL $version — MENU-BAR APP PREVIEW

Drag Netal.app into Applications, then open it. Netal lives in the menu bar;
it does not show a Dock icon or a normal main window.

The app requires an Apple silicon Mac and macOS 14 or newer. The experimental
SR/FG game runtime uses macOS 26 APIs; the measured game configuration is M3 Max.

This download contains the menu-bar app, artwork and model-identity metadata.
It does not contain NVIDIA DLLs, model weights, or the separate compiled game
runtime. Installing the app alone does not enable DLSS SR or neural FG.

For DLSS, prepare the supported runtime separately, then choose its folder in
Settings > Choose Runtime Folder. Only the documented DLL builds are tested.
The optional window-enhancement mode uses a separate MetalFX-based path.

Setup: https://github.com/nschlaepfer/Netal/blob/main/docs/getting-started.md
Supported inputs: https://github.com/nschlaepfer/Netal/blob/main/docs/MODEL_SETUP.md
Release/signing status: https://github.com/nschlaepfer/Netal/releases

Known runtime quality and provisioning limits remain documented. This is an
experimental preview, not certification of every game, device or configuration.
TEXT
mkdir -p "${output:h}"
hdiutil create -volname "Netal $version" -srcfolder "$stage" -fs HFS+ -format UDZO "$output"
if [[ -n "${NETAL_CODESIGN_IDENTITY:-}" ]]; then
  codesign --force --sign "$NETAL_CODESIGN_IDENTITY" --timestamp "$output"
  codesign --verify --strict "$output"
fi
hdiutil verify "$output"
print "dmg=$output"
print 'notarization=separate; verify the final app and disk image before publication'
