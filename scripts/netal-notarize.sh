#!/bin/zsh
# Notarize and staple the built Netal.app, then produce a distributable zip.
#
# Signing already happens in the build; this is the separate step that makes Gatekeeper trust the
# app on a machine that has never seen it. Netal ships as a menu bar app people download rather than
# build, so it needs it.
#
# One-time setup, which has to be done by the person holding the Apple ID - it asks for an
# app-specific password, and nothing else here or in the repository ever sees that value:
#
#   xcrun notarytool store-credentials netal \
#     --apple-id <your-apple-id> --team-id <your-team-id>
#
# The password is created at appleid.apple.com -> Sign-In and Security -> App-Specific Passwords.
# notarytool stores it in the login keychain under the profile name, and every run after that is
# just this script.
set -euo pipefail

repository="$(cd "$(dirname "$0")/.." && pwd)"
app="$repository/build/netal-menubar/Netal.app"
profile="${NETAL_NOTARY_PROFILE:-netal}"
out="$repository/build/netal-release"

fail() { print -u2 "\n  $1\n"; exit 1 }

[[ -d "$app" ]] || fail "no app at $app — run scripts/build-macos-netal-menubar.sh first"

# Notarization rejects anything not signed with a Developer ID, hardened runtime and a secure
# timestamp. Checking here turns a five-minute round trip into an immediate answer.
info="$(codesign -dv --verbose=2 "$app" 2>&1)"
print "$info" | grep -q "Developer ID Application" \
  || fail "not signed with a Developer ID — Gatekeeper would reject it and notarization will too"
print "$info" | grep -q "flags=.*runtime" \
  || fail "hardened runtime is not enabled; notarization requires it"
print "$info" | grep -q "^Timestamp=" \
  || fail "signed without a secure timestamp (was it signed offline?)"
codesign --verify --deep --strict "$app" || fail "signature does not verify"

xcrun notarytool history --keychain-profile "$profile" >/dev/null 2>&1 || fail \
  "no notary credentials stored under the profile '$profile'.
  Run this once, with your own Apple ID — it will prompt for an app-specific password:

    xcrun notarytool store-credentials $profile --apple-id <your-apple-id> --team-id <your-team-id>"

mkdir -p "$out"
zip="$out/Netal.zip"
rm -f "$zip"
# ditto, not zip: it preserves the bundle's symlinks and extended attributes, and a plain zip has
# been known to produce an archive that notarizes but will not launch.
/usr/bin/ditto -c -k --keepParent "$app" "$zip"

print "\nsubmitting $(du -h "$zip" | cut -f1) to Apple — this usually takes a few minutes"
xcrun notarytool submit "$zip" --keychain-profile "$profile" --wait | sed 's/^/  /'

# Stapling writes the ticket into the bundle so it validates without a network round trip - the
# difference between opening cleanly on a fresh machine and only after it can reach Apple.
xcrun stapler staple "$app" | sed 's/^/  /'
xcrun stapler validate "$app" | sed 's/^/  /'

# Re-zip after stapling: the archive made above holds the unstapled bundle.
rm -f "$zip"
/usr/bin/ditto -c -k --keepParent "$app" "$zip"

print "\nverifying the way Gatekeeper will, on this machine:"
spctl --assess --type execute --verbose=2 "$app" 2>&1 | sed 's/^/  /'
print "\nresult=pass  $zip"
