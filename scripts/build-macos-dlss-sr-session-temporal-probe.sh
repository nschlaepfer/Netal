#!/bin/zsh
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
repository="$(cd "$script_dir/.." && pwd)"
binary="$repository/build/macos_host/dlss-sr-session-temporal-probe"

mkdir -p "$(dirname "$binary")"
xcrun clang++ -fobjc-arc -std=c++20 -O2 -Wall -Wextra -Werror \
  -framework Foundation -framework Metal \
  "$repository/src/macos_host/dlss_sr_session.mm" \
  "$repository/src/macos_host/dlss_sr_session_temporal_probe.mm" \
  -o "$binary"
codesign --force --sign - "$binary" >/dev/null
print "binary=$binary"
print "sha256=$(shasum -a 256 "$binary" | awk '{print $1}')"
print "result=pass"
