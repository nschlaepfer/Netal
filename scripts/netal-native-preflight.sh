#!/bin/zsh
# Read-only launch eligibility, shared with the app and launcher. --observe allows diagnostics
# without static MetalFX evidence; it does not authorize model acceleration.
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$script_dir/netal-native-inspect.py" "$@"
