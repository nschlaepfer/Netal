#!/bin/zsh
# Install only the supported SR model using full hashes and atomic directory exchange.
# Model-byte identity is checked here; engine/gameplay validation is a separate release gate.
set -euo pipefail
repository="$(cd "$(dirname "$0")/.." && pwd)"
exec python3 "$repository/tools/netal-model-install.py" "$@"
