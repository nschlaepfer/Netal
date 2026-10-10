#!/bin/bash
# Synthetic GPU checks plus a denied-permission CLI integration test. No game is
# launched, no screen pixels are captured, and no permission prompt is requested.
set -euo pipefail
REPO_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
APP_SRC="$REPO_DIR/src/macos_app/NetalMenuBar"
TEST_BUILD="$REPO_DIR/build/netal-capture-tests"
mkdir -p "$TEST_BUILD"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--diagnostics-only" ) ]]; then
    echo "usage: $0 [--diagnostics-only]" >&2
    exit 64
fi
if [[ $# -eq 0 ]]; then
    xcrun swiftc -target arm64-apple-macos14.0 -O \
        -framework Metal -framework MetalFX -framework CoreVideo \
        "$APP_SRC/CaptureFrameProcessor.swift" "$APP_SRC/Tests/Capture/main.swift" \
        -o "$TEST_BUILD/capture-gpu-tests"
    "$TEST_BUILD/capture-gpu-tests" | tee "$TEST_BUILD/gpu-tests.log"
fi
xcrun swiftc -swift-version 5 -target arm64-apple-macos14.0 -O \
    "$APP_SRC/GameCaptureEngine.swift" "$APP_SRC/CaptureFrameProcessor.swift" \
    "$APP_SRC/CaptureDiagnostics.swift" "$APP_SRC/Tests/CaptureDiagnostics/main.swift" \
    -framework AppKit -framework Metal -framework MetalFX -framework ScreenCaptureKit \
    -framework CoreMedia -framework CoreVideo -framework Carbon \
    -o "$TEST_BUILD/capture-denied-permission-test"
TEST_RECEIPTS="$(mktemp -d "$TEST_BUILD/diagnostics.XXXXXX")"
"$TEST_BUILD/capture-denied-permission-test" --test-mode-validation
"$TEST_BUILD/capture-denied-permission-test" --capture-permission-status > "$TEST_RECEIPTS/status.json"
rg -Fxq '{"granted":false}' "$TEST_RECEIPTS/status.json"
set +e
"$TEST_BUILD/capture-denied-permission-test" --capture-smoke-window 1 --pid 1 --seconds 1 \
    --output "$TEST_RECEIPTS/denied.json" > "$TEST_RECEIPTS/stdout.json"
DENIED_EXIT=$?
set -e
[[ "$DENIED_EXIT" == 77 ]]
rg -Fq '"status":"permission-denied"' "$TEST_RECEIPTS/stdout.json"
rg -Fq '"permission_requested":false' "$TEST_RECEIPTS/stdout.json"
echo "capture permission-denied integration: PASS (exit 77; no permission request)"
echo "diagnostic receipts: $TEST_RECEIPTS"
