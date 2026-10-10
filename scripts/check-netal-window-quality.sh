#!/bin/bash
# Offline quality measurements using the production window processor and synthetic
# scenes. No game, screen capture, permission prompt, or installed-app change.
# Exit 0: fixture gates pass; 2: quality defects measured; other: harness error.
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
if [[ $# -lt 1 ]]; then
    echo "usage: $0 NEW_OUTPUT_DIRECTORY [--frames 3…120] [--scenes NAMES] [--modes sr,fg,combined] [--save-every N] [--video] [--backend netal|apple-video]" >&2
    exit 64
fi
OUTPUT="$1"
shift
if [[ -e "$OUTPUT" ]]; then
    echo "Choose a new output directory; previous evidence is never overwritten." >&2
    exit 64
fi
BUILD="$ROOT/build/netal-offline-quality"
SOURCE="${NETAL_QUALITY_PROCESSOR:-$ROOT/src/macos_app/NetalMenuBar/CaptureFrameProcessor.swift}"
FIXTURES="$ROOT/src/macos_app/NetalMenuBar/Tests/CaptureQuality"
mkdir -p "$BUILD"
xcrun swiftc -target arm64-apple-macos14.0 -O \
    -framework Metal -framework MetalFX -framework CoreVideo -framework ImageIO \
    -framework CoreGraphics -framework UniformTypeIdentifiers -framework AVFoundation -framework VideoToolbox \
    "$SOURCE" "$FIXTURES/main.swift" "$FIXTURES/GameSimulation.swift" "$FIXTURES/ReplayVideo.swift" "$FIXTURES/AppleVideoComparator.swift" \
    -o "$BUILD/capture-quality"
ENGINE_HASH="$(shasum -a 256 "$SOURCE" | awk '{print $1}')"
FIXTURE_HASH="$(cat "$FIXTURES/main.swift" "$FIXTURES/GameSimulation.swift" "$FIXTURES/ReplayVideo.swift" "$FIXTURES/AppleVideoComparator.swift" | shasum -a 256 | awk '{print $1}')"
"$BUILD/capture-quality" "$OUTPUT" "$ENGINE_HASH" --fixture-hash "$FIXTURE_HASH" "$@"
