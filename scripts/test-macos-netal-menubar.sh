#!/bin/zsh
# Headless checks for the menu-bar app. No XCTest: builds the model and monitor sources alongside
# each test's main.swift with plain swiftc, so it runs anywhere the app builds.
#   Parser  — key=value extraction and the session fold
#   Monitor — file following: truncation, partial lines, the pause-marker round trip
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"; repository="$(cd "$script_dir/.." && pwd)"
src="$repository/src/macos_app/NetalMenuBar"
out="$repository/build/netal-menubar"; mkdir -p "$out"
target=(-swift-version 5 -target arm64-apple-macos14.0)

print "== parser =="
swiftc "${target[@]}" "$src/RuntimeStatus.swift" "$src/MetricsHistory.swift" \
  "$src/Tests/Parser/main.swift" -o "$out/parser-tests"
"$out/parser-tests"

print "\n== monitor =="
swiftc "${target[@]}" -framework Combine -import-objc-header "$src/NetalProcessInfo.h" \
  "$src/RuntimeStatus.swift" "$src/RuntimeMonitor.swift" \
  "$src/MetricsHistory.swift" "$src/GameIdentity.swift" "$src/ModelStatus.swift" "$src/FGBundleStatus.swift" "$src/RuntimeFileSignature.swift" \
  "$src/SessionDiscovery.swift" "$src/RuntimeSession.swift" "$src/NotificationPolicy.swift" "$src/NetalVitals.swift" \
  "$src/CrossOverGames.swift" "$src/MetalFXApps.swift" \
  "$src/Tests/Monitor/main.swift" \
  -o "$out/monitor-tests"
"$out/monitor-tests"

print "\n== release behavior =="
swiftc "${target[@]}" -framework AppKit \
  "$src/RuntimeStatus.swift" "$src/DLSSFeatures.swift" "$src/GameIdentity.swift" \
  "$src/NetalBuildDefaults.swift" "$src/RuntimeProcess.swift" "$src/GameLauncher.swift" \
  "$src/NativeGameCompatibility.swift" \
  "$src/Tests/Release/main.swift" -o "$out/release-tests"
"$out/release-tests"

print "\n== installed game library =="
swiftc "${target[@]}" -framework AppKit \
  "$src/GameIdentity.swift" "$src/GameLibrary.swift" "$src/CrossOverLibrary.swift" "$src/Tests/GameLibrary/main.swift" \
  -o "$out/game-library-tests"
"$out/game-library-tests"

print "\n== CrossOver installed game library =="
swiftc "${target[@]}" -framework AppKit \
  "$src/GameIdentity.swift" "$src/GameLibrary.swift" "$src/CrossOverLibrary.swift" "$src/Tests/CrossOverLibrary/main.swift" \
  -o "$out/crossover-library-tests"
"$out/crossover-library-tests"

print "\n== native game eligibility and discovery =="
swiftc "${target[@]}" -framework AppKit -import-objc-header "$src/NetalProcessInfo.h" \
  "$src/GameIdentity.swift" "$src/MetalFXApps.swift" "$src/SessionDiscovery.swift" "$src/RuntimeSession.swift" \
  "$src/Tests/NativeDiscovery/main.swift" -o "$out/native-discovery-tests"
"$out/native-discovery-tests"
python3 "$repository/scripts/test-netal-native-launch.py"

print "\n== FG model and runtime verification =="
swiftc "${target[@]}" "$src/RuntimeFileSignature.swift" "$src/FGBundleStatus.swift" "$src/Tests/FGModel/main.swift" -o "$out/fg-model-tests"
"$out/fg-model-tests" "$repository/src/fg_runtime/runtime-inputs.json"
