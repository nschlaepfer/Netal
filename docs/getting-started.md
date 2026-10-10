# Getting started

This source preview can build the Netal menu-bar app. Gameplay currently requires a separately prepared runtime; the public source tree does not include model weights or captured runtime data.

You must supply your own supported DLL from an authorized source. Only one exact
`nvngx_dlss.dll` hash is validated for SR import; FG has a separate pinned
`nvngx_dlssg.dll` and incomplete fresh-Mac provisioning. Check the full hashes in
[model setup](MODEL_SETUP.md) before importing. A filename or newer version alone
does not establish support. Do not upload DLLs or extracted weights to this repository.

## Build and test the app

On Apple silicon with Apple's Swift compiler and macOS SDK:

```sh
zsh scripts/build-macos-netal-menubar.sh
zsh scripts/test-macos-netal-menubar.sh
open build/netal-menubar/Netal.app
```

The app targets macOS 14+. Game injection uses macOS 26 APIs; the current runtime target is M3 or newer. CrossOver support must be checked against the particular CrossOver/D3DMetal version. Xcode/SDK details should accompany build reports.

## Connect a prepared runtime

When run from `build/netal-menubar`, the app finds its checkout. After moving the app, open Settings → Choose Runtime Folder and select a prepared Netal checkout. For command-line development, `NETAL_RUNTIME_ROOT` overrides that selection.

A prepared runtime contains the launch scripts and compiled bridges plus:

- SR: `work/re9-sr-runtime-bundle`, including locally supplied model data, kernels, profiles, and its manifest.
- FG: `work/re9-fg-runtime-bundle`, including its generated runtime archive, Metal libraries, model tokens, tables, and manifest.
- Model import: source-built capture helpers and the supported locally supplied DLL; see [model setup](MODEL_SETUP.md).

Do not launch a game to fix a missing runtime bundle. Build/provision it first. The importer now verifies supported model bytes without private captured frames, but cannot bootstrap all engine dependencies from an arbitrary DLL; a reproducible public setup procedure is a release blocker.

## Launch a prepared CrossOver session

Save your progress and quit the game before relaunching. Use the exact executable name, Steam app ID, and bottle:

```sh
zsh scripts/crossover-exact-sr.sh --bottle '<bottle>' --appid '<app-id>' \
  --exe '<game.exe>' --sustained --mode hw --fg
```

Omit `--fg` for an SR-only diagnostic. The app's relaunch action supplies the detected executable and bottle, and requests FG when its bundle is installed. Discovery is not a compatibility certification.

For a supported native game with a prepared arm64 bridge:

```sh
zsh scripts/netal-native.sh '/Applications/Game.app'
```

The preflight must accept the game's entitlements; do not disable system protections to force it. Native FG is armed only when its bundle is available. Use `--no-fg` to diagnose SR independently.

## Understand status

“Hashes match” refers to the local model manifest, not an RTX parity run. “MetalFX interpolating” means Netal is passing through frame generation. “Netal network” requires the runtime to report serving frames. Source rate and display rate differ when FG is active.

NR is experimental and has no normal-launch control. See [release status](RELEASE.md) for remaining work and validation requirements.
