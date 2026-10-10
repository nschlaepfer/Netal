# Building the FG engine from source

This build-only path compiles the FG engine without an accepted research attempt,
model files, precompiled libraries, inference, or game launch. It is a compilation
milestone, not a complete runtime or a gameplay validation.

On macOS with Xcode command-line tools, the Metal compiler toolchain, and a macOS
26 or newer SDK available through `xcrun`:

```sh
python3 scripts/build-netal-fg-engine.py --output /path/to/new/fg-engine
```

The output must not exist. Compilation occurs in a unique sibling staging
folder; the completed engine is moved into place only after every compiler and
archive step succeeds. A failed build leaves existing outputs alone.

The recipe in `src/fg_runtime/kernel-build.json` selects the 640x360, u32 HMMA,
compact EX2, live-zero source configuration. It produces 73 Metal libraries,
arm64 and x86_64 static host archives, a universal archive, a public header,
generated source snapshots, and a SHA-256 build receipt with toolchain versions.
The recipe shares its overlay and host-tail generators with the existing
research build scripts. Those research scripts also execute workloads; this
build-only command does not.

## Assemble a local runtime

With an engine built above and permitted, locally supplied model buffers:

```sh
python3 scripts/assemble-netal-fg-runtime.py \
  --engine /path/to/fg-engine \
  --model /path/to/model-buffers \
  --output /path/to/new/fg-runtime
```

The RCP correction ROM defaults to the separately documented, pinned standalone
arithmetic asset in the source tree; `--rcp` can select an identical local copy.
The three FG tables now default to the pinned standalone arithmetic assets in
`runtime/fg-arithmetic-calibration/`; `--sigmoid`, `--ex2-unit`, and
`--ex2-fraction` can select identical local copies. Run
`check-netal-fg-calibration.py` on the target Mac before using its EX2 corrections;
see the [calibration record](../runtime/fg-arithmetic-calibration/README.md).
The assembler verifies the complete engine
inventory, all 67 model buffers (4,868,576 bytes), and all four math-table
identities before publishing a new bundle. It rejects missing/extra model
buffers, modified bytes, symlinks, duplicate manifest keys, unsafe paths, and an
existing output. The output directory must be outside the engine/model inputs.
It does not load the model or execute Metal.

`src/fg_runtime/runtime-inputs.json` records the supported input hashes. Model
identities come from the retained post-evaluate-0 capture manifest
`2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294`, for runtime
`5d5cbf14d2727d47f93fd10bf77bd91708ae122482a6f86fd564971641ebd47b`.
The table identities match the runtime's existing hard-coded checks. Hashes
identify data; they do not establish permission to redistribute it. No model
buffers are included in the source export. Five exact scalar-data identities
(two shared RCP/RSQ assets and three FG assets) are explicit binary exceptions.

The resulting bundle supplies `libs`, `lib`, `include`, `model`, tables, and the
existing `netal.fg.runtime-bundle.v1` inventory used by the bridge build. It
records the source-engine receipt and input-contract hashes, and leaves
execution/gameplay validation false. An unsigned build receipt is an integrity
record, not authentication of the supplied engine's origin.

The legacy `build-macos-fg-runtime-bundle.sh` is retained for research attempts.
It now requires a new output directory and preserves previous bundles; its
private default inputs are not a fresh-Mac setup path.

## Verified scope and remaining work

On the development M3 Max with macOS SDK 27.0, the source engine passed the
existing 16-frame synthetic API test in four fresh processes across two buffer
poison patterns. All 15 generated outputs matched the retained prepared-runtime
receipt. Input/canary checks, rejected-output preservation, reset/replay, and
non-audit output agreement passed. A relocated bundle also matched those 15
outputs through the Metal 4 command-buffer path. These are synthetic functional
checks, not sustained gameplay or a device support matrix.

The remaining FG input bootstrap dependency is obtaining the supported model
buffers from permitted local inputs. The scalar tables have explicit, pinned
source-tree assets and reproduction records. The assembler does
not extract them from a DLL. Fresh-Mac provisioning, SR+FG integration, visual
quality, sustained frame pacing, hardware-prefix modes, and distribution
validation remain open. Successful x86_64 compilation does not establish
supported Intel execution.

The public-source export retains required FG text headers and host includes.
Passing its content/inventory audit does not clear source redistribution
rights; see [source provenance review](SOURCE_PROVENANCE_REVIEW.md).

## Accelerated game path

The game adapter defaults to hardware arithmetic in the prefix and custom
block. The rebuilt runtime passed two fresh-process synthetic 16-frame runs in
that configuration, including poison independence, reset/replay and input/guard
checks. Its Metal 4 outputs matched those fresh accelerated outputs. Against
the recorded Reference outputs, 49,990 RGBA bytes differed across 15 generated
frames, with maximum byte error 9. This establishes that the two modes differ;
it is not a visual acceptance or sustained-gameplay result.

`NEURALSTUDIO_FG_EXACT=1` requests Reference and now rejects conflicting inherited
hardware overrides. The app displays the reported arithmetic mode rather than
calling accelerated output exact. Restart the game when changing modes.
