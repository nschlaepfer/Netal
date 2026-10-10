# Menu-bar app

The app provides a game library, experimental window enhancement, and the existing model-runtime controls. It does not run a model simply because a feature appears in the panel.

The familiar compact macOS interface retains Netal's original five-tab navigation, native glass popover, blob in the menu bar, and animated About panel. The generated chrome Netal artwork is the application icon; it does not replace the blob. The user-approved screenshot is the visual baseline. See [Oraculum's shared design language](ORACULUM_DESIGN_LANGUAGE.md).

Choose game under Status opens the game library without changing the five-tab navigation. It discovers installed Steam libraries, App Store games, and apps added manually. Window mode captures only the selected game's window and uses MetalFX spatial scaling plus optional experimental motion interpolation. It operates independently of the prepared native model runtime. This is not DLSS model inference or a universal compatibility claim. See [playing with Netal](WINDOW_MODE.md) for input and presentation limits.

The picker has All, Mac and CrossOver filters and searches titles, sources and bottle names. Its list has one scrolling region; selecting a game opens a compact detail view. Window setup presents the next available action instead of a permanently disabled Start button. Technical explanations are folded into How it works.

Window setup has independent Super Resolution and Frame Generation switches, an output-size choice, and the actual backend names. SR can be disabled while the picture fills the display; that uses ordinary presentation scaling. FG remains experimental and off for a new preference, with motion-artifact and latency guidance when enabled. Native model details identify the supported DLSS 4 Preset K. No selectable or default DLSS 4.5 option is exposed before the actual runtime and gameplay quality are validated.

CrossOver discovery reads installed Steam manifests in private bottles, follows each bottle's Windows drive mappings for external Steam libraries, and recognizes registered non-Steam app wrappers. Games with the same Steam ID in different bottles stay distinct. Shared redistributables, launchers, absent installations and stale Steam URL shortcuts are excluded. CrossOver icons come from existing local menu metadata. Shortcut command strings are never evaluated. Normal Open uses the selected bottle and returns after launch; it never uses the native Mac Steam URL for a CrossOver title. Optional Netal integration remains an explicit, unverified experiment requiring prepared models and a selected game executable.

The SR options expose Fast and Reference arithmetic. Selecting Reference disables batching; enabling batching selects Fast. Reference clears the bridge-managed fast input override. These are implementation choices with limited tested numerical scope, described in [arithmetic modes](SR_ARITHMETIC_MODES.md).

Status shows current runtime activity, frame timing when available, fallbacks, and errors. Frame Generation distinguishes Netal's network from MetalFX interpolation. Source rate must not be labeled display FPS.

The DLSS tab lists SR and FG first-class controls. NR is experimental and is not enabled by normal launchers. Ray Reconstruction remains planned. SR model bytes must match the full supported identities compiled into the app, as well as the model metadata in the bundle manifest. Legacy v1 manifest prefixes remain accepted only when the actual bytes match the full pinned hash; v2 requires full manifest hashes. FG model buffers and math tables must match the supported identities shipped inside the app; every listed runtime file must also match the complete bundle inventory. Missing, modified and unverified states are distinct. Editing a runtime manifest cannot approve different FG model bytes. Verification establishes recorded file integrity, not NVIDIA parity, executable-code trust or image quality.

Settings includes the runtime-folder picker for moved source-preview builds. Select the prepared Netal checkout root. Runtime components and model data must already be provisioned; this picker does not install them. The Metal HUD setting restores the app-owned preference when Netal exits normally or next starts.

Relaunch saves the selected executable and bottle, requests FG only when its model and runtime inventory verify, drains helper output, and displays failures. Save and quit the game before relaunching to avoid losing progress. Discovery does not certify compatibility.

Notifications use Netal's macOS permission. If access is denied, the panel continues to show status; Netal does not send notifications under a different application identity.

## Verification

```sh
zsh scripts/test-macos-netal-menubar.sh
python3 scripts/test-netal-native-launch.py
zsh scripts/test-macos-netal-capture.sh
zsh scripts/build-macos-netal-menubar.sh
build/netal-menubar/Netal.app/Contents/MacOS/Netal --render /tmp/netal-preview
```

The render command creates offscreen light/dark previews. Actual menu interactions, permissions, and game launch behavior require live tests. Automated checks cover log parsing, restart handling, controls, model verification, launch arguments, path relocation, and subprocess output handling.

Offscreen snapshots verify layout and text contrast; they do not reproduce the window compositor's live glass refraction. Window-mode GPU tests use deterministic synthetic images and do not establish game performance. The opt-in `--capture-permission-status` diagnostic only reads existing permission. A bounded `--capture-smoke-bundle <bundle-id> --seconds 8 --output <new-json> --fg` run requires the selected game already foreground and permission already granted; it never requests permission, launches games, or saves screen pixels. Its receipt records actual drawable presentation timestamps and generated/source labels.

Both model inspectors check file sizes, inode identities, modification times and
metadata-change times before reusing a cached result. It rejects symlinked
inputs, incomplete engine inventories, unknown model buffers and unsafe paths.
The app's `FGRuntimeInputs.json` resource comes from
`src/fg_runtime/runtime-inputs.json`; a missing or malformed resource cannot
produce a verified state. The UI check supplements runtime loading, which is
still authoritative for actual initialization and execution.

## FG arithmetic and delivery status

The FG row records **Accelerated** or **Reference** from the runtime's arming
event. Missing or unknown mode data remains unrecorded. Accelerated arithmetic
is the default game path and can differ numerically from Reference; neither
label is a blanket game-quality or pixel-parity certification. The tooltip no
longer calls every FG path exact or promises every other presented frame.

The adapter rejects `NEURALSTUDIO_FG_EXACT=1` when inherited
`NSFG_PREFIX_MODE=hw` or `NSFG_BLOCK_MODE=hw` conflicts with it. Remove those
overrides and restart the game for Reference. Both stage selections are fixed
before the process's first FG runtime use; changing modes requires a fresh game
process. Explicit non-hardware stage overrides remain respected.

Handled source-frame counts alone do not establish generation. The app requires
a positive reported generated-frame count and excludes paused, disarmed or
failed FG from the serving state. Missing generated counts remain unknown, and
a new arming event clears old generated-frame evidence. Initialization alone
now reads as waiting for frames. These observations still do not measure actual
display cadence or certify sustained gameplay.
