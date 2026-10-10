# Runtime bundles

The private research checkout contains captured and generated material that must not be assumed to belong in a public source release. The exporter omits the entire `work/` and `build/` trees, all binary model data, and compiled executables/libraries. Five exact hash-pinned scalar calibration files are explicit data exceptions, documented in `runtime/arithmetic-calibration/README.md` and `runtime/fg-arithmetic-calibration/README.md`; arbitrary binaries remain excluded.

## SR

The runtime expects `work/re9-sr-runtime-bundle` with a manifest, model weights and constants, Metal libraries, math tables, geometry, and pipeline descriptions. The historical builder depends on sibling research captures. The updated import script needs a runtime skeleton and source-built capture helpers; it validates the supported full model hashes without private frame captures or a temporal probe. See [local model setup](MODEL_SETUP.md).

A DLL picker alone is not a clean-machine installer. Implement and validate a complete bootstrap before advertising one-click model setup. Keep user model files local and do not upload them in bug reports.

## FG

The runtime expects `work/re9-fg-runtime-bundle`. The source-only engine builder compiles shaders and universal host archives without a research attempt. The explicit-input assembler combines that engine with identity-checked local model tokens and arithmetic tables; see [FG source build and assembly](FG_SOURCE_BUILD.md). The scalar tables are now explicit pinned assets with origin records and a local EX2 compatibility check. Obtaining the supported model tokens reproducibly on a fresh Mac remains open. Its recorded profile is 640×360 internally.

## NR

No NR model data is included. Native reference execution and potential MLX execution are separate experimental backends. NR is not enabled by SR/FG launchers.

## CrossOver launch configuration

Build the bridge, Windows injection helpers, Unix helper and native configuration tool before launching:

```sh
NEURALSTUDIO_FG_BUNDLE=/absolute/path/to/prepared/fg-bundle \
  zsh scripts/build-macos-re9-wine-injector.sh
zsh scripts/test-netal-launch-config.sh
python3 scripts/test-crossover-launch.py
```

The build requires an FG archive from an assembled bundle. It does not construct a complete runtime from public source. The injector now reuses these binaries rather than compiling during each attachment. Dynamic-library install names are relative, and the helper contains no compiled checkout or user-home paths.

For each selected macOS process, `netal-launch-config` records its PID, owner and start timestamp. Settings and logs live in `~/Library/Application Support/Netal/Sessions/<pid>-<seconds>-<microseconds>/`. The session directory is private (0700), and `launch.conf` is written atomically with private permissions (0600). The Unix helper rejects missing, malformed, duplicate, stale, linked or publicly readable configuration files. It uses the configured bridge, SR and FG paths at launch. This is local process routing, not a signature or trust decision about supplied libraries.

The menu-bar app discovers these logs and reads the selected bundle locations from the associated private configuration. Live controls stay beside the session log. A recycled PID cannot reuse a previous process's configuration or readiness result. Old session folders are retained for diagnosis; no automatic cleanup is implemented yet.

Use `mac-dlss-attach.sh --exe 'Game Name.exe'` for manual attachment. An explicit bare executable name avoids guessing from an ambiguous process command line. The lower-level injector also requires `--pid=<macOS PID>`. `NETAL_CROSSOVER_ROOT` can select a non-default CrossOver installation.

The helper tests use a synthetic x86_64 host and a fixture bridge, including relocation to a directory containing spaces. The separate CrossOver integration test uses real Wine processes in disposable bottles. Neither executes a game or a model.

### Portable CrossOver launch and process identity

The release launcher and injector load helpers from Netal's own folder. The old hook-copying route has been removed: neither installs DLLs or Unix libraries into CrossOver.app or the bottle. `--portable` is accepted for older callers but is no longer an opt-in mode. Conflicting shared hooks cause refusal.

The `netal_wine_host_v1.dll` / `.so` pair has a versioned interface so historical one-entry Unix helpers cannot be confused with the new protocol. Rebuild the complete pair and controller together when updating. Quit Steam and games before switching runtime builds.

The controller supports three operations:

- `--find <executable>` finds a unique Windows process inside the selected Wine server. Missing and duplicate matches are distinct failures.
- `--query <executable> <absolute Windows DLL path>` loads only the helper and returns its actual Mac PID. It does not configure or start the graphics bridge.
- `<executable> <absolute Windows DLL path>` explicitly starts the native bridge after its private launch configuration has been prepared. Native failures now produce a nonzero controller exit status. The launcher still requires the bridge's readiness log and mapped image; none of these alone proves active SR/FG frames.

The controller resolves its versioned exports from the target's mapped image instead of assuming the DLL on disk has the same function offsets. The injector compares the bottle-scoped query result with the caller's Mac PID before writing settings or starting the bridge.

The target must start with Netal's helper folder in its Unix `WINEDLLPATH`. CrossOver's wrapper sets its own paths, so the launcher obtains the selected bottle's normal path, prepends the helper folder, and passes the result through `--env` when starting Steam. It checks any existing Steam client before terminating a game; an unprepared client produces an instruction to quit Steam and launch again through Netal. Manual attachment also selects its target inside the specified bottle, using an explicit executable name.

Real CrossOver 26.3.0.39832 fixture tests pass discovery without activation, explicit startup, missing configuration, missing startup path, malformed arguments, and two simultaneous bottles running identically named executables. Each bottle receives only its own configuration and native receipt; duplicate instances within one bottle are rejected. Protected hook paths remain absent.

The installed 64-bit Steam client also passed a separate startup check: it inherited the helper path, returned its native PID, mapped the versioned Unix helper, created no graphics runtime session, and stopped after a normal shutdown request. This does not establish Steam-to-game environment propagation, UI/game compatibility or SR/FG output.

That check exposed a Wine output-lifetime problem: captured pipes could stay open after a query command exited. `netal-wine-output.sh` now spools command output to private temporary files, forwards bounded output after the command exits, and removes its temporary paths. Both command success and failure are tested while a child still holds the original output handles. The production capture helper was exercised against the real Steam client. Startup failures and bridge readiness remain separate from successful output collection.

To reproduce this scoped test on a Mac with CrossOver, Xcode/Command Line Tools, Homebrew LLVM and LLD:

```sh
zsh scripts/build-macos-re9-wine-injector.sh --helpers-only
python3 scripts/test-netal-crossover-helper.py --output "$PWD/build/crossover-helper-test"
python3 scripts/test-netal-portable-injector.py
```

The evidence output must not exist. The test creates and stops its own bottles, retaining logs, hashes and results there; it never stops an existing user bottle. `--helpers-only` builds no Metal bridge and needs no FG archive. Normal full builds retain the prepared-FG dependency.

A separate Resident Evil Requiem startup check on macOS 27.0 (26A5406e), CrossOver 26.3 and game build 23634047 reached the game through Steam with the portable settings. The real bridge loaded; FG initialized and SR reported all eleven pipeline layers and initialization readiness. This used a private copy of hash-pinned prepared assets, not a clean-source runtime. SR was still initializing at its logged 356-second wait marker. The desktop was locked, no game input was sent, and no SR completion or FG generation event was observed. The test game and Steam were stopped and the GPU reservation released. This proves startup integration only; frame execution, visual acceptance, sustained pacing and recovery remain open.

### Interpreting a live receipt

```sh
python3 scripts/summarize-re9-live-sustained-sr.py --log '/absolute/path/to/session/re9-live-metalfx.log' --output session-summary.json
```

The version 2 receipt covers SR and FG separately. Initialization, scheduled/completed SR work, FG invocation/generation counters, failures and timing samples are distinct fields. Missing observations remain null or empty rather than becoming measured zeroes. Incomplete trailing writes are ignored, nonfinite metrics are excluded and flagged, and logged counters are labeled as sampled observations. The legacy event word `exact` does not certify pixel parity or turn Fast mode into Reference. Game command-buffer GPU time includes other game work; no physical dispatch count or displayed FPS is inferred. Visual, sustained-gameplay and parity claims remain false until backed by separate validation.

Wine's current development branch provides an additional direct Unix-library-by-name API, but the installed CrossOver version returned `STATUS_INVALID_INFO_CLASS` for that probe. Netal does not depend on it. Relevant upstream contracts: [Wine 11 loader](https://github.com/wine-mirror/wine/blob/wine-11.0/dlls/ntdll/unix/loader.c) and [current Wine memory information classes](https://github.com/wine-mirror/wine/blob/master/include/winternl.h). The local CrossOver test establishes the compatibility claim, not the upstream header alone.

## Distribution work still required

Build all redistributable components reproducibly, establish the provenance of generated source and tables, install permitted user-supplied model data locally, and separate read-only app resources from writable Application Support data. See [release gate](RELEASE.md). The MIT code license does not license every historical research artifact.

## Version 2 runtime inventory

For a fully assembled local runtime, `python3 scripts/seal-netal-sr-runtime.py /path/to/runtime` records every regular file except the root manifest, validates required component presence and pinned original model bytes, rejects symlinks/unsafe paths and captured pointer slots, and writes a new manifest. Existing manifests require the explicit `--replace` flag. This command supplies no missing libraries or scalar data.

The `netal.sr.runtime.v2` root hashes the UTF-8 prefix `netal.sr.runtime.v2\n`, followed by rows sorted by their ASCII relative path. Each row is path, NUL, decimal byte length, NUL, lowercase full SHA-256, newline. Native loading recomputes this root, checks every file and rejects missing or unlisted assets. It also pins weights/constants and checks pointer sanitization. The validated root is retained in the session rather than reread from mutable metadata later. Hashes establish integrity of the recorded bundle, not authorship, legal permission, correct inference or gameplay acceptance.

Legacy v1 bundles remain supported, but their core identity is now actually recomputed from sorted file rows rather than trusting a copied root string. Only the local `ingest` receipt is excluded from that historical core identity. A changed listed kernel cannot retain the old immutable label. Experimental replacements should use v2 or explicit existing override mechanisms; do not rewrite legacy rows under an unchanged root.

Model reinstallation checks v2 engine inventory before making changes, repairs/imports the supported model through the existing atomic directory swap, and reseals the complete staged inventory. A malformed model or nonzero captured pointer remains rejected even if the inventory root is recomputed. File integrity is separate from trust in user-selected code.

## Source-built live texture conversion

`python3 scripts/build-netal-sr-texture-kernels.py --output /new/texture-kernels` builds the four-entry converter and the existing optional format-extension library. The main asset builder includes both. The converter uses the adapter's texture/buffer/size bindings and Metal's format operations; it contains no model or recorded frame data.

On the tested M3 Max, synthetic comparison against the prepared converter passes RGB9E5 import over 32,896 pixels (including all per-channel mantissa/exponent combinations), export over 65,536 varied half-value triples, all 65,536 R16 exposure encodings, and 527 nonuniform depth/stencil samples. CPU checks independently verify RGB9E5 decoding and depth values. Oversized dispatches, buffer guards and untouched texture margins are checked. Metal 4 import/export agrees with the Metal 3 reference results. A no-write export changes 261,463 bytes and is rejected.

These checks do not cover every joint RGB triple, every device/compiler, the optional extension's full format set, SR inference quality or live gameplay. Depth import is retained for the adapter ABI; the current temporal adapter generally uses a depth blit. Use `build-netal-texture-converter-probe.sh` to build the GPU probe, then supply an explicit prepared reference and candidate library. No GPU work is started by the build command itself.

For the complete converter comparison and no-write control, run under the local GPU reservation workflow:

```sh
python3 scripts/test-netal-texture-converter.py --reference /path/to/pinned/prepared/dlss_sr_live_texture_convert.metallib --output /new/conformance-output
```

The reference hash is fixed by the tool; comparing the new library against itself cannot satisfy this command. The recorded reference binary is a test input, not a source-build dependency or part of the public export.
