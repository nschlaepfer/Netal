# SR body source build

The SR body's data and shared Metal library can now be built from source plus the supported local model. This removes the private `work/native/*/pipeline/layout-*.json` dependency and the NumPy dependency from this build step.

**This is not a complete runtime bundle.** General scalar math beyond sampled contracts and complete runtime assembly remain unfinished. The live texture converter now has a source implementation and scoped synthetic conformance checks. Reference body and boundary libraries now build from source; normal-depth variants have compile/load checks only. Boundary metadata and per-profile manifests now come from local DLL extraction and source formulas; see [boundary metadata](SR_BOUNDARY_METADATA.md). The output intentionally has a source-build receipt instead of a runtime `manifest.json`, so it cannot masquerade as a launch-ready bundle.

## Build

Prerequisites: Python 3.10+, Apple's Metal compiler/linker for Metal 4, and the supported extracted model described in [model setup](MODEL_SETUP.md).

```sh
python3 scripts/build-netal-sr-assets.py \
  --model '/path/to/local/model' --output '/new/local/sr-assets'
```

The source-built data includes:

- Eleven PWin stage argument/configuration pairs for each requested output geometry.
- Shape and relocation metadata derived from the model topology and resolution formulas, with zeroed pointer slots.
- Eleven repacked model-weight files and their generated layouts.
- All stage pipeline descriptions and temporal window-shift phases.
- `metallib/pwin_all.metallib`, compiled from the existing project kernels.

Default geometries are 1024×576, 1280×720, 1280×800, 1920×1080, 2560×1440, 3440×1440, and 3840×2160. These are recorded layout cases, not a claim of gameplay support at every size. `--resolutions` selects geometries and `--no-compile` builds CPU data only. Existing output is refused, and a failed build does not expose a partial result.

The builder finds `metal` through `xcrun` and uses the matching adjacent `metallib` when present, accommodating a standalone Apple Metal toolchain. It records compiler identity, source/specification hashes and every generated output hash. Model data stays local and is excluded from source exports.

## Source reconstruction details

`netal_sr_layouts.py` encodes the stage topology, per-layer weight offsets, window shifts, arena assignments, and level-area formulas used by the runtime's geometry synthesis. It constructs typed ABI fields rather than embedding captured binary templates.

`repack.py` maps A/B matrix fragments into row-major tensors using Python's standard library. Half-to-float widening preserves finite values exactly and quiets signaling NaNs while retaining sign/payload. Bias broadcasting and alignment follow the runtime layout.

The encoder-4 projection contains four padded 48-channel groups with 40 live channels each. Both its weight matrix and bias are compacted per group. Merely taking the first 160 columns is wrong. This transformation was missing from the saved source recipe and is now included in both `spec-enc4.json` and its generator.

## Verification

```sh
python3 scripts/test-netal-sr-assets.py
python3 scripts/verify-netal-sr-assets.py \
  --generated '/path/to/sr-assets' --reference '/path/to/reference-runtime' \
  --report '/path/to/comparison.json'
```

Source-only tests require no model, captures, NumPy or GPU. They check geometry, pointer sanitization, pinned ABI fingerprints, every finite half encoding, fragment mapping, bounds failures, group compaction and source-spec regeneration.

The reference comparison checks generated data against a separately supplied prepared runtime. It reports the shared library hashes but never treats differing or matching library bytes as a GPU inference test.

## Current measured boundary

On the release-preparation Mac, all 154 stage ABI/config files across the seven recorded profiles and all eleven repacked weights match the prepared runtime byte-for-byte. Pipeline descriptions match as structured data. The comparison checks 563 data assets, including derived geometry metadata.

The rebuilt body preserves the existing Mac runtime's output on four stored frames at each of 720p, 1080p, 1440p and 2160p. At 1080p, zeroing one pipeline weight file changes every frame, while changing output-buffer poison does not change any frame. These establish a scoped regression check, not general correctness.

**The accelerated baseline configuration and rebuilt body both fail byte identity against the stored RTX outputs in these runs.** The legacy input-kernel override and the default exact-named path were both checked at 1080p. Do not treat historical Mac output hashes as RTX proof or advertise this checkpoint as byte-identical to NVIDIA. Capture provenance, numerical differences, visual quality and sustained game behavior remain to be investigated and validated. No new speed claim or live-game test is made here.

A later isolation check found that the original translated Reference path matches all four stored 1080p frames, while larger/smaller resolutions retain temporal discrepancies. The hardware and batched paths first differ at encoder 0. See [arithmetic modes](SR_ARITHMETIC_MODES.md); the earlier “baseline” here means the accelerated configuration, not every implementation Netal contains.

The main build now includes three formula-generated math tables, two pinned scalar-calibration ROMs, six boundary variants and eleven Reference kernels, bringing the default asset count to 630, including the live converter and existing format extension. The earlier 628 assets remain unchanged; see [boundary kernels and math](SR_BOUNDARY_KERNELS_AND_MATH.md). These remain component builds, not a complete launch-ready bundle or publication clearance.

## Reference kernels and complete runtime inventories

`build-netal-sr-reference-kernels.py` preserves and compiles the eleven retained checked variants, using safe/precise math. The main builder includes them automatically. Their original aliases/hashes are recorded under `src/sr_kernels/reference/SOURCE_ORIGIN.json`; translated-input rights remain under review.

Replacing all eleven kernels preserves four stored 1080p Reference outputs. A disabled encoder-0 control changes 4,399,111 first-frame bytes. The tested Fast input boundary build uses `-ffast-math` to preserve the prepared Fast output on those frames; safe-math candidates drifted and were rejected. Fast output remains numerically different from the RTX oracle. Normal-depth boundary sources compile and load, but the current temporal API rejects non-inverted depth, so their execution is not validated here.

`seal-netal-sr-runtime.py /assembled/runtime --replace` creates a v2 inventory for an already assembled local bundle. It does not fill missing assets or make this partial source-asset output runnable. See [runtime inventory](RUNTIME_BUNDLE.md).
