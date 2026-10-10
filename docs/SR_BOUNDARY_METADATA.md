# Source-built SR boundary metadata

The supported local DLL extraction now produces three canonical boundary templates alongside the model files. These come from DLL execution under the source-built Wine recorder, not from private RTX frame captures. No GPU is required for extraction.

The extractor accepts the pinned quality model only by default. It requires exactly one launch of each expected boundary function, validates argument sizes and canonical hashes, removes GPU/process pointer slots, and zeros known padding. The live half-precision coefficient at output byte 192 is preserved; bytes 194–199 are padding, not another coefficient or a pointer to retain.

## Build flow

```sh
python3 tools/netal-dlss-extract.py '/path/to/nvngx_dlss.dll' \
  --capture-dir build/model-capture --out '/new/local/model'
python3 scripts/build-netal-sr-assets.py \
  --model '/new/local/model' --output '/new/local/sr-assets'
```

Re-extract older model packages to obtain their `abi/` directory. `--body-only` retains the previous diagnostic build behavior for packages without boundary metadata.

The asset builder derives both-axis render ratios, bounds, and half-precision limits for each requested profile. It now emits 606 assets for the default seven profiles, including the boundary ABI and complete per-profile geometry manifests. It still does not emit a root runtime manifest: boundary shader implementations, arithmetic tables and legacy fallback kernels are separate unfinished dependencies.

Model installation retains validated metadata under `model/abi`. It does not silently replace an existing engine's ABI files. Engine provisioning remains an explicit build step.

## Validation

Two independent local DLL runs produced identical canonical template files. Against the prepared runtime, all 24 root/profile ABI files match after explicit pointer/padding normalization, and all seven generated profile contracts match. The complete data comparison covers 594 assets. This is not a raw-byte equality claim for discarded padding.

An isolated generated-profile candidate preserves all four stored 1080p RTX outputs in Reference mode. The test kept the existing boundary shaders, arithmetic tables and remaining engine binaries, so it is not a full clean-source engine proof.

## Render-size corrections

The runtime previously updated the horizontal half-precision limit at byte 190 during a render-size change but left the vertical limit at byte 192 unchanged. For 960×540 → 1920×1080, that produced `[-32, -24]` where the captured runtime arguments contain `[-32, -32]`. Both axes are now updated.

A second case kept rendering at 1280×720 while synthesizing a 2560×1440 output. The old template retained approximately 2/3 render/output ratios; the corrected runtime uses 1/2 on both axes and limits `[-32, -32]`. This case had no RTX output oracle and remains explicitly unverified for pixel correctness; the check establishes parameter propagation.

With the correction, the forced-synthesis 2560×1440 path produces the same outputs as its recorded profile on four stored frames. The 960×540 render-change run remains finite, and its argument limits match the capture. Temporal byte parity is still incomplete in those cases; the fix is not a claim that all numerical or visual discrepancies are resolved.

## Tests

```sh
python3 scripts/test-netal-sr-abi.py
zsh scripts/test-netal-sr-boundary-geometry.sh
```

The source-only tests cover address/padding sanitation, preservation of the coefficient at 192, geometry/bounds, malformed packages, duplicate/missing launches, path traversal, and truncated or changed files. The existing extraction/install tests still apply. GPU regression evidence remains separate from these unit checks.
