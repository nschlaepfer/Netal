# Native Metal DLSS 5 neural rendering (`src/nr_fast/`) and opt-in adapter

`src/nr_fast/` and `src/macos_host/netal_nr_fast_bridge.mm` provide a standalone,
zero-MLX native Metal 3.1 / C++20 DLSS 5 neural rendering runtime and an opt-in
in-process game adapter for Apple Silicon (`MTLGPUFamilyApple7` and newer).

This path is an engineering and experimental gameplay component, separate from
the reference PTX/Swin parity probes in `src/nr_swin/`. It uses hardware
`simdgroup_half8x8` matrix multiply-accumulate kernels rather than integer/FP32
fixed-point tensor-core emulation, and remains **off by default** in all normal
Netal launches. No model weights are included in the repository.

## Architecture

1. **Single-encoder native Metal runtime (`src/nr_fast/`)**:
   - `nr_fast_runtime.{h,mm}` and `nr_fast_kernels.metal` load either `.dlssmodel`
     bundles (`manifest.json` SHA-256 verified) or standalone `.safetensors`
     checkpoints (`F16`, `F32`, or `mlxdlss-native-e4m3-v1` `F8_E4M3` decoded
     via an exact 256-entry IEEE-754 binary16 LUT), as well as deterministic
     synthetic weights (`Small31` 31-block `[2,2,2,2,2]` and `HT71` 71-block
     `[4,4,6,8,8]`) for model-free CI and benchmarking.
   - `Context::encode()` submits the full frame graph inside a single serial
     `MTLComputeCommandEncoder` without host readbacks:
     PCG RXS-M-XS (`0x108ef2d9`) Gaussian noise + 5-tap Catmull-Rom temporal
     reprojection (`motion_scale_x`, `motion_scale_y`) -> fused `simdgroup_half8x8`
     window, branched-FFN, grouped-GEMM, and SRAM-resident global attention
     blocks -> temporal sigmoid blend + double-buffered history update ->
     optional separable Gaussian detail-split upsampling.
   - `FrameInputs::export_delta_buffer` and `Context::encode_compose_delta()`
     decouple heavy U-Net evaluation from display composition, using an isolated
     `async_compose_scratch_buf` so background worker encoding and 60–120 Hz
     display composition never contend on mutable GPU buffers.

2. **GPU optical-flow & scene-cut synthesizer (`src/macos_host/netal_nr_optical_flow.{h,metal}`)**:
   - For spatial MetalFX games and color-only `CAMetalLayer` / Metal 4 renderers
     where `motionTexture == nil`, computes perceptual luma pyramids, frame-wide
     scene-cut detection (zeroing flow on cuts on the GPU), SRAM-cached
     quarter-resolution patch search (`+/-96` full-res px), and edge-aware
     subpixel upsampling to `RG16Float` pixel-space flow with zero CPU stalls.

3. **Continuous state machine & in-process bridge (`netal_nr_continuous_state.h`, `netal_nr_fast_bridge.mm`)**:
   - `libNetalNRFastBridge.dylib` exits its constructor immediately without
     allocating state, opening files, or installing hooks unless
     `NETAL_NR_ADAPTER_MODE` is explicitly set (`continuous`, `once`, or
     `delayed_once`) and `NEURALSTUDIO_TARGET_EXE` (when set) matches the target
     render executable (keeping parent launchers such as `nxnative` unhooked
     while activating in child render hosts such as `dmhost`).
   - Supports three interception seams:
     - **Metal 3 `MTLFXTemporalScaler`** (`encodeToCommandBuffer:`, using native
       motion vectors and depth when present, falling back to the original
       MetalFX scaler while weights prepare or when paused);
     - **Metal 3 `MTLFXSpatialScaler`** when `NETAL_NR_HOOK_SPATIAL=1`;
     - **`CAMetalLayer` + Metal 4 `MTL4CommandQueue signalDrawable:`** when
       `NETAL_NR_HOOK_DRAWABLE=1`, synchronizing `MTL4CommandQueue` <-> helper
       `MTLCommandQueue` via `MTLSharedEvent` and using a cached 3-vertex
       fullscreen render pass when `drawable.texture` carries
       `MTLTextureUsageRenderTarget` without `MTLTextureUsageShaderWrite` or
       uses hardware `*_sRGB` pixel formats.

## Source build and model-free verification

Build the standalone runtime + CLI and the in-process game bridge into fresh
output directories (existing destinations are refused; builds stage atomically
with SHA-256 `build-receipt.json` manifests):

```sh
python3 scripts/build-netal-nr-fast.py --output /Users/example/work/nr-fast
python3 scripts/build-netal-nr-fast-bridge.py --output /Users/example/work/nr-fast-bridge
```

Run the model-free verification suites:

```sh
zsh scripts/test-netal-nr-continuous-state.sh
python3 scripts/test-netal-nr-optical-flow.py
python3 scripts/test-netal-nr-fast.py
python3 scripts/test-netal-nr-fast-bridge.py
```

When a local `mlxdlss` binary and permitted local `.dlssmodel` / `.safetensors`
checkpoint are available, verify numerical parity and latency with:

```sh
python3 scripts/compare-nr-native-vs-mlx.py \
  --mlx-bin /Users/example/bin/mlxdlss \
  --model /Users/example/models/NeuralRendering.dlssmodel \
  --fp8-model /Users/example/models/dlssnr-weights-native-fp8.safetensors
```

## Measured Apple M4 Max parity and GPU timings

Measured on an Apple M4 Max (32-core GPU, macOS 27.2, Xcode 27.2):

### Parity against `MLX-DLSS` (`HT71` 71-block checkpoint, `256x384`)

| Check | `netal_nr_fast` p50 GPU | `MLX-DLSS` warm | RGB MAE | PSNR (peak=1) | Max abs diff |
| --- | --- | --- | --- | --- | --- |
| Raw `[1, 256, 384, 16] -> [1, 256, 384, 4]` U-Net head | 10.16 ms | 49.05 ms (4.83x) | 0.000000 | >99 dB (exact) | 0.000000 |
| `F8_E4M3` safetensors vs `.dlssmodel` F16 | 10.10 ms | — | 0.000000 | bit-exact | 0.000000 |
| End-to-end `render-image` (`standard`) | 11.39 ms | 45.62 ms | 0.001888 | 50.75 dB | 0.019608 |
| End-to-end `render-image` (`cinematic`) | 11.34 ms | 51.82 ms | 0.004882 | 44.10 dB | 0.027451 |

### Synthetic temporal GPU latency (`netal-nr-fast-cli bench --temporal`)

| Topology | Internal network | Output | Dispatches | p50 GPU | p95 GPU |
| --- | --- | --- | --- | --- | --- |
| `Small31` (31 blocks) | `640x384` | `1280x720` | 151 | 9.02 ms | 9.27 ms |
| `Small31` (31 blocks) | `960x544` | `1920x1080` | 151 | 16.86 ms | 17.09 ms |
| `HT71` (71 blocks) | `640x384` | `1280x720` | 351 | 16.43 ms | 18.97 ms |
| `HT71` (71 blocks) | `960x544` | `1920x1080` | 351 | 30.11 ms | 30.20 ms |
| `HT71` (71 blocks) | `1280x704` | `1920x1080` | 351 | 49.41 ms | 58.22 ms |

Auxiliary adapter passes at `1920x1080`:
- `OpticalFlowSynthesizer`: ~0.46–1.19 ms median GPU time (depending on GPU P-state), zero CPU readback.
- `Context::encode_compose_delta()`: ~0.15–0.35 ms GPU time on the display queue in async worker mode (`NETAL_NR_ASYNC=1`).

### Scoped live Metal 4 drawable telemetry (`NETAL_NR_HOOK_DRAWABLE=1`, `NETAL_NR_ASYNC=1`)

In a local two-binary native Metal 4 game host session on Apple M4 Max (`macOS 27.2`,
`NEURALSTUDIO_TARGET_EXE` child-process gate active, `HT71` `F8_E4M3` checkpoint,
`NETAL_NR_SCALE=0.375` -> `1280x832` internal network composited onto a `3456x2168`
Retina `CAMetalDrawable` via `MTL4CommandQueue signalDrawable:`):
- Parent launcher process remained unhooked; child Metal 4 render host hooked
  `AGXG16XFamilyCommandQueue_mtlnext` cleanly and completed 180 consecutive
  published NR display frames with zero fallbacks after two initial async warm-up
  frames (`exit_code = 0` on `SIGTERM`).
- Background `HT71` + optical-flow worker pass (`1280x832`): ~`56.3–60.9 ms` GPU time
  (86 completed neural updates across 180 display presents).
- Display-queue delta composition + `RenderTarget` fullscreen writeback (`3456x2168`):
  ~`4.38–12.44 ms` whole helper command-buffer GPU time.
- Scope note: this establishes technical execution of the opt-in Metal 4 drawable
  seam on one machine and title, not universal game compatibility, HUD separation,
  or NVIDIA RTX parity.

## Opt-in adapter environment variables

| Variable | Values / Default | Purpose |
| --- | --- | --- |
| `NETAL_NR_ADAPTER_MODE` | `off` (default), `continuous`, `once`, `delayed_once` | Master gate; `off` exits constructor with zero hooks |
| `NETAL_NR_MODEL` | Path to `.dlssmodel` or `.safetensors` | Local NR checkpoint path |
| `NETAL_NR_SYNTHETIC` | `small` or `ht` (unset by default) | Weight-free synthetic graph for CI/diagnostics |
| `NETAL_NR_SCALE` | `[0.25, 1.0]`, default `1.0` | Internal network scale factor (rounded to multiple of 64) |
| `NETAL_NR_PROFILE` | `standard` (default), `natural`, `cinematic`, `neutral` | Style, local tone, and local structure preset |
| `NETAL_NR_DETAIL` / `NETAL_NR_COLOUR` | `[0, 8]` / `[0, 4]`, default `1.0` | High-frequency detail vs low-frequency colour gain |
| `NETAL_NR_INTENSITY` | `[0, 1]`, default `1.0` | Overall neural delta blend factor |
| `NETAL_NR_ASYNC` | `0` (default) or `1` | `1` runs U-Net on a background queue and composes cached deltas on display frames |
| `NETAL_NR_HOOK_SPATIAL` | `0` (default) or `1` | Intercept `MTLFXSpatialScaler` |
| `NETAL_NR_HOOK_DRAWABLE` | `0` (default) or `1` | Intercept `CAMetalLayer` + Metal 4 `MTL4CommandQueue signalDrawable:` |
| `NEURALSTUDIO_TARGET_EXE` | Comma-separated basenames (optional) | Restrict hook installation to matching child executables |
| `NETAL_NR_LOG` | File path (optional, fallback `stderr`) | Structured `key=value` telemetry log; `<log_dir>/netal-nr-pause` pauses NR live |
| `NETAL_NR_CAPTURE_PAIR_DIR` + `NETAL_NR_CAPTURE_FRAMES` | Dir + CSV frame IDs (off by default) | Async diagnostic `raw_<frame>.png` / `nr_<frame>.png` pair dump |

`scripts/netal-native.sh` also accepts `--nr-experimental --nr-bridge-dir DIR --nr-model PATH [--nr-scale FLOAT] [--nr-inline]` for eligible native `.app` bundles without altering default SR/FG launches.

## Scope limits and provenance

- **Hardware FP16 matrix math vs RTX parity**: `src/nr_fast/` matches `MLX-DLSS`
  on Apple Silicon `simdgroup_half8x8`, not NVIDIA RTX tensor-core accumulation
  bit-for-bit. Exact scalar/block RTX oracle probes remain in `src/nr_swin/`.
- **Async delta reuse trade-off**: `NETAL_NR_ASYNC=1` keeps display composition
  under ~0.4 ms so a 60 Hz render loop does not stall on a 16–30 ms U-Net pass,
  at the cost of applying the most recently completed neural detail delta across
  1–2 display frames during fast motion.
- **Post-UI drawable hook**: When `NETAL_NR_HOOK_DRAWABLE=1` intercepts final
  `CAMetalDrawable` presentation on color-only games without engine upscaler
  seams, HUD/UI elements rendered into the same drawable are also processed by
  NR. Engine-level `MTLFXTemporalScaler` / `MTLFXSpatialScaler` seams run before
  UI composition whenever the game exposes them.
- **Provenance**: Portions of `src/nr_fast/` are adapted from WaveCut's
  `iamwavecut/MLX-DLSS` (Apache-2.0, commit `7870371e8bf6c03967213f1a5d40db6c5a3bee38`);
  see `LICENSES/MLX-DLSS-Apache-2.0.txt` and `THIRD_PARTY_NOTICES.md`. Model
  weights remain external user-supplied inputs and are never redistributed here.
