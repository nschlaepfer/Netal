# Experimental native iOS SR

Updated **2026-11-08**. This port builds Netal's existing DLSS 4 Preset K SR
engine for physical iOS and iPadOS devices with **A19 Pro or A20 Pro**. It also
provides an unsigned model-execution test app and an opt-in, approximate Metal
Performance Primitives (MPP) GEMM backend. It does not include a game integration,
Frame Generation, a DLSS 4.5 implementation, or a demonstrated 2–3 ms latency.
An additional experimental A20 Pro/iOS 27 path uses derived E4M3FN weights and
FP8-input TensorOps; it keeps the original model and default path available.

**A physical A20 Pro has completed the procedural model test.** The isolated
SR timings below do not establish image quality, Neural Accelerator instruction
execution, sustained game performance, or a frame-rate improvement. Mac results
in the other documents remain separate evidence.

## Hardware and API scope

The mobile gate identifies the GPU's chip, rather than requiring a product name
containing “Pro.” Apple's current specifications therefore include these
hardware candidates:

| Products | Chip | GPU configuration documented by Apple |
| --- | --- | --- |
| iPhone 17 Pro and Pro Max | A19 Pro | 6 cores with Neural Accelerators |
| iPhone Air | A19 Pro | 5 cores with Neural Accelerators |
| iPhone 18 Pro and Pro Max | A20 Pro | 7 cores with Neural Accelerators |
| iPhone Duo | A20 Pro | 7 cores with Neural Accelerators |
| iPad mini with A20 Pro | A20 Pro | 7 cores with Neural Accelerators |

Sources: [iPhone 17 Pro announcement, 2025-09-09](https://www.apple.com/newsroom/2025/09/apple-unveils-iphone-17-pro-and-iphone-17-pro-max/),
[Air specifications](https://www.apple.com/iphone-air/specs/),
[iPhone 18 Pro specifications](https://www.apple.com/iphone-18-pro/specs/),
[Duo specifications](https://www.apple.com/iphone-duo/specs/), and
[iPad mini specifications](https://www.apple.com/ipad-mini/specs/), accessed
2026-11-08. This table identifies hardware candidates, not tested products. In
particular, different A20 Pro iPad mini storage tiers have different Neural
Engine configurations; the gate does not require a particular Neural Engine
core count.

These GPU Neural Accelerators and the separate Neural Engine are different
hardware. A19 Pro, A20 Pro and M5 share the concept of GPU matrix acceleration;
this does not establish equal throughput, bandwidth, power limits or sustained
performance. Apple documents M5 with a 10-core GPU, while the mobile products
above have fewer GPU cores. [M5 announcement, 2025-10-15](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/).
MPP TensorOps is the public GPU route used by the candidate here. This port
does not invoke the Neural Engine or convert the model to Core ML. Apple's
[M5/A19 GPU ML Tech Talk](https://developer.apple.com/videos/play/tech-talks/111432/)
explains that TensorOps uses available GPU acceleration and can fall back to
shader implementations on older GPUs; API use alone does not prove a specific
hardware execution path. Accessed 2026-11-08.

The capabilities check in
[`netal_ios_sr_capabilities.mm`](../src/ios_host/netal_ios_sr_capabilities.mm)
requires a physical iOS/iPadOS target, iOS/iPadOS 26+, an identified A19 Pro or
A20 Pro GPU, Metal 4, Apple10-or-later GPU capabilities and a successful small
FP16 tensor allocation. The SDK-aware implementation also checks Apple11 on
iOS 27+. Simulator and Catalyst targets are excluded. The exact normalized chip
label must be `A19 Pro` or `A20 Pro`, optionally surrounded by `Apple` and `GPU`;
unknown labels and other chips fail closed. It does not use a device-SKU list,
GPU core count or Apple10 membership alone as chip identity. This is a
compatibility policy based on OS-provided GPU names, not hardware attestation.
The existing native Mac path has its own policy and remains separate.

| Capability | Availability and meaning |
| --- | --- |
| Metal 4, `MTLTensor`, FP16 MPP candidate | iOS/iPadOS 26+ and the checks above. This is the current SR arithmetic route. |
| GPU Neural Accelerator candidate | Apple10 and later hardware. Family membership plus chip admission is an eligibility signal, not a verified dispatch. |
| FP8 E4M3/E5M2 tensor types | iOS/iPadOS 27+ with a sufficiently recent SDK and successful allocations. The capability flag proves small allocations only. |
| Quantized FP8 SR | Experimental A20 Pro/iOS 27 mixed-precision candidate. Explicit source-derived pack, native FP8-input TensorOps, per-row GPU quantization and separate quality/performance evidence are required. Allocation alone is insufficient. |

Apple's [`MTLTensor` documentation](https://developer.apple.com/documentation/metal/mtltensor)
records iOS/iPadOS 26 availability. The
[inline ML sample](https://developer.apple.com/documentation/metal/running-inline-ml-operations-in-a-shader-with-metal-4)
documents Apple10-and-later GPU Neural Accelerators. The
[WWDC26 tensor session](https://developer.apple.com/videos/play/wwdc2026/330/)
covers the iOS 27 quantized tensor additions. These API references were checked
2026-11-08 against the iPhoneOS 27 SDK. Building with a newer SDK does not change
an older device's runtime availability.

## Model identity and arithmetic modes

The port uses the existing, genuine NVIDIA model data supported by Netal's
reconstructed SR engine. It is not a substitute neural network labeled DLSS.
Supply your permitted local inputs using [model setup](MODEL_SETUP.md). The
complete identity contract is:

| Input | Identity |
| --- | --- |
| Source DLL | `nvngx_dlss.dll`, file version `310.8.0.0` |
| DLL SHA-256 | `c85f971ce023c9f3492fc7455f0b01a24ba18ea39636407a846902c4360b0b7e` |
| Model | `DLSS_SR_Feature1_Quality_K` |
| `weights.bin` | 3,545,824 bytes; SHA-256 `ddaa59cd82deed53bad9d65b82b454f1439fc4875e4220b3ffafafb6cd0296a1` |
| `constants.bin` | 524,288 bytes; SHA-256 `305f4c6773d1cb333b577094e28a46beb1867898716d8290121862a2cb06e773` |

These are the same inputs pinned in the [public README](../README.md) and
[`netal_sr_runtime_manifest.py`](../scripts/netal_sr_runtime_manifest.py).
The full hashes are authoritative; a DLL name, marketing version or Preset K
label is insufficient. A newer DLL is not automatically compatible. No DLSS 4.5
model or output has been validated by the iOS port.

NVIDIA's latest published SDK checked on 2026-11-08 is
[310.9.16, released 2026-10-16](https://github.com/NVIDIA/DLSS/releases/tag/v310.9.16).
Its Windows/Linux ARM builds do not supply an iOS/Metal runtime or portable
DLSS 4.5 model. CPU instruction-set compatibility alone does not replace the
RTX/NGX runtime contract. This port keeps its previously identified model rather
than treating that SDK release as a validated model upgrade.

The engine has three distinct routes:

- **Reference:** retained translated per-window kernels. Historical Mac oracle
  matches are limited to the documented frames, profiles and arithmetic mode;
  they do not establish iOS parity or universal exactness.
- **Batched:** the existing public PWin pipeline. Missing pipeline layers can
  fall back to the Reference implementation. The probe records coverage and
  fallback status; a successful session alone does not establish a complete
  batched run.
- **MPP candidate:** replaces supported GEMM dispatches with public TensorOps
  matrix multiplication. Other PWin operations remain the existing kernels.
  FP16 handoffs and FP32 accumulation do not reproduce the Reference reduction
  order. This route is explicitly approximate and opt-in; compilation or a
  synthetic GEMM match is not full-model or temporal-quality validation.

See [arithmetic modes](SR_ARITHMETIC_MODES.md) and
[source-origin review](SOURCE_PROVENANCE_REVIEW.md). The mobile build does not
resolve the existing distribution-rights review for retained translated inputs.
NVIDIA DLLs, extracted model bytes, SDK headers, runtime bundles and private
scalar-contract tables are never supplied by these commands or included in the
public source. Keep local inputs and generated assets out of commits and PRs.

## Explicit arithmetic experiments

The default Reference/Batched paths retain their existing contracts. Approximate
arithmetic is opt-in and its NVIDIA model identity remains DLSS 4 K, not 4.5.
Kernel-builder variants include `staged`, `direct`, `direct-mlp`, `fast-half`,
`fast-half-mlp`, `fp8` and `fp8-mlp`; inspect each build receipt and use matching
sidecars rather than changing a name in configuration.

- `direct-mlp` adds FP16 TensorOps expansion/GELU/contraction while preserving its
  documented rounded intermediates. Six MLP dispatches use 64-token/128-thread
  tiles and need matching grids.
- `fast-half-mlp` uses one full-K FP16-input/FP32-accumulation GEMM, omits original
  K16 FP16 handoffs, then applies real-unit bias/residual. This is a deliberate
  arithmetic relaxation, not reference equality.
- `fp8-mlp` uses 40 E4M3-input GEMMs, 40 GPU activation quantizers, two sensitive
  FP16 exceptions and six FP16 TensorOps MLPs. Norms, attention, biases, residual
  work, boundaries and history retain their declared existing precision.

A reproducible fast-half sidecar build uses the separate generator:

```sh
python3 scripts/build-netal-sr-mpp-kernels.py \
  --sdk iphoneos --deployment-target 26.0 --variant fast-half-mlp \
  --output build/ios-fast-half-kernels
python3 scripts/build-netal-sr-fast-pipelines.py \
  --bundle '/path/to/private/source-runtime' --variant fast-half-mlp \
  --output build/ios-fast-half-sidecars
```

Copy generated `pipeline/` and `profiles/` JSONs into a **fresh copy** of the
source runtime, and copy `pwin_mpp_all.metallib` into its `metallib/` directory.
Preserve the build/generator receipts, then reseal with
`python3 scripts/seal-netal-sr-runtime.py NEW_RUNTIME --replace`. Matching grids
are required: replacing the MLP shader while retaining its old 256-thread grids
is unsupported. The generator does not modify or copy model weights.

The FP8 pack builder deep-verifies the pinned source runtime and canonical
public graph, then writes separate private payloads. It transposes and pads
weights, uses per-output-column FP32 dequantization scales, and records exact
E4M3FN ties-to-even/saturation semantics. Two sensitive tensors (`enc0.W1`,
`dec0.W7`) stay FP16. The pack contains 52 quantized and two retained matrices;
12 packed MLP matrices are reserved for future work and are not used by the
current mixed runtime. All original blobs remain present for non-GEMM work and
fallback, so total resident memory does not automatically halve.

The aligned pack occupies 2,049,536 bytes versus 3,158,016 active FP16 matrix
bytes. Quantized-weight reconstruction relative RMS is 2.56953%; this is not
image error. Per-row activation scaling and quantization are real GPU work and
must be included in full-pass timings. The native shader consumes FP8 format
operands directly; the host cannot establish the driver's final instruction
precision merely from that API choice.

To build the separate mixed FP8 experiment:

```sh
python3 scripts/build-netal-sr-fp8-pack.py \
  --runtime '/path/to/private/source-runtime' --output build/private-fp8-pack
python3 scripts/build-netal-sr-mpp-kernels.py \
  --sdk iphoneos --deployment-target 27.0 --variant fp8-mlp \
  --output build/ios-fp8-kernels
python3 scripts/build-netal-sr-fp8-pipelines.py \
  --bundle '/path/to/private/source-runtime' \
  --pack build/private-fp8-pack --output build/private-fp8-sidecars
```

Use another fresh source-runtime copy. Copy the pack's `pipeline/fp8/` payloads
into that same relative path, its `manifest.json` to `pipeline/fp8/pack.json`,
the generated `pipeline/` and `profiles/` sidecars to their matching locations,
and the new metallib to `metallib/pwin_mpp_all.metallib`. Preserve receipts and
reseal. The probe's `fp8-candidate` admission requires the supported mixed
`fp8-mlp` graph; the separate original-MLP ablation is not admitted by that
validator. All pack data and generated libraries are private local outputs.

A model-free FP8 lab executes E4M3/E5M2 matrix operations and compares CPU-oracle
math, packing, canaries and valid GPU timings on A20 Pro/iOS 27. It is a separate
operator experiment and does not load the DLSS model. The ordinary SR library
and probe still build for iOS 26 without this optional lab.

## Build the physical-device library and local runtime

Use an Apple-silicon Mac, Python 3.10+, Xcode with the iPhoneOS 26+ SDK and the
matching Metal compiler. Use Xcode/iPhoneOS 27+ when investigating the FP8
capability flag. The SR builders require the physical `iphoneos` SDK; simulator
compilation is not a substitute. Run these commands from the repository root,
with fresh output directories:

```sh
python3 scripts/build-netal-ios-sr.py \
  --sdk iphoneos --deployment-target 26.0 \
  --output build/ios-sr-core

python3 scripts/build-netal-sr-assets.py \
  --sdk iphoneos --deployment-target 26.0 \
  --model '/path/to/local-supported-model' \
  --resolutions 1280x720 \
  --output build/ios-sr-assets

python3 scripts/assemble-netal-sr-runtime.py \
  --assets build/ios-sr-assets \
  --model '/path/to/local-supported-model' \
  --scalar-contract '/path/to/local-scalar-contract' \
  --output build/ios-sr-runtime
```

The core output contains `lib/libNetalIOSSR.a`, public headers, separately
compiled kernel sets and a build receipt. Link Foundation, Metal and libc++;
`-ObjC` is recommended when linking the static library. The asset builder
targets each Metal library explicitly for iOS, creates the selected output
profiles, and retains a 1920×1080 compatibility profile required by the current
assembly layout. A profile's name is its output size, not its input render size
or the phone's display resolution. Read the actual session dimensions from the
probe receipt.

Assembly requires four caller-supplied scalar tables:
`lg2-input-table.bin`, `lg2-output-table.bin`, `ex2-input-table.bin` and
`ex2-output-table.bin`. Each contains 1,048,576 uint32 entries. The assembler
checks the source-asset inventory and supported model, records the scalar
identities, and seals a v2 runtime manifest. Sealing is an integrity inventory,
not numerical correctness or permission to redistribute the assembled bundle.
See [complete source build](COMPLETE_SOURCE_BUILD.md).

To create a separate MPP experiment, preserve the original assembled runtime
and regenerate all profile sidecars in a copy:

```sh
python3 scripts/build-netal-sr-mpp-kernels.py \
  --sdk iphoneos --deployment-target 26.0 \
  --output build/ios-sr-mpp-kernels

cp -R build/ios-sr-runtime build/ios-sr-runtime-mpp
cp build/ios-sr-mpp-kernels/pwin_mpp_all.metallib \
  build/ios-sr-runtime-mpp/metallib/pwin_mpp_all.metallib
python3 tools/pwin_pipeline/build-profile-pipelines.py \
  build/ios-sr-runtime-mpp \
  --layout-dir build/ios-sr-runtime-mpp/pipeline/layouts \
  --gemm-backend mpp
python3 scripts/seal-netal-sr-runtime.py build/ios-sr-runtime-mpp --replace
```

The kernel builder also accepts `--variant direct` (the default is `staged`).
This keeps the same library/function/descriptor ABI while selecting direct device
tensors and cooperative multiply-accumulate. Use a fresh output/runtime copy and
record the library hash; the two variants are not interchangeable evidence. The
direct route follows [Apple's MPP guide, 2026-03-16](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf).

The converter checks the supported GEMM ABI, shapes and dispatch geometry.
The selected profile needs all eleven layers and the shifted temporal phase
sidecars. A partial conversion must not be reported as an MPP result. The probe
checks these descriptors, but descriptor validation does not confirm that the
GPU executed a particular hardware instruction.

## Build and run the standalone probe

The app contains procedural input generation and the engine, with **zero
embedded model assets**. It does not launch or modify a game. Build it unsigned:

```sh
python3 scripts/build-netal-ios-sr-probe.py \
  --library-root build/ios-sr-core \
  --sdk iphoneos --deployment-target 26.0 \
  --output build/ios-sr-probe
```

For an Xcode project, install XcodeGen and add `--prepare-xcode` with a fresh
output directory. The optional `--team-id` is supplied locally by the caller
for automatic signing; no developer identity or provisioning profile is built
into this repository. A generated project without a team defaults to signing
disabled. Signing and installing on a permitted physical device are separate
steps, not results of the unsigned build.

After installing a signed app, copy your complete iOS runtime into the app's
`Documents/runtime`. File Sharing is enabled. Add `Documents/benchmark.json`:

```json
{
  "profile": "1280x720",
  "backend": "batched",
  "metal_api": "metal4",
  "warmup_frames": 3,
  "measured_frames": 60,
  "save_last_output": false
}
```

Use `reference`, `batched`, or `mpp-candidate` for `backend` (`mpp` is an alias
for the last). MPP selection also requires the converted runtime above; changing
the JSON alone does not replace the loaded kernels. `metal_api` accepts `metal3`
or `metal4` command submission. This selector does not lower the engine's iOS
26/Metal 4 capability requirement. Allowed counts are 1–16 warm-up frames and
1–600 measured frames. The bounded run uses one frame in flight, a 30-second
per-frame timeout and a 180-second run deadline.

The app uses the scene lifecycle required by iOS 27. It stops before the next
frame when cancelled, inactive, critically hot or past its run deadline. In-flight
resources remain retained until real GPU completion, including after a timeout.
To compare outputs privately, set `save_last_output` to `true`; it writes the
last completed frame to `Documents/probe-output.rgba16le` after measurements.
This is opt-in and does not affect the measured GPU scope.

Tap **Run**. The app writes `Documents/probe-receipt.json`, including partial
progress during a run. Inspect its final state, model/root identity, selected
pipeline metadata, layer coverage, output canaries, finite-output checks,
thermal states and timing validity before interpreting the numbers. Successful
output writes and intact canaries are execution smoke checks, not an independent
image-quality oracle. The procedural translating HDR plane includes non-zero
motion, inverted depth, exposure and an 18-phase jitter sequence; it is not
representative gameplay.

Source-only checks require no model or physical device:

```sh
python3 scripts/test-netal-ios-sr-device-policy.py
python3 scripts/test-netal-ios-sr-build.py
python3 scripts/test-netal-ios-sr-probe.py
python3 scripts/test-netal-sr-mpp.py
```

Passing these tests is not physical-device execution evidence. Record the exact
source revision, SDK/compiler and generated build/runtime hashes with any later
device result. Do not publish personal paths, device identifiers, signing
identities, provisioning profiles, private captures or input assets in receipts.

## Temporal input contract for a game adapter

A temporal SR adapter must supply all of the following from the same rendered
frame. The current test uses the buffer API in
[`dlss_sr_session.h`](../src/macos_host/dlss_sr_session.h):

| Input or state | Required contract |
| --- | --- |
| Color | Linear HDR scene color at the session input size, RGBA16F, before display tone mapping and UI composition. Color/exposure conventions must match the adapter. |
| Motion | RG16F at input size, with a verified current-to-previous mapping and sign, units, jitter convention and coordinate orientation. The probe uses normalized vectors and scales them by input width/height. |
| Depth | R32F at input size with a correct, documented depth convention. The sustained validated-contract route requires reversed depth; normal-depth kernel compilation does not establish temporal support. |
| Exposure | One R32F exposure value for the frame. The current texture adapter requires `preExposure == 1`; a missing exposure texture defaults to 1 only when that is correct for the source scene. |
| Jitter | The actual subpixel projection jitter used to render the frame. The default buffer route accepts its known jitter set; changing validation gates does not establish correctness for another sequence. |
| Geometry and output | Actual input content size and matching output profile; RGBA16F output at session output size. Validate dynamic-resolution and viewport changes rather than treating a display-size label as the render geometry. |
| History and resets | A persistent session with ordered frames. Start with an accepted reset frame; reset on cuts, skipped/fallback frames, changed geometry or other history discontinuities. Repeating reset-only inference is not a temporal integration. |
| GPU ownership | Compatible resources on the same Metal device, valid buffer sizes/offsets and lifetime through completion. Respect producer/consumer synchronization. For Metal 4, the caller supplies residency for passed frame buffers. |

The texture adapter adds import/export and accepts only its implemented formats
and scalar contracts; inspect
[`dlss_sr_texture_adapter.h`](../src/macos_host/dlss_sr_texture_adapter.h) and its
implementation before connecting a new renderer. The standalone probe is
serial; overlapping frames and resets need separate synchronization validation.
Do not fabricate zero motion/depth, inject jitter that the game never rendered,
or use the final tone-mapped frame as a replacement for missing temporal inputs.

**The current OoT emulator color-only presentation hook is insufficient.** A
final color texture alone does not provide motion, depth, exposure, projection
jitter or reliable temporal resets. No working Netal temporal SR integration
with OoT is established by this port. Those resources must first be identified
and passed correctly at the scene-rendering stage; a color-only resize or
sharpening pass must be labeled by its actual method. The existing game and its
saves should remain independently usable while that adapter is developed.

## Optional operator, MetalFX and capture builds

The ordinary probe remains model-free until a private runtime is supplied.
Optional modules can be linked explicitly:

```sh
python3 scripts/build-netal-ios-fp8-lab.py --output-dir build/ios-fp8-lab
python3 scripts/build-netal-ios-metalfx-lab.py --output build/ios-metalfx-lab
python3 scripts/build-netal-ios-sr-probe.py \
  --library-root build/ios-sr-core --deployment-target 27.0 \
  --fp8-lab-root build/ios-fp8-lab \
  --metalfx-lab-root build/ios-metalfx-lab \
  --output build/ios-comparison-probe
```

FP8 requires SDK/iOS 27 and MSL 4.1. These optional modules contain procedural
fixtures and code, never model weights. The builder verifies the module/header/
shared-stimulus identities and copies the model-free FP8 metallib when requested.
Use `--prepare-xcode` and caller-supplied signing settings for device installation.
The default unsigned build does not install or sign an app.

`probe_mode` defaults to `sr`. `fp8_lab` runs the synthetic matrix/packing tests
before any model is loaded. `metalfx_spatial` and `metalfx_temporal` run the
standalone Apple scalers, likewise without a Netal model session. Check mode,
actual dimensions, final state, finite/guard checks and timing scope in every
receipt. The [research note](IOS_UPSCALING_RESEARCH.md) records public contracts
and actual MetalFX/MLX/S2Arm findings.

Capture uses a **separate** build with `--enable-metal-capture` and MetalFX
configuration `capture: true`. It records one warmed frame to a private
`.gputrace`. Capture-enabled builds refuse normal SR/operator/scaler performance
runs. Restore an ordinary build before comparing latency. No capture resource
or raw output belongs in this repository.

### Boundary experiments and diagnostics

`fast_input_boundary: true` selects the existing separate inverse-depth native
math input library. `fast_output_boundary: true` requires a separately built
approximate output library:

```sh
python3 scripts/build-netal-sr-boundary-kernels.py \
  --sdk iphoneos --deployment-target 26.0 \
  --only hiluma_output_depthinv_exact --fast-output \
  --output build/ios-fast-output
```

Copy only the fast metallib into a separate runtime and reseal that runtime. The
exact output library remains unchanged. The optional macro changes selected
math/sampling helpers while retaining the kernel body, history updates, ABI,
bounds and CUDA gather ordering; it is a deliberate numerical approximation.
Receipts report actual Core-selected library names and inventory hashes.

`diagnostic_output_repeat` accepts integers 1–8. Values above 1 repeat the output
pass and produce **profiling-only** results, not normal SR latency. Restart the
app between repeat values because Core caches the setting. The probe bounds the
full graph against 160 Metal 4 argument tables. Check every temporal output hash
against the ordinary run before using amplification for attribution.

On the A20 procedural case, 4× and 8× output runs preserved all 136 frame hashes
and suggested approximately 2.48 ms per extra exact-output pass. This is a
controlled diagnostic estimate, not a precise counter-based stage profile.

## Separate render size from output size

NVIDIA reduces the scene's internal rendering resolution before reconstruction.
Typical per-axis ratios are Quality about 66.7%, Balanced about 58%, Performance
50%, and Ultra Performance about 33.3%. For 1920×1080 output, the latter two
mean 960×540 and 640×360 input: respectively one quarter and one ninth of the
native scene pixels, not corresponding guaranteed FPS multipliers. See
[NVIDIA's rendering-resolution explanation](https://developer.nvidia.com/blog/tips-getting-the-most-out-of-the-dlss-unreal-engine-4-plugin/).

The probe accepts an optional explicit `render_size`, independently of the
output `profile`:

```json
{
  "profile": "1920x1080",
  "render_size": "960x540",
  "backend": "batched",
  "metal_api": "metal4",
  "warmup_frames": 16,
  "measured_frames": 120,
  "save_last_output": true
}
```

Omit `render_size` for the profile's existing reference geometry. Requested
input dimensions must be positive, bounded, no larger than the output, and match
its aspect ratio within rounding tolerance. The probe verifies the resulting
session dimensions. Each run constructs a new session and resets its history.
This does not implement NVIDIA's optimal-settings query or dynamic-resolution
integration in a game.

These are **exploratory input ratios**, using the existing supported K model.
NVIDIA recommends 18/24/32/72 jitter phases for Quality/Balanced/Performance/
Ultra respectively; Netal's current temporal gate accepts its known set of 18
captured positions. The ratio experiments retain that gate and sequence rather
than silently enabling arbitrary jitter. They do not certify the corresponding
NVIDIA modes or establish quality with 32/72-phase sequences.
[Current NVIDIA guide, §3.7.1.1](https://github.com/NVIDIA/DLSS/blob/9981c6350dc58c0ce21f024fd65e31e851c31689/doc/DLSS_Programming_Guide_Release.pdf).

At a fixed output, Netal's dense network dimensions and arenas still follow the
output profile. Lowering scene rendering can save game work without lowering the
same-output neural pass proportionally. Test reduced rendering plus SR,
import/export and presentation against the original complete frame. Enlarging
output alone does not create a performance gain.

The ratio probe uses a versioned normalized-world stimulus: fixed spatial
frequencies and normalized motion, with jitter in actual input-pixel units.
This keeps scene extent and motion consistent across input sizes. Measurements
from this stimulus are separate from the initial pixel-based checkpoint below.

### Controlled input-ratio measurements

A fresh normalized-world stimulus v2 run on the A20 Pro used the same Batched
model, 16 warm-up plus 120 measured frames per case, and nominal thermal state.
Baseline repeats bracketed each output group; their medians stayed within 0.03 ms.

| Output | Actual input | Source-pixel fraction | SR GPU p50 |
| --- | --- | ---: | ---: |
| 1280×720 | 854×480 | 44.48% | 13.316 ms |
| 1280×720 | 640×360 | 25.00% | 12.535 ms |
| 1280×720 | 426×240 | 11.09% | 12.363 ms |
| 1920×1080 | 1280×720 | 44.44% | 23.011 ms |
| 1920×1080 | 960×540 | 25.00% | 22.136 ms |
| 1920×1080 | 640×360 | 11.11% | 21.340 ms |

These are the known-18-jitter exploratory ratios described above. The measured
SR cost falls modestly; the potentially larger benefit is avoided scene-rendering
work, which this standalone test excludes. No game FPS or equivalent perceptual
quality is established by this table.

## Recorded isolated A20 Pro measurements

On 2026-11-08, a physical iPhone 18 Pro Max (A20 Pro GPU), iOS 27.2,
ran the supported DLSS 4 K model using the iPhoneOS 27.0 SDK build and Metal 4
submission. Each row below used eight warm-up frames and sixty measured frames,
one frame in flight, identical procedural input generation, and nominal thermal
state at the start and end. These short runs do not establish sustained thermal
behavior. No external cooler was used.

| Input → output | Actual pipeline | GPU p50 | GPU p95 | GPU p99 |
| --- | --- | ---: | ---: | ---: |
| 854×480 → 1280×720 | Batched | 13.334 ms | 13.607 ms | 13.656 ms |
| 854×480 → 1280×720 | MPP direct candidate | 13.182 ms | 13.394 ms | 13.499 ms |
| 1280×720 → 1920×1080 | Batched | 24.065 ms | 24.233 ms | 24.277 ms |
| 1280×720 → 1920×1080 | MPP direct candidate | 24.001 ms | 24.207 ms | 24.310 ms |

All rows completed 68 temporal frames, loaded all eleven pipeline layers,
passed full-output NaN-poison and guard checks, and obtained real driver timing.
This proves bounded model execution, not RTX parity or correct game integration.
The direct candidate's small differences from Batched are insufficient to claim
a meaningful speed advantage from these single runs. The first staged candidate
was slower (about 14.75 ms at 720p and 26.65 ms at 1080p); it remains an explicit
experiment. The existing Batched default has not been replaced.

The final direct library SHA-256 is
`29d1339df3d36442886244298a17834cf0aceb91439999731469e82bf8b637da`;
the staged library is
`30112df6071def53816f07e1e33e27ddddb642b2f4b62b1f8ada157443b6ca90`.
The signed procedural probe executable SHA-256 is
`c72d5a17676cf3725ff670eff43ba7f841732e01a4b29a7e22c8846bdf3dd0b9`.
These identify private locally built artifacts, which are not distributed.

For the same last temporal frame at 720p (frame 67), direct MPP versus Batched
had RGB RMS difference `0.0001326735` and maximum absolute difference
`0.0029296875` over 2,764,800 RGB components. Both ranges were approximately
`0.0021763–2.0`, with no nonfinite components. This compares two approximate
routes on one procedural scene; it is not a visual-quality threshold, a claim
of bit-exactness, or an independent NVIDIA oracle. The guard-only final direct
revision reproduced every frame hash from the previous direct run.

The full-pass timings exceed 2–3 ms. Even 13 ms consumes most of a 16.67 ms
60-FPS frame budget before rendering, import/export and presentation. The current
OoT presentation hook still lacks the verified temporal inputs described above;
this test does not enable DLSS in that game. Further work needs measured stage
costs, acceleration of remaining operators, and a correct game adapter.

## Final speed-mode checkpoint

After the explicit arithmetic and boundary experiments, the current A20 Pro
standalone test recorded the following with stimulus v2, 16 warm-up plus 120
measured frames, all 136 frames completed, nominal thermal state, verified actual
boundary library hashes, and capture disabled:

| Actual input → output | Batched baseline p50 | Fast-half + native boundaries p50 | Fast p95 / p99 |
| --- | ---: | ---: | ---: |
| 640×360 → 1280×720 | 12.535 ms | 8.309 ms | 8.385 / 8.409 ms |
| 960×540 → 1920×1080 | 22.136 ms | 14.406 ms | 14.527 / 14.622 ms |
| 640×360 → 1920×1080 | 21.340 ms | 14.354 ms | 14.532 / 14.611 ms |

A bracketed 720p repeat measured 8.305 ms. This is approximately one-third less
SR GPU time than the matched Batched cases, not a measured game-FPS increase.
It is still above the requested 2–3 ms SR target. These are exploratory input
ratios with the known 18-phase contract, not newly certified NVIDIA presets.

The last frame (135) differed from matching Batched output by RGB RMS
0.0002282 at 720p, 0.0002565 at 1080p/2×, and 0.0002846 at 1080p/3×; maximum
absolute differences were respectively 0.0039063, 0.0029297 and 0.0068359. No
nonfinite values were found. This compares one smooth procedural scene and two
approximate implementations; it does not establish RTX parity, broad visual
quality, disocclusion behavior or sustained gameplay.

The selected fast-half library SHA-256 is
`1c3c30df9256f75270f06431346c9ac22bf6c628c99df8a59cf8323ff3e6ccda`;
fast input is
`9134a0bf8ace55a1812735fb807a30e18203c3dda2335571dc963e67282df358`;
fast output is
`e50158bfc8127bc2f4806b95d51b45ddb5fe45fe872e9e50977a77259d927d0c`.
The original exact output remains byte-identical to the pinned earlier build.

The explicit mixed FP8 candidate also passed full model execution and descriptor
admission: 40 FP8 GEMMs, 40 GPU quantizers, two FP16 sensitive exceptions and six
half MLPs, with 137 total per-frame dispatches including boundaries. With exact
boundaries it measured 12.533 ms at 640×360 → 1280×720. With the same fast input/output boundaries as the fast-half candidate, the
fully admitted FP8 run measured 9.730 ms (p95 9.869, p99 9.902), compared with
8.309 ms for fast-half. FP8 is therefore not selected merely for its format name;
packing and all surrounding work must win in the complete path. The model-free physical operator lab passed all 30 records,
but did not prove the driver's final native-FP8 instruction implementation or a
universal throughput advantage.

## What a performance result must measure

The probe keeps these scopes separate:

| Receipt field | Timing scope |
| --- | --- |
| `cold_prepare_wall_ms` | CPU wall time to create the session, load the runtime and prepare pipelines. It is not a complete game shader warm-up. |
| `cold_first_frame` | First reset frame, recorded separately from steady-state summaries. |
| `gpu_full_pass_ms` | Valid driver GPU start/end timestamps for the SR command buffer: input/output boundary transforms, model and history. Excludes game rendering, texture import/export, presentation and CPU procedural input generation. |
| `cpu_encode_ms` | CPU wall time inside the SR encode call; not total frame CPU time. |
| `completion_wall_ms` | Wall time from encode start through GPU completion, including submission and waiting. Excludes procedural input generation and post-completion validation/readback work. |
| `run_wall_ms` | Total bounded loop wall time, including procedural generation, validation and receipt writes. It is not displayed FPS. |

Warm-up frames are excluded from p50/p95/p99 summaries. Invalid or absent GPU
timestamps remain unavailable, never zero-cost samples. The receipt's
`hardware_acceleration_validated`, `visual_quality_validated`,
`rtx_parity_validated`, `gameplay_validated` and `latency_target_validated` flags
remain false: the test has no independent evidence for those claims.

A 2–3 ms target requires a measured, complete SR pass at stated input/output
dimensions, backend, model identity and quality. A GEMM microbenchmark, FP8
allocation, an isolated cold-phone run, or a reciprocal of the SR time is not
a game-FPS result. Use Metal profiling to verify the accelerator route, compare
temporal outputs against an independent oracle, and measure frame times during
the same sustained gameplay with rendering, import/export and presentation
included. Record ordinary phone cooling, power state, thermal changes, memory
pressure and visual defects. Keep cold compilation, warm SR cost and sustained
gameplay separate. No cooler, 45–60 FPS, stutter-free gameplay or higher native
display-resolution result is promised by this experimental port.
