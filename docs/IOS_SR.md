# Experimental native iOS SR

Updated **2026-11-08**. This port builds Netal's existing DLSS 4 Preset K SR
engine for physical iOS and iPadOS devices with **A19 Pro or A20 Pro**. It also
provides an unsigned model-execution test app and an opt-in, approximate Metal
Performance Primitives (MPP) GEMM backend. It does not include a game integration,
Frame Generation, a DLSS 4.5 implementation, or a demonstrated 2–3 ms latency.

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
| Quantized FP8 SR | Not implemented or validated by this port. Allocating FP8 storage does not quantize the model or validate its accuracy. |

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
