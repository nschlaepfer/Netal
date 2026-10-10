# Standalone DLSS Frame Generation host

This directory contains an isolated, non-interactive D3D12 harness for the
hash-pinned `nvngx_dlssg.dll` 310.8.0 runtime. It is an execution-oracle seed,
not a reimplementation and not evidence of Mac parity.

The default run is the smallest 2x contract: RGBA8 backbuffer, RG16F motion,
R32F depth, RGBA8 interpolated output, one reset/warmup rendered frame, and one
non-reset rendered frame with `MultiFrameCount=1` / `MultiFrameIndex=1`.
Inputs are deterministic and every input/output is written and SHA-256 hashed.
The second output is accepted only if evaluation succeeds, the selected output
region is nontrivial, and it differs from both adjacent rendered inputs.

Build on Windows with the public NVIDIA DLSS headers:

```powershell
.\scripts\build-fg-standalone.ps1 `
  -NgxInclude D:\path\to\NVIDIA-DLSS\include `
  -RuntimePath .\runtime\streamline-2.13\nvngx_dlssg.dll `
  -BuildDirectory D:\CodexLab\<new-job>\build
```

Run the minimal 2x contract into a new or empty evidence directory:

```powershell
& D:\CodexLab\<new-job>\build\neuralstudio-fg-standalone.exe `
  --arm-execution `
  --runtime D:\CodexLab\<new-job>\build\nvngx_dlssg.dll `
  --thunk D:\CodexLab\<new-job>\build\nvngx.dll `
  --output-dir D:\CodexLab\<new-job>\run-core-2x `
  --width 640 --height 360 --rendered-frames 2
```

Optional feature switches are explicit so an unsupported combination fails
instead of silently degrading:

- `--hudless`
- `--ui-alpha` or `--ui-color-alpha` (requires `--hudless`)
- `--bidirectional-distortion` (zero, signed RGBA16F half-resolution field)
- `--output-real`
- `--disable-interpolation-output`
- `--menu-detection`
- `--reset-every-frame` (diagnostic; normally prevents acceptance)
- `--subrect-x`, `--subrect-y`, `--subrect-width`, `--subrect-height`
  mirror the selected modern extent into runtime 310.8's deprecated shared
  backbuffer-subrect quartet. A smaller or nonzero-origin extent also enables
  create-time dynamic resolution and uses the selected dimensions as the
  Feature-11 create and internal render size. The backing textures retain their
  full physical dimensions; receipts record resource and feature-create sizes
  separately.
- `--multi-frame-count 1..3`, `--multi-frame-index 1..3`
- `--output-poison-byte 0..255` (default `165`; repeat with another value and
  require identical final hashes before treating output coverage as accepted)
- `--nvapi-proxy C:\absolute\path\nvapi64.dll` preloads a scoped trace proxy
  before D3D12 device creation and rejects any resolved path or hash mismatch.
- `--capture-runtime-allocations` installs the official NGX D3D12 allocation
  and release callbacks. It requires
  `--expected-baseline-output-sha256 <prior-frame-1-hash>` and rejects callback
  errors, double releases, leaks, or output drift. Address-free descriptor and
  extent records go to `runtime-allocations-semantic.jsonl`; GPU virtual
  addresses and allocation sizes go only to
  `runtime-allocations-volatile.jsonl`.
- `--capture-runtime-upload-buffers` additionally snapshots still-live
  CPU-visible BUFFER allocations after Feature-11 creation is fenced and
  before evaluation. It requires runtime-allocation capture and the same
  baseline-output hash. DEFAULT resources are never mapped. Tokenized dumps
  are sealed read-only beneath `runtime-upload-buffers/`; the address-free
  `runtime-upload-buffers-manifest.json` records exact logical byte counts and
  SHA-256 values. Captured bytes are never written back to execution.
  The generic host additionally requires explicit expected buffer-count and
  logical-byte gates. The pinned 640×360 core runner supplies 67 buffers and
  4,868,576 bytes.
- `--capture-runtime-default-buffers` is a separate experiment requiring
  allocation capture, baseline-output validation, an explicit comma-separated
  allocation-token list, and exact count/byte expectations. Live BUFFER
  resources created on DEFAULT heaps in COMMON and single-subresource DEFAULT
  textures are supported. Buffers use D3D12's implicit
  COMMON-to-COPY_SOURCE promotion/decay; textures use an explicit transition
  from and back to their allocation-time state. Both paths copy into dedicated
  READBACK resources and never map DEFAULT memory directly.
  `--runtime-default-buffer-capture-point` must explicitly select
  `post_create`, `post_evaluate_0`, or `post_evaluate_1`; the last option
  requires at least three rendered frames so its fenced copy remains between
  the first generated evaluation and the next live evaluation. Dump and
  manifest names are distinct. Dumps are never fed back into execution.
- `--hdr` is intentionally rejected until a correct pinned 32-bit HDR texture
  contract is implemented.

For the richest current 2x resource profile, add:

```text
--hudless --ui-alpha --bidirectional-distortion --output-real --disable-interpolation-output
```

Successful execution writes `fg-run-receipt.json` plus raw resource files. A
bounded, control-character-escaped verbose NGX callback stream is persisted as
`ngx-callback.log`; the callback remains installed through feature release and
runtime shutdown, and its hash is included in the receipt. Native NGX file
logging remains enabled under `ngx-data/` for independent diagnostics.

A successful process is evidence only that the pinned NVIDIA runtime evaluated
the deterministic D3D12 contract; it makes no graph-recovery, quality,
performance, cross-platform, or product-integration claim.
