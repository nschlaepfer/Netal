# FG NvAPI forwarding trace

This is a capture-only `nvapi64.dll` for the pinned Frame Generation recovery
lane. It forwards the real NVIDIA NvAPI and wraps the 18 D3D12 CUDA interfaces
listed at `work/nvapi/nvapi_interface.h:372-389`, plus the official
`NvAPI_D3D12_CaptureUAVInfo` provenance interface. All other queried interface
IDs are returned unchanged from the matching real dispatcher. Common core,
GPU-architecture, and DRS pass-through IDs retain their official names in the
trace but are never wrapped.

The one resolved ID absent from the pinned public table, `0xAD298D3F`, is
handled by a signature-independent x64 tail sentinel. The generated thunk
preserves all register and stack arguments, counts invocations, and jumps to
the exact real function. Address-free phase summaries go to the main trace;
the most recent raw argument registers are written only to the volatile
sidecar if a call occurs. No API name or signature is inferred. A fresh runner
must also prove the sentinel leaves the accepted generated-frame hash
unchanged before its zero-call result can remove the ID as a model dependency.

The normalized JSONL records interface IDs, real NvAPI statuses, stable
module/function/shader/resource tokens, function names, launch grid and block
dimensions, dynamic shared memory, and hashes of exact binary artifacts. It
never writes raw pointers, GPU virtual addresses, or D3D12 descriptor values to
the main JSONL. Module/cubin inputs and packed launch parameter bytes are saved
in the dump directory as private binary evidence. Raw UAV addresses and driver
surface handles are isolated in a required volatile sidecar and must never be
included in normalized or repeat roots.

The sidecar also records each successful CUDA resource-object result with its
stable main-trace resource token, current phase, exact output-handle width, and
little-endian byte order. Merged texture/sampler calls include both input
descriptor pointers; independent descriptor calls include type plus descriptor
pointer; texture and surface calls include their input descriptor pointers.
These raw values exist only to locate the corresponding little-endian handles
inside packed kernel parameters. They are volatile provenance—not semantic
model state.

## Optional D3D12 descriptor provenance

The proxy additionally exports
`NeuralStudioFgNvapiTraceAttachD3D12Device` and
`NeuralStudioFgNvapiTraceDetachD3D12Device`. The standalone host resolves these
exports opportunistically: no proxy, an older proxy with neither export, or a
run without descriptor-capture paths keeps the normal host path unchanged.

Set both values to fresh files in an existing job directory to enable the
capture before NGX initialization:

- `NEURALSTUDIO_FG_D3D12_DESCRIPTOR_TRACE`: address-free resource, heap,
  SRV/UAV/sampler, copy, and request relationships.
- `NEURALSTUDIO_FG_D3D12_DESCRIPTOR_VOLATILE_TRACE`: raw COM identities and
  CPU descriptor handles used only to validate those relationships.

The attach hook observes nine base `ID3D12Device` methods and restores every
vtable entry at detach. NvAPI request bindings are snapshotted before calling
the real driver because successful calls rewrite their scratch descriptor
slots. CUDA texture/surface results remain opaque object handles and are never
reported as GPU virtual addresses.

## Build on the RTX job host

From a job-scoped checkout beneath `D:\CodexLab`:

```powershell
.\scripts\build-fg-nvapi-trace.ps1
```

The script validates the Authenticode signer of
`C:\Windows\System32\nvapi64.dll`, builds the x64 proxy, and stages a
hash-identical renamed copy as `build\fg_nvapi_trace\nvapi64.real.dll`.

## Required launch environment

Set all five values before the process loads the proxy:

- `NEURALSTUDIO_FG_REAL_NVAPI64`: absolute path to the renamed real DLL.
- `NEURALSTUDIO_FG_REAL_NVAPI64_SHA256`: exact 64-digit SHA-256 printed by the
  build script.
- `NEURALSTUDIO_FG_NVAPI_TRACE`: absolute path for a JSONL file that does not
  already exist; its parent directory must exist.
- `NEURALSTUDIO_FG_NVAPI_VOLATILE_TRACE`: distinct absolute path for a JSONL
  file that does not already exist. It contains raw `gpuVAStart`, size, and
  `surfaceHandle` values solely for parameter provenance and is excluded from
  normalized roots.
- `NEURALSTUDIO_FG_NVAPI_DUMP_DIR`: absolute path to an existing, job-scoped
  dump directory.

Initialization fails closed if a path is absent, the main and volatile paths
are not distinct, the real DLL is not renamed, its hash differs, either real
dispatcher export is absent, or either trace cannot be created exclusively. A
dump collision or failed write also stops the wrapped call instead of producing
an incomplete capture receipt.

`NvAPI_D3D12_LaunchCuKernelChainEx` has two parameter encodings. Packed
`pParams` bytes are captured exactly. The alternative `kernelParams` form is
an unbounded pointer array with no argument sizes in this ABI; it is forwarded
and explicitly marked `pointer_array_unbounded_not_captured` without exposing
addresses. Recover argument extents from the captured cubin before promoting
that path to an exact parameter receipt.

The host may correlate model lifecycle calls with the kernel stream by loading
`NeuralStudioFgNvapiTraceSetPhase` from the proxy and calling it as
`void WINAPI(const char*)`. The marker is serialized into the same JSONL as a
`phase_marker` event. Phase names are limited to 128 bytes; ASCII letters,
digits, `.`, `_`, `-`, `:`, and `/` are preserved and all other bytes become
`_`.
