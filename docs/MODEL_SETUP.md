# Local SR model setup

**Bring your own supported DLL.** Netal does not provide a download of NVIDIA DLLs or weights. Obtain the DLL from your own authorized installation or another NVIDIA-authorized source, under its applicable terms. Do not use an arbitrary DLL download site or upload a DLL in an issue/PR. The tested filename is `nvngx_dlss.dll`, and **only the exact SHA-256 below is validated for this SR import path**. Matching the filename, marketing version or Preset K label alone is insufficient; newer DLLs are not automatically supported.

```sh
shasum -a 256 '/path/to/nvngx_dlss.dll'
```

Compare the full result with the pinned SR hash before importing. A different hash means an unvalidated build, not permission to bypass the normal installer's identity checks. The exact tested SR and FG DLLs both report **file version 310.8.0.0**, verified from their binary version metadata after checking the full hashes. This version label is supplementary; the full hash is the authoritative build identifier.

The release branch can build its model-import helpers from source and recover the supported SR model from a locally supplied DLL using standalone Wine. An NVIDIA GPU, private captured frames, and the temporal inference probe are not required for this step.

This supplies **model data only**. The extracted package also contains canonical boundary metadata. Profiles and ABI templates can now be generated from it; remaining boundary kernels, arithmetic tables, fallback kernels and FG inputs are still separate runtime-bootstrap work. See [boundary metadata](SR_BOUNDARY_METADATA.md). Model-byte identity does not certify the engine or gameplay.

## Supported input

The default path is restricted to the tested quality model:

- Filename: `nvngx_dlss.dll`; file version: `310.8.0.0`.
- DLL SHA-256: `c85f971ce023c9f3492fc7455f0b01a24ba18ea39636407a846902c4360b0b7e`.
- Model: `DLSS_SR_Feature1_Quality_K`.
- `weights.bin`: 3,545,824 bytes; SHA-256 `ddaa59cd82deed53bad9d65b82b454f1439fc4875e4220b3ffafafb6cd0296a1`.
- `constants.bin`: 524,288 bytes; SHA-256 `305f4c6773d1cb333b577094e28a46beb1867898716d8290121862a2cb06e773`.

No DLL, model bytes, or NVIDIA SDK headers are included in the public export. Supply these inputs locally under their applicable terms. This document records a technical setup procedure, not a redistribution-rights determination.

## Build the helpers

Prerequisites: Python 3.10+, a running Docker-compatible engine, and a local NVIDIA DLSS SDK directory with `include/nvsdk_ngx.h` and `include/nvsdk_ngx_params.h`.

```sh
python3 scripts/build-netal-capture-tools.py --headers '/path/to/DLSS-SDK'
```

The default output is `build/model-capture`. The builder refuses an existing output; pass a new `--output` directory for a rebuild. It mounts only the capture source directory and the explicitly supplied headers read-only, compiles in Linux/amd64, and writes five helpers plus a receipt. The base image is digest-pinned; installed compiler packages and SDK/header hashes are recorded. Package repositories can change, so this is a recorded source build, not a claim of byte-reproducible PE binaries.

`src/model_capture/SOURCE_ORIGIN.json` records the original source revision and file hashes. The helpers originated in Netal's SR research code. Their binary distribution and all SDK/compiler notices still need review before inclusion in a public app package.

## Extract without an existing runtime

Use standalone `wine64` (or an explicit `NETAL_WINE` executable), with its matching `wineserver`. The extractor does not use a CrossOver game bottle.

```sh
python3 tools/netal-dlss-extract.py '/path/to/nvngx_dlss.dll' \
  --capture-dir build/model-capture --out '/new/local/model-directory'
```

Every run uses a new Wine prefix. Its background services are stopped only within that prefix. The loader's uploads are reconstructed by destination address and checked for coverage and bounds. Both full model hashes and the three canonical boundary-template hashes must match the supported contract before a complete output directory is published. Existing outputs are refused. An extraction receipt records input/helper/output identities and explicitly leaves inference/gameplay validation false.

`--work /new/diagnostic-directory` retains trace data for local diagnosis. It may contain proprietary model and kernel data: keep it private. The default temporary run directory is cleaned up. The research-only `--allow-unknown-model` extractor flag never reports a reference match, and the normal installer refuses unknown models.

## Install into a runtime skeleton

Once engine assets and a matching manifest have been provisioned:

```sh
zsh scripts/netal-dlss-ingest.sh '/path/to/nvngx_dlss.dll' \
  --bundle '/path/to/re9-sr-runtime-bundle' --capture-dir build/model-capture
```

The app discovers helpers in `build/model-capture` or the legacy `vendor/dlss-capture` directory. `NETAL_CAPTURE_DIR` or `--capture-dir` selects another set for command-line use.

The installer serializes imports, verifies the full model bytes, stages a complete copy of the runtime, updates the model manifest, rechecks the staged model, then atomically exchanges the old/new directories. It never replaces files one at a time in the live tree. A process interruption before the exchange leaves the old tree; an interruption after it leaves the complete new tree and may preserve the old staging tree for recovery. This is process-interruption protection, not a power-loss durability claim. Restart the game to load the installed model.

`--dry-run` performs extraction and validation without changing the runtime. A linked runtime directory is refused. The skeleton must identify the supported model; this installer does not generate missing engine assets or certify that all of them are present.

## Validation

```sh
python3 scripts/test-netal-model-extraction.py
python3 scripts/test-netal-model-install.py
```

The fault suite checks transfer layout/coverage/bounds, corrupt or missing model data, incomplete-output rejection, preservation on commit failure, concurrent-import refusal, and process death immediately after the atomic exchange. Actual Wine extraction is an additional integration check and uses local inputs; no proprietary model is required by CI.

## FG model setup boundary

The retained FG reference DLL is **`nvngx_dlssg.dll`, file version `310.8.0.0`**, SHA-256
`5d5cbf14d2727d47f93fd10bf77bd91708ae122482a6f86fd564971641ebd47b`.
This is a separate pinned input from the SR DLL. A filename or version label does
not replace the hash. See the identity contract in
[`src/fg_runtime/runtime-inputs.json`](../src/fg_runtime/runtime-inputs.json).
**Owning this DLL alone does not complete FG setup on a fresh Mac.**

FG currently accepts the supported 67 locally supplied post-evaluate model
buffers through [source assembly](FG_SOURCE_BUILD.md). Its math tables are
separate pinned scalar assets. The retained extraction host in `src/fg_host/`
selects an NVIDIA D3D12 feature-level-12 adapter and reads the DEFAULT buffers
after the first evaluation completes. The extraction audit records that the
post-create buffers were zero; copying those buffers before evaluation is not
a model setup solution. No Mac-only FG DLL extraction path is verified.

The app verifies those supported model identities and the complete staged FG
inventory before showing **Verified** or automatically requesting FG during
launch. This makes incomplete setup visible; it does not obtain the missing
buffers, grant redistribution permission, or certify gameplay.

## Support at a glance

| Feature | User-supplied source | Supported setup scope |
| --- | --- | --- |
| SR | Exact `nvngx_dlss.dll` hash above, yielding `DLSS_SR_Feature1_Quality_K` | Source-built Wine extraction tested; model import still needs the remaining engine/runtime assets |
| FG | Exact `nvngx_dlssg.dll` hash above and its pinned post-evaluate buffers | Existing prepared local runtime works; a complete Mac-only DLL-to-runtime setup is not verified |
| NR | Separate experimental model/runtime inputs | Not part of the normal SR/FG setup or a generally supported user import flow |

The successful Resident Evil/CrossOver sessions used prepared supported assets.
They establish game integration, not compatibility with every DLL build or a
one-click installation on another Mac.

## App verification

The SR model row checks these same supported full model identities independently
of the selected runtime manifest. A self-consistent manifest for different bytes
cannot produce **Verified**. Legacy v1 manifests may retain their historical
40-character digest prefixes, but the app still hashes all model bytes against
the full supported SHA-256. Version 2 requires full manifest hashes. Missing or
invalid metadata, unsupported model names, symlinks and truncated files cannot
verify. File edits invalidate the cached result even when size and modification
time are restored. This is model verification; the runtime loader separately
checks the engine, and actual gameplay remains a separate release gate.
