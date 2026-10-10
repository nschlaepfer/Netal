# Netal 0.1 beta release gate

Target: SR + FG + a usable app. NR stays experimental with native and MLX considered separately. No normal-launch NR toggle is exposed until a continuous game path is validated.

**October 10 clarification:** native SR and Netal FG already worked in Resident
Evil Requiem through CrossOver. [The current status](STATUS.md)
records the positive game telemetry and maintainer observation. The unchecked
items below concern a distributable build, complete provisioning and broader
acceptance; they must not be read as a claim that in-game generation never worked.
Repository rules are [prepared but plan-blocked while private](REPOSITORY_GOVERNANCE.md).

## Completed in the release-preparation branch

- Duplicate documentation paths differing only by letter case consolidated for macOS.
- Runtime paths resolved at launch; a moved app has an explicit runtime-folder setup flow.
- Wine helpers no longer compile in checkout or user-home paths. Private launch settings and logs use per-process Application Support folders; prebuilt helpers are reused. Relocated synthetic-host checks pass.
- CrossOver executable and bottle forwarded through the launcher and injector; FG requested when staged.
- SR model status now requires the full supported model identities independently of runtime manifest claims. Malformed metadata, unsupported identities, duplicate rows, symlinks, truncation and stale same-size edits fail verification; legacy v1 prefixes remain compatible only with the supported full model bytes. FG now verifies supported model/math identities from an app resource and hashes the complete runtime inventory before showing Verified or automatically requesting FG. Same-size edits with restored modification times, missing required shaders, and symlinks invalidate FG readiness.
- Launch and import output drained while tools run; final errors retained with bounded memory.
- Native bridge build includes the FG session and runtime archive.
- SR extraction helpers built from source; two independent Wine extractions matched full reference model hashes. Model installation stages and atomically swaps the complete bundle.
- Notification denial is respected rather than bypassed through a scripting host.
- Public source export excludes history, model data, binaries, experiments, and local-path research helpers.

These are code changes, not gameplay certification. See the task's validation report for commands actually run.

## Required before a downloadable gameplay beta

- [ ] Reproducible complete SR runtime bootstrap on a clean Mac. The SR body layouts, repacked weights, pipeline descriptions and shared Metal library now build from source ([details](SR_SOURCE_BUILD.md)). Local model extraction now builds from source and matches the supported full hashes; installation no longer needs private frame captures or a temporal probe. Canonical boundary ABI and profile manifests now build from local DLL metadata. The recovered input boundary, tracked output boundary and exposure implementation now build from source and preserve the four stored 1080p Reference outputs together; three math tables are formula-generated. RCP/RSQ ROMs now have source packing recipes and explicitly pinned standalone-calibration data, with fresh M3 baseline checks. The eleven Reference body kernels and three legacy boundary variants now build from recovered sources; Reference output and the Fast input regression pass within the recorded scope. The live texture converter now builds from source and matches the prepared converter in bounded Metal 3/4 format tests. General scalar math and complete runtime assembly remain. Translated-input provenance is unresolved; see [boundary kernels and math](SR_BOUNDARY_KERNELS_AND_MATH.md). See [model setup](MODEL_SETUP.md).
- [ ] Reproducible FG bootstrap. The build-only engine path now compiles 73 shader libraries and both host architectures from a clean source export, without a research attempt or models ([details](FG_SOURCE_BUILD.md)). An explicit-input assembler now validates the supported model buffers and tables and produces a local bundle; four fresh-process synthetic runs and relocated Metal 4 output comparison pass. The three FG scalar tables now have explicit standalone assets, checked origin records and fresh native repacking; their new CUDA regeneration helper remains unexecuted. Reproducing model tokens on a fresh Mac, device compatibility, actual gameplay, and distribution validation remain.
- [ ] Complete separation of immutable app resources from writable per-user runtime data. CrossOver session settings/logs now use Application Support, but model installation and prepared-runtime packaging still need integration. The preview is not a self-contained installer.
- [x] Remove build-time absolute paths from the Wine helper. The launcher supplies a private process-lifetime configuration; a synthetic host loads the same helper from original and relocated paths. Actual Wine/game integration remains part of the following gate.
- [ ] Validate the distributable CrossOver runtime. Historical RE9 SR/FG gameplay and positive generation counters are established. A later portable-startup test initialized the bridge and both sessions, with all eleven SR layers present, but the locked desktop prevented that particular run from observing frame delivery. Complete fresh packaged-game acceptance, quality, pacing and recovery checks for the portable route. See [runtime integration](RUNTIME_BUNDLE.md).
- [ ] Complete numerical and visual acceptance for Fast mode and investigate the remaining non-1080p temporal discrepancy in Reference mode. The original translated path passes four stored 1080p frames; accelerated differences start at encoder 0 and are quantified. Mode selection and missing-oracle reporting have been corrected. See [arithmetic modes](SR_ARITHMETIC_MODES.md).
- [ ] Fix or explain the documented SR flashes; test scene cuts, resizing, pause/resume, loading, and repeated launches.
- [ ] Measure SR-only, FG-only where supported, SR+FG, and fallback. Record source FPS and displayed FPS separately, frame-time tails, memory, stalls, and visible artifacts.
- [ ] Verify native FG's 640×360 internal profile is acceptable for each supported output size. Reject unsupported geometry cleanly.
- [x] Test model reconstruction failures, concurrent imports, commit failure, and process interruption without exposing partial model files. Complete installer/app interruption and clean-machine checks remain part of distribution validation.
- [ ] Test on a clean account and a second Mac; publish the exact tested matrix. Do not promise all M3+ machines from one M3 Max result.
- [ ] Developer ID sign every executable, notarize the final package, staple it, and test a quarantined download with no source checkout present.

## Required before public source publication

- [ ] Review ownership and notices for imported, translated, and generated source. An extension allowlist cannot establish redistribution rights.
- [ ] Review the source manifest and exclusion report. Ensure source references to omitted research inputs are documented.
- [ ] Review the candidate for confidential text in addition to the automated credential/path checks.
- [ ] Publish only the reviewed history-free source snapshot, or separately review every reachable branch, tag, history object, release asset, and LFS object before exposing the research repository.
- [ ] Add a supported build environment and required CI checks to the new public repository.

## NR policy

Native exact NR is the reference lane. Approximate native and MLX execution must preserve declared model identity and report numerical and visual differences. MLX-DLSS is not vendored by this cleanup; integration remains work to do. On M3 Max, saved MLX 540p still-frame pipeline results were about 62 ms before game overhead. That does not establish 60 FPS, a 20 ms runtime, or moving-frame quality.

Keep the existing live-publication hold and its failure evidence in the private research archive. A controlled single frame does not authorize continuous inline NR. No NR launch defaults are changed by this release work.

## Generate a public source candidate

```sh
python3 scripts/export-public-source.py --output /absolute/path/to/Netal-public
python3 scripts/audit-public-source.py /absolute/path/to/Netal-public
```

The destination must not exist. The exporter reads tracked files plus explicitly allowed new release files, preserves modes, writes a SHA-256 inventory, and never copies `.git`. It does not rewrite or delete research history. Export success means the mechanical content policy passed; it is not legal clearance or a gameplay release approval.
