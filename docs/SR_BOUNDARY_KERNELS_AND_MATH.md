# SR boundary kernels and formula-generated math

Six available boundary implementations now have a recorded source build:

- The regular-depth input/output variants are compiled and loaded but remain execution-unverified under the current temporal API gate.
- `hiluma_input_depthinv_fast.metal` preserves the prepared Fast output only with its recorded fast-math policy; safer-math candidates changed the temporal result.
- `hiluma_input_depthinv_exact.metal`: recovered translated input source with the texture/MUFU contract and reciprocal residual refinement; compiled with safe math. Translated-input rights remain unresolved.
- `netal_copy_exposure.metal`: a small implementation against Netal's host resource ABI, Apache-2.0 attributed for its adaptation of CuMetal helper behavior.
- `hiluma_output_depthinv_exact.metal`: the existing translated output source, compiled with safe math. Its input-source redistribution basis remains unresolved; compilation is not publication clearance.

```sh
python3 scripts/build-netal-sr-boundary-kernels.py --output /new/local/boundary-kernels
```

`--only netal_copy_exposure` builds just the exposure component. The build receipt records source hashes, math flags and compiler identity. It explicitly does not claim a complete runtime, inference acceptance or redistribution clearance.

## Exposure behavior

The runtime contract is a one-channel float32 exposure resource copied to a one-channel half resource through the host's 24-byte resource descriptor. The replacement preserves the sampled legacy conversion behavior: finite truncation toward zero, infinity on half-range overflow, canonical positive NaNs for nonfinite inputs, and positive zero for float32 zeros/subnormals. This is not presented as a general IEEE rounding implementation for arbitrary resource formats.

A GPU conformance probe checked 2,571 structured and deterministic random bit patterns against the prepared reference, with zero differences and intact canaries. Earlier candidates disagreed on signed zero and nonfinite/subnormal cases; those failures were preserved and corrected before the final check.

A four-frame 1080p Reference check passed for the new exposure implementation, the rebuilt output boundary, and both together. A forced-one exposure control changed later temporal frames; a null output control changed the output immediately. This proves the tested kernels were used. The other boundary inputs, arithmetic resources and reference corpus were still supplied from the prepared runtime.

## Generated arithmetic resources

```sh
python3 scripts/build-netal-sr-math.py --output /new/local/math
python3 scripts/test-netal-sr-math.py
```

Three complete files are regenerated without a model, GPU, private binary input, or captured table:

| Asset | Generation rule | Verification |
| --- | --- | --- |
| Half reciprocal LUT | Reciprocal rounded to nearest-even half; signed zero/infinity and canonical NaN handling | All 65,536 entries match the pinned runtime artifact |
| Half exp2 LUT | exp2 truncated toward zero, plus one explicit SM120 calibration point at input `0x8001` | All 65,536 entries match the pinned runtime artifact |
| Texture weights | Joint fixed-point product rounded half-up; remaining coefficients derived to preserve the two marginals and total | All 264,196 coefficients and the file header match |

Independent ties-to-even rounding of four bilinear products was wrong at 2,048 entries. The source formula instead computes the joint integer coefficient as `(x*y + 128) // 256`, then derives the other three. Every partition sums to one with the exact requested horizontal/vertical marginals.

Each generator checks its final full-file SHA-256 and refuses contract drift. The tests exercise scalar edge cases and all 66,049 texture partitions. Matching these assets establishes reproduction of the existing runtime contract; it is not a universal claim about every NVIDIA architecture or driver.

The main asset builder now includes these three tables and, with Metal compilation and boundary metadata enabled, all six source boundary variants. Together with the eleven Reference kernels and two pinned calibration ROMs it produces 630 assets for the default profiles, including the live texture converter and format extension. General scalar math beyond sampled contracts and complete runtime assembly remain unfinished. The current prepared runtime also has sampled scalar-math lookup data; that is not silently copied into the source build.

## Measured RCP/RSQ calibration data

The correction ROMs are reconstructed from standalone scalar sweeps, separately from the three formula-generated files. The original NVIDIA oracle source hashes and retained sweep hashes match their historical receipt seals. Those records report equal repeats, but the second raw sweep files are no longer retained; no fresh NVIDIA execution is claimed here.

Fresh M3 Max measurements match the old Metal baselines over 16,777,216 inputs per operation, with two poison patterns, exact per-index write coverage and intact canaries. The source packer checks 33,554,432 NVIDIA/base word pairs, preserves exponent-band/parity indexing and signed ULP corrections (including eight RSQ -2 cases), and reproduces the pinned two- and four-MiB ROMs byte-for-byte.

These six MiB are explicitly included under `runtime/arithmetic-calibration/` with metadata and probe sources. They are neither neural weights nor compiled kernels, and are not described as formula-derived. The source export permits only the two exact paths, sizes and hashes. Mac asset builds read these pinned data without needing CUDA, private sweeps or a prepared runtime. Reproduction from new scalar measurements and the hardware/compiler limits are documented in [the calibration recipe](../tools/math-calibration/README.md).

This closes the retained-ROM packaging dependency for the tested profile; it does not establish all-FP32 parity, another Mac's behavior, general LG2/EX2 coverage or publication clearance. Model tensors, vendor DLLs and SDK headers remain excluded.

## Recovered input source and combined validation

The prepared exact input library is byte-identical to its `hiluma_input_depthinv_texexact_contract_ffma_rnercp.metallib` alias. The corresponding retained research source, `hiluma_input_texexact_de14_contract_ffma_rnercp.metal`, has SHA-256 `59477cbf22ec046b356e9d81efda6dcb0242f0ad125457974875bc2bdb825798` and is now preserved in the source tree as `hiluma_input_depthinv_exact.metal`.

The rebuilt library has a different binary hash from the prepared library. Acceptance is based on frame output, not an assumed binary match: a new baseline, the rebuilt input alone, and the rebuilt input/output/exposure combination each match all four stored 1080p RTX outputs in Reference mode. Output canaries remain intact. A null input kernel changes 4,480,093 bytes in the first frame and fails the parity gate, confirming the input replacement is selected and affects inference.

The initial three boundary binaries matched those tested builds. That checkpoint preserved its previous 609 assets and added three libraries; the later Reference/legacy recovery brings the current total to 628. The tests still use prepared correction ROMs, sampled scalar-math data, legacy body libraries and the reference corpus. The input's scalar lookup falls back to Metal math for unrecorded inputs, so the fixture result is not a general floating-point parity guarantee. Other resolutions, real-game quality, and full fresh-Mac setup remain separate gates.

## Provenance

See [source provenance review](SOURCE_PROVENANCE_REVIEW.md). Compiler/helper licensing, rights in translated inputs, correctness, performance and release readiness are distinct. The newly recorded source build and local tests do not clear unresolved translated-source distribution rights.
