# SR boundary source builds and scoped parity

Build these sources with `python3 scripts/build-netal-sr-boundary-kernels.py --output /new/output`.
The builder locates Apple's Metal compiler and adjacent linker and records source/library hashes
and math flags. All three libraries are also included by the main SR asset builder when boundary
metadata and Metal compilation are enabled.

- `hiluma_input_depthinv_exact.metal` preserves the recovered texture/MUFU-contract input source.
- `hiluma_output_depthinv_exact.metal` preserves the translated output source.
- `netal_copy_exposure.metal` implements the exposure-resource copy, with Apache-2.0 helper attribution.

Input alone and the combined input/output/exposure replacements match four stored 1080p RTX
outputs in Reference mode on the tested M3 Max. A null input changes the first frame; prior
output-null and forced-exposure controls also establish that their replacements are exercised.
Different compiled-library hashes are not treated as failure or success without output validation.

These checks still use prepared math resources and body libraries. Names containing `exact` and
comments describing intended arithmetic do not establish parity outside the tested scope. Input
and output are translated sources whose input-origin/distribution rights remain under review.
See [boundary kernels and math](../../docs/SR_BOUNDARY_KERNELS_AND_MATH.md) and
[source provenance](../../docs/SOURCE_PROVENANCE_REVIEW.md).

## Output-source discriminator

Earlier output near-misses inlined reciprocal helpers, producing 125 AIR `fdiv` operations instead
of one and diverging on temporal frames. The retained source produced one `fdiv` and 127 `cm_rcp`
calls in that comparison. The helper applies correction data to its division baseline to reproduce
the pinned approximation; the correction ROM is not a universal round-to-nearest guarantee.
Instruction shape is only a diagnostic: forcing `noinline` did not by itself recover the output.
Always finish with the frame receipt and a negative control.

## Selecting a candidate

`ExactLibrary()` can prefer an `_exact.metallib` over the apparent legacy filename. Run a null
control against the selected library before interpreting a positive result. A null replacement
that leaves the expected frames unchanged means the test did not establish its participation.
