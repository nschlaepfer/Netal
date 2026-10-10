# Frame Generation central block

This directory reconstructs the dependency-closed launch-71
`k_central_block` unit from selected module token 61:

- module bytes: `5752`
- module SHA-256:
  `184542107e98be3dc6229251c19cae4ecc44a7cc1b78cbf741b7fb3b8d81f060`
- live launch: grid `40x23x1`, block `16x16x1`
- logical extent: `640x368`
- inputs: two FP16x4 images, two FP16 scalar images, one FP16x4 motion
  image and one FP16x3 image
- output: nine packed u32 values (`36` bytes) per pixel

The selected module contains two entries. `k_central_block_original` directly
writes the semantic result. The live-selected `k_central_block` first stages
the same nine u32 values per thread in 9,216 bytes of shared memory, then
remaps the stores for coalescing. The RTX oracle executes both entries with
the same independently generated inputs in every process and requires byte
equality.

`central_block.metal` is the direct semantic form. It implements signed
negative-coordinate floor before clamp and explicitly spells the SM120
driver's measured HFMA2 contractions. All other implicit contraction is
disabled at compile time. The output is written as nine explicit u32 values
to avoid Metal structure-layout ambiguity.

Neither backend opens an expected tensor or captured model intermediate. The
CUDA and Metal executables each generate their inputs locally; the Python
verifier independently regenerates and validates the post-execution inputs.
The accepted receipt requires four fresh processes per backend, both `0x5a`
and `0xa5` output poison values, input immutability, guard canaries, write
coverage and exact RTX/Metal output bytes.

This proves only the launch-71 central-block unit for the recorded numerical
domain. It does not prove a multi-kernel FG group, a complete frame, Netal
integration or a performance benefit.
