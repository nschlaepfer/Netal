# FG custom three-plane upsample

This directory reconstructs only the pinned NVIDIA Frame Generation
`evaluate_1` launch 70, `custom_upsample_hf_kernel` from module 60.

Selected identity:

- module SHA-256: `1d19bfddd2affc5587ec19829259b070404e44c28a6f0c570a4b7769fd0dbba6`;
- function: `custom_upsample_hf_kernel`;
- CUDA grid/block: `40x23x3` / `16x16x1`;
- packed parameter bytes: 144;
- address-free captured parameter SHA-256:
  `80189cac9cec275fc90e1a52f3e814475d063b67ed2fd80b05ae98ee1e9654f6`.

The packed ABI contains six 24-byte records.  Each record is a device pointer
followed by `N,H,W,C`.  Records 0-2 are input planes `320x184` with channel
counts `4,1,3`; records 3-5 are their `640x368` outputs.  All six pointers are
bound to provenance-complete live allocations in the frozen trace.

The PTX chooses a record pair through `ctaid.z`.  The Metal reconstruction
uses three explicit dispatches, one per plane.  Although the source PTX spells
nine separate half operations, direct selected-module output rejects that
literal model.  The RTX driver contracts the left product of both horizontal
adds and of the final vertical add while rounding each right product first.
Metal therefore uses three explicit half FMAs:

1. a rounded right horizontal product and FMA for the upper row;
2. the same two operations for the lower row;
3. a rounded lower vertical product and final FMA.

No weights, descriptors, shared memory, approximate division, or archived
intermediates are consumed.  RTX and Metal independently generate their
inputs in-process.

The accepted generator spatially mixes positive and negative finite FP16
normals plus signed zero, including cancellation cases.  The receipt does not
claim exhaustive behavior for FP16 subnormal inputs or all possible subnormal
intermediates, infinities, NaNs, arbitrary scale ratios, arbitrary shapes, the
surrounding convolution block, the complete generated frame, or Netal
integration.
