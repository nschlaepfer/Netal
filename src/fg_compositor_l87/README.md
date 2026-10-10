# Frame Generation final compositor launch 87

This directory reconstructs only the pinned `evaluate_1` final compositor:
`main_kernel`, module token 20, SHA-256
`6cc983a76decaed1a8c7f207a95531bece5a1ad092c4db3a0159b590c331c24b`.
It does not claim a complete Frame Generation graph, a complete generated
frame on Metal, Netal integration, or performance.

The selected 72-byte packed ABI binds an R8_UNORM 640x360 point/border mask,
an RGBA8_UNORM 640x360 output surface, a zero output origin, and four packed
FP16 RGBA pyramid buffers. The live `level=10` branch dereferences only the
320x180 head. It reconstructs each output pixel with premultiplied-alpha
weights `9/16, 3/16, 3/16, 1/16`, normalizes RGB with
`div.approx.ftz.f32`, writes alpha one, and skips the surface write when the
mask sample is zero. The low boundary duplicates its edge; the high halo is
zero.

## Oracle and Metal result

The accepted RTX oracle creates independent D3D12 resources and invokes the
exact selected fatbin through the same NvAPI CUDA-module path. A generic CUDA
array cannot represent the D3D UNORM surface contract and is retained as a
failed counterexample. NvAPI also requires the observed shader-visible
resource and sampler descriptor heaps; CPU-only heaps returned a null surface
handle despite status success and are likewise preserved as a failure.

Four fresh processes per backend cover both `0xA5` and `0x5A` output poisons.
They independently regenerate the 460,800-byte pyramid head and 230,400-byte
mask, produce an identical 921,600-byte result, leave zero active pixels
untouched, preserve every input and guard, and never consume an expected
output or archived producer tensor. A fifth process gates the zero-mask branch:
exactly 15,701 masked pixels remain untouched on both backends.

The reconstruction math itself matches at all 3,686,400 diagnostic float
bytes. Native Metal RGBA8_UNORM conversion is not equivalent to selected
`SUST.P`: it is RTX+1 at 10,919 RGB bytes. The measured selected conversion is

```text
fixed12 = floor(clamp(x, 0, 1) * 4096)
unorm8  = min(255, (fixed12 * 255 + 2047) >> 12)
```

That is truncation to 12 fractional bits followed by nearest/tie-down UNORM8
conversion. Encoding this architecture/format rule before the Metal texture
write yields zero RTX/Mac mismatches for both the full and sparse branches.
The 2 MiB SM120 reciprocal correction ROM is the already accepted
architecture table (SHA-256 `7cf4f8ce...d29750`); it contains no model values
or captured intermediates.

## Reproduction

Run the RTX builder in a fresh retained `D:\CodexLab` job with the exact module
and owned sources staged, then run:

```sh
scripts/build-macos-fg-compositor-l87.sh <new-mac-attempt>
python3 scripts/verify-fg-compositor-l87.py backend --backend rtx --root <rtx-attempt> --output <rtx-receipt>
python3 scripts/verify-fg-compositor-l87.py backend --backend mac --root <mac-attempt> --output <mac-receipt>
python3 scripts/verify-fg-compositor-l87.py final --rtx-root <rtx-attempt> --mac-root <mac-attempt> --abi evidence/specs/fg-compositor-l87-v1/abi-receipt.json --output <accepted-receipt>
```

The retained RTX job is
`D:\CodexLab\fg-compositor-l87-20260904T112919Z`; accepted attempt
`oracle-attempt-005-final`. The accepted local attempt is
`build/fg_compositor_l87/macos/attempt-006-final`.

Unsupported levels 11-13 and nonzero output origins fail closed in the Metal
host until each is independently gated. Joining this compositor to its
upstream pyramid producer remains separate work.
