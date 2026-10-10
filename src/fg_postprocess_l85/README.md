# Frame Generation postprocess launch 85

This directory reconstructs the buffer-only `evaluate_1` launch 85 selected by
the pinned NVIDIA Frame Generation runtime. The selected entry is
`main_kernel` from module token 18, SHA-256
`07877ed3dcec1977a5ddfd7843314a5044ed9559bb2d222713906afdfc609d1c`.

## Live ABI and operation

The traced packed ABI is 56 bytes:

| Offset | Type | Meaning at the live launch |
| ---: | --- | --- |
| 0 | `u64` | 40x23 input, packed FP16 RGBA |
| 8 | `u64` | 20x12 level-0 output |
| 16 | `u64` | 10x6 level-1 output, inactive |
| 24 | `u64` | 5x3 level-2 output, inactive |
| 32 | `u64` | 3x2 level-3 output, inactive |
| 40 | `u32` | level count, `1` |
| 44 | `u32` | width, `40` |
| 48 | `u32` | height, `23` |
| 52 | `u32` | reserved and unread, `0` |

For every 2x2 tile it alpha-weights RGB, reduces the four lanes in the fixed
pairwise order `(p0+p1)+(p2+p3)`, normalizes RGB by the alpha sum, and stores
alpha sum divided by four. Odd-edge lanes contribute zero. With level count
one, the other three output pointers are valid but never written.

## Exact approximate division

The selected PTX uses `div.approx.ftz.f32`, not precise division. The dedicated
RTX probe independently regenerates the same bounded FP16 input domain and
records every numerator, denominator, approximate quotient, `MUFU.RCP`, RCP
product, and precise quotient. Across 960 divisions:

- `div.approx` and `mul.ftz(numerator, MUFU.RCP(denominator))` differ 0 times;
- precise division differs in 226 FP32 words and 12 final FP16 values;
- all denominators and reciprocal results stay in the accepted positive-normal
  domain.

The Metal kernel therefore binds the existing 2 MiB SM120 RCP architecture
correction ROM, SHA-256
`7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750`.
That ROM contains no model values or captured intermediates. It corrects
Metal's precise reciprocal by at most one ULP, then the kernel performs the
measured multiply. `fg_postprocess_l85_strict_division` is retained only as a
negative control.

## Reproduction

On the RTX rig, stage the exact selected module and RCP ROM in a new
`D:\CodexLab` job and run `scripts/build-fg-postprocess-l85-rtx.ps1`. On the
Mac, run `scripts/build-macos-fg-postprocess-l85.sh` with a new attempt path.
Then run `scripts/verify-fg-postprocess-l85.py final` over the two backend
roots. Neither backend accepts an expected output or archived producer buffer.

The accepted receipt covers four fresh processes per backend, both `0xA5` and
`0x5A` poisons, independent generated inputs, input and ROM immutability,
canaries, complete active writes, inactive-level non-writes, and exact 1,920
byte RTX/Metal equality. Ten receipt/artifact mutations are rejected.

## Boundary

This proves only launch-85 parity for the stated numerical domain and
`level_count=1`. It does not prove the other pyramid levels, the adjacent
launches, the complete generated frame, Netal integration, performance, or a
general all-FP32 emulation of NVIDIA division. Inputs outside the positive
normal denominator/reciprocal domain must fail closed until characterized.
