# Three exact lowerings of NVIDIA's HMMA, raced on hardware

Built and measured 2026-09-04. `hmma-lowerings.metal` + `race-lowerings.mm`; 16,384 tiles =
1,048,576 K=16 stages, conv-shaped data (activations N(0,1), weights N(0,0.02)), M3 Max.

    xcrun -sdk macosx metal -std=metal3.1 -fmetal-math-mode=safe -c hmma-lowerings.metal -o l.air
    xcrun -sdk macosx metallib l.air -o l.metallib && ./race 16384 l.metallib

| lowering | ms (floor) | vs oracle | bit-identical to oracle | fallback |
|---|---:|---:|---|---:|
| `oracle` — integer bit decomposition, FG's shape | 0.197 - 0.795 | 1.0x | (reference) | - |
| `twofloat` — double-float accumulator, no MMA | 0.042 | 4.7x - 18.9x | **yes, 0 of 1,048,576** | none |
| `checked` — simdgroup MMA + provable bound | **0.042** | 4.7x - 18.9x | **yes, 0 of 1,048,576** | 1.97% |

## The result that matters

**Exactness is solved, twice.** Both fast lowerings reproduce the oracle bit-for-bit over a million
outputs. The question that blocked FG's convolutions since the 117-byte divergence - whether a fast
path can be exact at all - is answered yes, and by two independent routes.

**And the simpler route is not slower.** `twofloat` is about thirty lines, uses no matrix
instruction, needs no error bound, no predicate and no fallback path, and reaches the same floor as
the checked MMA. I designed the checked path expecting it to be necessary; on this benchmark it is
not.

**One caveat decides which to ship, and this benchmark cannot settle it.** Every tile here gets
fresh operands, so there is no data reuse - which is precisely the MMA's advantage. A real
convolution reuses each weight tile across many output positions, and the checked path should pull
ahead there. The honest recommendation is to implement both behind the same manifest and let the
real tensors choose; what is no longer open is whether either can be exact.

`checked` measured 0.042 ms in all five runs while `twofloat` ranged 0.042-0.166, so the MMA path is
also the more schedule-stable of the two.

## The trap that cost an hour, and would cost anyone else the same

**Metal compiles with fast math by default, and fast math silently destroys compensated summation.**
`two_sum` computes `(a - (s - bb)) + (b - bb)`, which is algebraically zero and which an optimising
compiler will duly fold to zero. Without `-fmetal-math-mode=safe` the double-float accumulator
degrades to plain fp32 and reports itself as working - `twofloat` and `checked` returned *identical*
mismatch counts, which is what gave it away, because two different algorithms have no business
agreeing to the byte.

Every compensated-arithmetic kernel in this project must be compiled with `-fmetal-math-mode=safe`,
the same flag the SR hiluma kernels already require for a different reason.

Two smaller findings from the same session, both worth keeping:

- The remaining 55 disagreements are between the GPU lowerings and a CPU `double` reference, not
  between the lowerings. They are not subnormals (checked: 0 of 55).
- fp16 subnormals *were* implicated before the fast-math fix (28 of 306) and disappeared with it.
  The double-rounding theorem `q >= 2p + 2` holds exactly at fp32-to-fp16 (24 >= 24) in the normal
  range and does not in the subnormal range; anything rounding through fp32 to half needs to know
  that.
