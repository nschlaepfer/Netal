# FG standalone arithmetic calibration

These three measured tables are separate from the 67 FG model buffers. They
contain no model weights, activations, game frames, DLLs or compiled code. They
are not formula-derived constants and are not made portable merely by copying
them to another GPU.

| File | Bytes | Meaning |
| --- | ---: | --- |
| `ex2-unit.bin` | 4,194,305 | Two-bit corrections for 16,777,217 inputs `-i * 2^-24` |
| `ex2-fraction.bin` | 2,097,153 | Two-bit corrections for 8,388,609 inputs `-2 + i * 2^-23` |
| `sigmoid.bin` | 262,144 | Float result bits for the specified scalar instruction chain over all 65,536 half bit patterns |

Each EX2 correction code is `Metal fast::exp2 bits - NVIDIA EX2 bits + 1`,
packed four codes per byte with lower indices in the low bits. The native
baseline hashes are pinned in `manifest.json` and
`scripts/netal_fg_calibration.py`. The sigmoid chain is half-to-float conversion,
`mul.ftz.f32` by bits `0xBFB8AA3B`, `ex2.approx.ftz.f32`, `add.ftz.f32` with 1.0,
and `rcp.approx.ftz.f32`.

## Origin and verification

The manifest records the original standalone probe source hashes, historical
receipt hashes, measured raw-table hashes, GPU and compiler. All three original
source hashes were checked against the retained receipts. Both retained sigmoid
outputs were rehashed and agree. The original unit/fraction receipts report
repeat equality, but only one raw copy of each is retained; that historical
repeat cannot be independently rechecked from two current files.

Fresh Apple M3 Max native sweeps using two different output poison patterns
match the pinned native fingerprints. Repacking all 25,165,826 grid entries per
process from those fresh sweeps and the retained NVIDIA raw tables reproduces
both compact files exactly, with no reconstruction mismatches. This verifies
these grids and asset identities; it is not a new all-input NVIDIA conformance
claim or a gameplay test.

## Check a local Mac

From the source root, under the applicable local GPU reservation workflow:

```sh
python3 scripts/check-netal-fg-calibration.py --output /path/to/new/check
```

This compiles a scalar Metal probe and runs two fresh processes. It checks every
native output through the pinned full-table hashes, as well as the guard bytes
and both poison patterns. It loads no model. The output includes raw sweeps and
a compatibility receipt. A different baseline fails the check; do not relabel
that GPU/compiler as supported. The current execution evidence covers the M3
Max and recorded toolchain only. This command checks EX2 compatibility, not
every FG arithmetic operation or device feature.

The assembler uses these three pinned assets by default and accepts explicit
paths for identical local copies. It checks integrity but does not itself run
the GPU compatibility check.

## Regeneration

`tools/math-calibration/fg-scalar-tables.cu` is a standalone regeneration helper
for the recorded table-producing instruction chains. Unlike the historical
fraction probe, it contains no extra normalization experiments or selected
live-input examples. This new helper has **not yet been compiled or executed**;
the retained historical sources/receipts and fresh Mac repacking establish the
current data identity, not validation of the new helper.

On an authorized CUDA machine, its intended build and invocation are:

```sh
nvcc -std=c++17 -arch=sm_120 tools/math-calibration/fg-scalar-tables.cu -o fg-scalar-tables
./fg-scalar-tables unit unit-u32.bin
./fg-scalar-tables fraction fraction-u32.bin
./fg-scalar-tables sigmoid sigmoid.bin
```

Compare every raw result's size and SHA-256 with `manifest.json`; generation
alone is insufficient. For each EX2 grid, compile
`src/fg_math_ex2/pack_fast_table_corrections.cc` on macOS and call it with:

```sh
pack <NVIDIA-u32-table> <native-fast-u32-table> <new-packed-file> <new-receipt>
```

The native tables come from the local compatibility check. The packed sizes and
hashes must equal the pinned profiles. The runtime's RCP table is the separately
reviewed asset under `runtime/arithmetic-calibration/`.

The source exporter allows only these exact file paths, sizes and hashes in
addition to the existing RCP/RSQ exception. Arbitrary binary data remains
excluded. This technical origin/integrity record does not clear redistribution
of the overall project or resolve translated-program provenance.
