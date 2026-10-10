# Pinned scalar calibration data

These two files contain packed ULP corrections from standalone arithmetic probes. They contain no model weights, model activations, game frames, vendor DLLs, executable code or vendor headers. They are measured data, not formula-generated constants.

| File | Size | Meaning |
| --- | --- | --- |
| `rcp-correction-rom.bin` | 2 MiB | One two-bit correction per mantissa; observed equal across input exponents 127 and 128 |
| `rsq-correction-rom.bin` | 4 MiB | One two-bit correction per exponent parity and mantissa; observed at exponents 128 and 129 |

The NVIDIA source, output hash and historical receipt seal were verified against retained files. The original records report matching second runs, but those second raw sweep files are no longer retained. No new NVIDIA execution is claimed. The original standalone CUDA sources are preserved under `tools/math-calibration/` without vendor headers.

Fresh Apple M3 Max measurements on macOS 27.0 (26A5406e) match both historical Metal baselines over 16,777,216 inputs each. Each operation ran twice with different output poison, exact per-index write counts and canaries. Packing checked all 33,554,432 NVIDIA/base word pairs and reproduced both existing runtime ROM hashes. The eight RSQ corrections of -2 ULP are represented by code 3; they must not be treated as zero.

`manifest.json` pins the profiles, source identities, historical provenance and limitations. `scripts/netal_sr_calibration.py` verifies the two data files before the source-asset builder uses them. The source export admits only these exact binary paths, sizes and hashes; arbitrary binaries or modified bytes are rejected.

These data do not establish a license for translated shader inputs, whole-model parity, gameplay quality or support on another GPU/compiler combination. Publication clearance remains separate. Other Apple hardware and compiler versions must validate the baseline before relying on this profile.

See [reproduction](../../tools/math-calibration/README.md) for the source probes and packing commands.
