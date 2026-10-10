# Source provenance review — current findings

This inventory does **not establish legal clearance**. Public availability and the source export's content/hash audit do not establish the legal basis for distributing translated input source. The review remains open in this experimental release.

## Identified compiler origin

Twenty-two current source shaders have explicit CuMetal generic-PTX-lowering markers. The local compiler snapshot is `Lulzx/cuda-metal` commit `14a677715904d2e3d14f4973479df5620eb2f8b9`, whose license is Apache-2.0. Its license text is retained in `LICENSES/CuMetal-Apache-2.0.txt`. No root NOTICE file was found in that local snapshot.

The markers identify a generator. They do not, by themselves, establish who owns each input program or whether that input can be redistributed. The shader names and project research records associate several with vendor-runtime reconstruction, which requires a separate input-provenance review.

```sh
python3 scripts/source-provenance-inventory.py --output /path/to/inventory.json
```

The inventory pins each marked source's current hash and records the unresolved input-rights status. Unmarked sources are **not** automatically considered original or cleared. This is a bounded first pass, not the completed repository-wide provenance audit.

## License boundaries

- Original Netal code remains under the repository's MIT license, except where more specific notices apply.
- CuMetal-generated helper portions and adaptations retain the applicable Apache-2.0 attribution and license. Compiler licensing is not a blanket license for translated input code.
- `netal_copy_exposure.metal` is a small project implementation against the host resource ABI, with the observed CuMetal conversion behavior attributed under Apache-2.0. It is not represented as a clean-room implementation.
- The translated input- and output-boundary sources remain explicitly marked for input-rights review. Compiling and validating it locally does not resolve redistribution rights.
- SDK headers, vendor DLLs, weights, and model-associated captured runtime data remain outside the source export. The standalone arithmetic data exception below is explicitly pinned and documented.

The current published NVIDIA RTX SDK license includes reverse-engineering and distribution restrictions. This review does not establish which agreements govern every archived runtime, or the scope of any permissions or applicable legal exceptions. A source file cannot be cleared merely by changing its extension, regenerating it through a compiler, or placing MIT at the repository root.

Primary sources: [CuMetal license](https://github.com/Lulzx/cuda-metal/blob/main/LICENSE), [NVIDIA RTX SDK license](https://github.com/NVIDIA/DLSS/blob/main/LICENSE.txt).

## Standalone scalar calibration

The two explicitly allowlisted RCP/RSQ data files come from synthetic arithmetic probes, not model inputs or captured game tensors. Their original CUDA sources and receipt seals are retained; current M3 measurements and exhaustive packing reproduce the runtime hashes. Historical NVIDIA repeat equality is reported in the record, but a second raw output is no longer available for rechecking. This origin/identity review is separate from translated-shader rights and overall publication clearance. See `runtime/arithmetic-calibration/manifest.json` and the accompanying reproduction instructions.

## Open review work

Map generated/translated files to their input origin and applicable terms. Preserve required third-party notices. For unresolved inputs, establish a supported distribution basis or replace the affected implementation while retaining the intended SR/FG functionality. Review the remaining unmarked tree and dependency binaries as well. Do not treat local test success as publication approval.

## FG scalar arithmetic assets

Three additional pinned assets contain compact EX2 corrections and a sigmoid
instruction-chain table from standalone arithmetic probes. Their original source
hashes match retained receipts. Fresh M3 Max sweeps and exhaustive repacking
reproduce both compact tables. Both retained sigmoid copies match; the NVIDIA
EX2 repeat claims are historical because only one raw copy per grid remains.
The new standalone CUDA regeneration helper has not yet been compiled or run.
These are measured scalar data, not extracted model buffers. Their bounded
technical origin record is in `runtime/fg-arithmetic-calibration/manifest.json`.
This does not resolve translated-input rights or establish overall publication
clearance.
