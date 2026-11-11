# Third-party notices

Netal is an independent compatibility and graphics research project. It is not affiliated with or endorsed by NVIDIA, Apple, CodeWeavers, Capcom, RenoDX, or ReShade.

The top-level MIT license covers original Netal code. Preserve upstream copyright and license notices on imported or adapted material. Model weights, captured tensors, proprietary vendor runtimes, and game assets are not relicensed by Netal.

## Public source export

The release exporter omits model data, DLLs, executable binaries, compiled GPU libraries, model-associated captured research data, and Git history. Five explicitly hash-pinned scalar arithmetic calibration files are documented data exceptions; their measured origins, source identities and limits accompany them under `runtime/arithmetic-calibration/` and `runtime/fg-arithmetic-calibration/`. The latter also records that its new standalone CUDA regeneration helper remains unexecuted. This does not admit arbitrary binaries or clear overall publication rights. The existing private research archive remains separate. Historical notices about archived NVIDIA-signed runtimes do not mean those files belong in a public release.

Generated or translated source still needs an ownership and provenance review. Excluding binaries does not settle that review. See `docs/RELEASE.md`.

## Research references and upstream contributions

The research history records influence from RenoDX (MIT), NIGos' `dlss5-dx11-bridge` (MIT), Alan Backer's `dlss5-vk-bridge` (MIT), and NVIDIA Streamline sample/source material. Preserve the applicable notices for any corresponding code included in a distribution. References are not permission to redistribute every file from those projects.

OptiScaler was recorded as an architectural reference, not copied source. Verify that remains true for any contribution that imports transport code.

NGX and Streamline headers, if supplied locally for a build, retain their upstream terms.

## Adapted MLX-DLSS neural rendering formulation (`src/nr_fast/`)

`src/nr_fast/` is a standalone native Metal 3.1 and C++20 implementation; it does not vendor MLX, MLX Swift, or any model files. Portions of its weight-package contract and operator math are adapted from WaveCut's `iamwavecut/MLX-DLSS` (Apache-2.0), pinned at commit `7870371e8bf6c03967213f1a5d40db6c5a3bee38`. Upstream license and `NOTICE` texts are preserved in `LICENSES/MLX-DLSS-Apache-2.0.txt`, and modified/adapted source files carry explicit `SPDX-License-Identifier: MIT AND Apache-2.0` attribution headers.

Specifically adapted contracts in `src/nr_fast/` (`nr_fast_runtime.h`, `nr_fast_runtime.mm`, `nr_fast_kernels.metal`) are:

- `mlxdlss.neural-rendering-transformer.v1` `.dlssmodel` manifest validation and `mlxdlss-native-e4m3-v1` `F8_E4M3`/`F16`/`F32` safetensors layout, plus dynamic U-Net topology inference (`Small31` `[2,2,2,2,2]` 31-block and `HT71` `[4,4,6,8,8]` 71-block);
- 12-bit window attention-bias fragment swizzle (`headCount == 1` and `headCount == 16`), bit-affine FP16 softmax (`clamp(fma(score, 0.044921875, 1.30078125))`), vendor cosine skip normalization, and E4M3 quantization/decoding math;
- Deterministic PCG RXS-M-XS (`0x108ef2d9`) 3-channel Gaussian noise preprocessor contract, 5-tap Catmull-Rom temporal history reprojection, and separable Gaussian detail-split composition.

Apache-2.0 covers the open-source MLX-DLSS code and Netal's native Metal adaptation. It does **not** relicense or clear third-party model weights, proprietary vendor runtimes, or game assets. Experimental NR remains strictly opt-in (`off` by default) in all bridges and launchers; neither building `src/nr_fast/` nor supplying a local model file establishes universal game compatibility or RTX bit-exactness on hardware-accelerated paths.

## Source-built SR model capture helpers

`src/model_capture/` contains the project's capture-helper source recovered from the revision recorded in `SOURCE_ORIGIN.json`. The build takes NVIDIA DLSS SDK headers as an explicit external input; those headers and the NVIDIA DLL are not copied into the public export. MinGW/GCC build package versions and all input/output hashes are recorded by the builder. Applicable SDK and compiler/runtime redistribution notices must be reviewed before distributing the compiled helpers inside an app package. Successful compilation or extraction does not complete that review.

## CuMetal compiler helpers and translated shader inputs

CuMetal (`Lulzx/cuda-metal`) is Apache-2.0 licensed. The local compiler snapshot used by the research build is commit `14a677715904d2e3d14f4973479df5620eb2f8b9`; its license is retained in `LICENSES/CuMetal-Apache-2.0.txt`. Generated helper code and adaptations require the applicable upstream notices. The new `netal_copy_exposure.metal` is explicitly Apache-2.0 attributed for its adaptation of that helper contract; this is not a clean-room authorship claim.

Several source shaders carry explicit CuMetal generic-PTX-lowering markers. The compiler/helper license does **not** establish the rights in their input programs. These files are not cleared for public distribution merely because the surrounding project uses MIT or the export contains no model binaries. Run `scripts/source-provenance-inventory.py` to produce the bounded marked-file inventory. Unmarked source also needs origin review; the tool is not a complete ownership audit.

The NVIDIA SDK headers remain external inputs. The published RTX SDK license contains restrictions on reverse engineering and distribution. This review has not established the governing terms, permissions, or any applicable exceptions for each archived runtime and translated input. The input provenance and distribution basis remain unresolved in this experimental release; public availability does not relabel such source as MIT.

Primary references: [CuMetal license](https://github.com/Lulzx/cuda-metal/blob/main/LICENSE), [NVIDIA RTX SDK license](https://github.com/NVIDIA/DLSS/blob/main/LICENSE.txt).
