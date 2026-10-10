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

## Optional MLX NR backend

MLX-DLSS is a separate upstream project and is not vendored in this source candidate. Any future integration must pin its revision, preserve its license and dependency notices, document installation, and keep experimental NR off by default. Neither installing MLX nor possessing a model file proves that continuous in-game NR is supported.

## Source-built SR model capture helpers

`src/model_capture/` contains the project's capture-helper source recovered from the revision recorded in `SOURCE_ORIGIN.json`. The build takes NVIDIA DLSS SDK headers as an explicit external input; those headers and the NVIDIA DLL are not copied into the public export. MinGW/GCC build package versions and all input/output hashes are recorded by the builder. Applicable SDK and compiler/runtime redistribution notices must be reviewed before distributing the compiled helpers inside an app package. Successful compilation or extraction does not complete that review.

## CuMetal compiler helpers and translated shader inputs

CuMetal (`Lulzx/cuda-metal`) is Apache-2.0 licensed. The local compiler snapshot used by the research build is commit `14a677715904d2e3d14f4973479df5620eb2f8b9`; its license is retained in `LICENSES/CuMetal-Apache-2.0.txt`. Generated helper code and adaptations require the applicable upstream notices. The new `netal_copy_exposure.metal` is explicitly Apache-2.0 attributed for its adaptation of that helper contract; this is not a clean-room authorship claim.

Several source shaders carry explicit CuMetal generic-PTX-lowering markers. The compiler/helper license does **not** establish the rights in their input programs. These files are not cleared for public distribution merely because the surrounding project uses MIT or the export contains no model binaries. Run `scripts/source-provenance-inventory.py` to produce the bounded marked-file inventory. Unmarked source also needs origin review; the tool is not a complete ownership audit.

The NVIDIA SDK headers remain external inputs. The published RTX SDK license contains restrictions on reverse engineering and distribution. This review has not established the governing terms, permissions, or any applicable exceptions for each archived runtime and translated input. Resolve the input provenance and distribution basis before publication; do not relabel such source as MIT by default.

Primary references: [CuMetal license](https://github.com/Lulzx/cuda-metal/blob/main/LICENSE), [NVIDIA RTX SDK license](https://github.com/NVIDIA/DLSS/blob/main/LICENSE.txt).
