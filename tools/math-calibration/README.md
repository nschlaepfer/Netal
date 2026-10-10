# Reproduce the scalar calibration profile

The dataset uses synthetic positive-normal FP32 bit patterns. No model is loaded. The retained NVIDIA measurements used an RTX 5090 (SM120), driver 616.56 and CUDA 12.8. The pinned profile is in `runtime/arithmetic-calibration/manifest.json`.

On a CUDA/SM120 host, build the original probes with an installed NVIDIA toolkit and run only their synthetic sweep mode:

```sh
nvcc -O2 -std=c++17 -arch=sm_120 oracle-rcp.cu -o oracle-rcp
nvcc -O2 -std=c++17 -arch=sm_120 oracle-rsq.cu -o oracle-rsq
./oracle-rcp --sweep rcp-nvidia.bin
./oracle-rsq --sweep rsq-nvidia.bin
```

The original probes also retain a historical `--file` diagnostic mode; it is not used to generate these calibration assets. Vendor SDK headers and compiler/runtime binaries are not shipped here. A different NVIDIA output hash is a profile mismatch, not permission to silently replace the data.

On a Mac with Apple's Metal tools, from the repository root:

```sh
zsh scripts/build-netal-math-probe.sh
build/math-calibration-tools/baseline build/math-calibration-tools/baseline.metallib rcp /new/rcp-m3.bin
build/math-calibration-tools/baseline build/math-calibration-tools/baseline.metallib rsq /new/rsq-m3.bin
python3 scripts/pack-netal-math-calibration.py --operation rcp --nvidia-sweep /path/rcp-nvidia.bin --m3-sweep /path/rcp-m3.bin --output /new/packed-rcp
python3 scripts/pack-netal-math-calibration.py --operation rsq --nvidia-sweep /path/rsq-nvidia.bin --m3-sweep /path/rsq-m3.bin --output /new/packed-rsq
```

Builds alone do not run the GPU. Coordinate GPU execution with other local workloads. Each Metal run processes 16,777,216 values twice, checks poison independence, checks exact coverage and guard bytes, and reports a SHA-256. Outputs must not already exist. The packer requires exact pinned input hashes, validates signed correction ranges and RCP band agreement, checks every reconstructed word, then atomically publishes the output directory. It uses only Python's standard library.

These commands reproduce a bounded calibration contract. They do not prove all-FP32 behavior, a new architecture result, neural inference, visual quality or game integration. The source-asset builder uses the pinned six MiB of packed data directly; ordinary Mac builds do not need CUDA or the 64 MiB raw sweeps.

## Unverified LG2 characterization

`lg2-characterization.cu` is a prepared, uncompiled characterization program. It is not a runtime implementation or accepted calibration profile. It uses complete mantissas for two exponent bands and 8,192 deterministic mantissas per normal exponent, with two poison passes and exact write coverage. The prior common-latent reconstruction was refuted by retained observations; see `tools/xlate/numerics/LG2.md`. Do not infer a general LG2 solution from the presence of this source.
