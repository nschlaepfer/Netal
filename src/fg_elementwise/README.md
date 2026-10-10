# FG selected elementwise add

This directory reconstructs exactly one live-selected NVIDIA Frame Generation
unit: `evaluate_1` launch token 33, function `k_element_wise`, module token 37.

The selected 2,152-byte module has SHA-256
`1970b33c56b1580790a7d56377e4bfd5c1ed8522cd3d5e28b07192b481067d26`.
Its embedded SM89 PTX declares three 64-bit pointers followed by seven 32-bit
scalars. Inspection and execution resolve the pointer roles as input A, input
B, then output. The address-free scalar contract is:

```text
output width=10, output height=6, channels=256,
input A width=10, input A height=6,
input B width=10, input B height=6
```

The live launch is grid `[3,2,1]`, block `[4,4,8]`, with no dynamic shared
memory. The selected code uses `add.f16x2` over NHWC storage, eight channels
per vector and four vectors per z-thread. `elementwise_add.metal` independently
implements that contract; it does not load RTX output or expected tensor bytes.

`rtx_oracle.cpp` loads the exact selected module through the CUDA Driver API.
`macos_probe.mm` runs the independent Metal implementation. Each executable
generates the same deterministic finite-normal binary16 inputs internally and
accepts only an output poison byte, never a tensor or expected-output path.

Only the 256-channel module-37 variant is accepted here. Modules 41, 46 and 50
have the same elementwise-add arithmetic but distinct hard-coded channel-vector
shifts for 128, 64 and 32 channels. They remain unaccepted until separately
executed and receipted. This unit does not establish a dependency-closed FG
group, full-frame parity, Netal integration, or performance.
