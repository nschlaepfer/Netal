# Selected elementwise family v2

The v2 package extends the immutable module-37 v1 proof to every selected
`k_element_wise` launch in the frozen FG trace. It uses separate source files
so the hashes embedded in the v1 receipt remain valid.

| Module | Launch | NHWC shape | Grid | Block | Channel-vector shift |
| --- | ---: | --- | --- | --- | ---: |
| 37 | 33 | `[1,6,10,256]` | `[3,2,1]` | `[4,4,8]` | 5 |
| 41 | 37 | `[1,12,20,128]` | `[5,3,1]` | `[4,4,8]` | 4 |
| 41 | 41 | `[1,24,40,128]` | `[10,6,1]` | `[4,4,8]` | 4 |
| 46 | 45 | `[1,48,80,64]` | `[20,12,1]` | `[4,4,8]` | 3 |
| 50 | 49 | `[1,96,160,32]` | `[40,24,1]` | `[4,4,4]` | 2 |

All four selected modules contain the same `add.f16x2` arithmetic and differ
in the hard-coded vector stride appropriate to their channel count. The RTX
runner loads each exact module with the CUDA Driver API. Module 41 is launched
twice with its two live scalar contracts. The Metal runner uses one semantic
kernel while preserving every live launch's grid and block geometry.

Each backend independently generates 906,240 binary16 values per input and
process. The random corpus uses finite normal values with equal operand signs;
twelve fixed pairs per launch add round-to-nearest-even ties and opposite-sign
cancellation. NaNs, infinities, subnormal inputs, overflow and broad random
opposite-sign pairs are explicit domain limits, not implied successes.

The accepted family output is 1,812,480 bytes per process with root SHA-256
`042f91bf4bede61f3af9019b49fd9d9648669ae4cc08c7a4b0ccc4c0d4afa1b2`.
This is an exact elementwise-family result, not a dependency-closed neural
group or generated-frame result.
