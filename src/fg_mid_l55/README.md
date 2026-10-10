# Frame Generation selected launch L55

This directory reconstructs only `evaluate_1` launch 55, `main_kernel` from
module token 3, SHA-256
`66a9e6bda87ec8b63c5e10752e84d07435dc4392bbe42ec142e78b0d0b624e37`.
The selected 240-byte ABI uses accumulation mode 1, a 640x368 physical domain,
a 640x360 active domain, and optional UI disabled.

The live inputs are two RGBA16F 640x360 point/border textures, two RG16F
320x180 point/border textures, and two packed-half2 candidate buffers. The
outputs are two RGBA16F 640x360 surfaces, one R8_UNORM 640x360 mask surface,
two padded 640x368 half4 buffers, and one padded 640x368 half mask. The bottom
eight physical rows are zeroed only in the linear outputs, matching the PTX.

Four fresh processes per backend independently generate every input, cover
both `0xA5` and `0x5A` poisons, preserve inputs and canaries, and produce the
same 8,156,160 output bytes. No expected output or archived producer tensor is
accepted by either executable.

The accepted SM120 RCP architecture ROM supplies exact `div.approx.ftz.f32`
behavior. Normalized point sampling also requires the measured selected rule:

```text
fixed12 = floor(normalized_coordinate * 4096) / 4096
texel   = floor(fixed12 * dimension)
```

Native Metal point sampling missed 1,112,927 bytes. An early integer-boundary
heuristic still missed 451,328 bytes. The fixed12 rule matches all six outputs
byte-for-byte on the independently generated candidate-offset domain.

Only mode 1 with optional UI disabled is accepted. Modes 2/4/8/16, the
optional texture branch, connected execution, full-frame parity, Netal
integration, and performance remain outside this receipt.
