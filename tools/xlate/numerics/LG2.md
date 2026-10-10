# LG2 generalization remains open

The runtime's Reference input uses a bounded memoized LG2/EX2 contract and falls back to Metal math on a miss. Matching recorded frames does not establish behavior for unseen scalar inputs.

## Retained evidence

The historical SM120 audit covers all 2^23 mantissas of `[1,2)`. Its retained 32 MiB output has SHA-256 `7fd9be9a924e40a0ee18ba20e7f5158135112947597c8dd86131d3d4d8f9dbfa`. The source and old runner are `lg2-rtx-audit.cu` and `lg2-rtx-audit.ps1`. The historical environment was RTX 5090, driver 616.56, CUDA 13.3; it has not been rerun by this review.

The earlier exponent sample compared direct outputs with `e + lg2(m)`. Most sampled exponents differed by up to one ULP, and exponent -1 had much larger ULP differences near zero. **That is evidence against treating the decomposition as exact**, not proof that adding an exponent closes the gap.

The old note also measured that eight correction bits covered about 99.81% of the normalized binade relative to Metal `fast::log2`. This measured width does not supply the remaining corrections, a near-one formula, or cross-exponent semantics. The proposed general table was neither built nor verified. Claims that it was exact everywhere or removed all drift were unsupported and have been withdrawn.

## Local falsification of a stronger rounding hypothesis

A later candidate allowed two mantissa bands and a rounding-direction correction, assuming one common latent mantissa value plus an integer exponent. Its CPU implementation passes 1,211,072 cases under that mathematical assumption. This is a conditional algorithm check, not NVIDIA validation.

Exact rational rounding-interval checks against 79,911 retained scalar observations found 16,188 conflicts with that assumption. For example, the retained table maps input `0x3f000000` (0.5) to `0xbf7ffffe`, while the normalized-binade observation at mantissa zero is `0x00000000`. Their rounding intervals cannot come from the same latent value with an exact exponent offset. These are retained observations; fresh NVIDIA characterization is needed before attributing the discrepancy to a hardware mechanism or trusting a new reconstruction.

The candidate is not integrated into the runtime. A new standalone characterization program is prepared under `tools/math-calibration/lg2-characterization.cu`, but has not been compiled or run. It uses only synthetic inputs, coverage counters and poison guards. It measures full mantissas at exponents 0/-1 and a deterministic mantissa sample across every positive-normal exponent band. Any future general reconstruction must be checked independently against direct instruction results, including near-one and special-value behavior.

## Reproduce the local audit

```sh
python3 scripts/test-lg2-latent-reconstruction.py
python3 scripts/audit-lg2-latent-model.py --binade /path/lg2-binade.bin --keys /path/lg2-input-table.bin --values /path/lg2-output-table.bin --output /new/audit.json
```

The audit consumes historical arithmetic observations for diagnosis only. It does not generate model outputs or certify general LG2 behavior. It does not publish the underlying captured scalar tables as source assets.
