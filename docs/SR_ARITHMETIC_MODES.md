# SR arithmetic modes and validation scope

Netal's SR model weights are unchanged between these modes. The arithmetic and kernel selection differ. “Reference” names an implementation path, not a promise that every input and resolution has been proved identical to NVIDIA.

| Mode | Configuration | Current evidence |
| --- | --- | --- |
| Fast | Hardware lowering (`_hw7`), normally with batched pipelines; the live launcher selects the non-exact input boundary | Lower latency is the design intent. Output differs from the stored RTX reference; visual/gameplay acceptance remains separate |
| Reference | Empty lowering suffix, batching disabled, managed fast input override cleared | All four stored 1080p frames match RTX byte-for-byte. Reset frames at 720p, 1440p and 2160p match; subsequent stored temporal frames at those sizes still differ |

During initialization, SR mode changes wait until construction finishes; session publication is synchronized with the SR hook. FG can still update its enable flag, and separate polling cadences prevent SR updates being starved by FG. Live transition stress testing remains outstanding.

Choosing Reference now disables batching as part of the same control update. The bridge also enforces that rule for contradictory control files and removes its known fast input-kernel override. Other explicit developer kernel overrides are preserved; they are outside the validated standard configuration. Enabling batching selects Fast. FG settings are unchanged by this selection.

## What was isolated

The 1080p capture's twenty input/output artifacts match their original inventory hashes. Its runtime identity, motion scale, depth convention, reset sequence, and recorded window shifts agree with the tested configuration.

With the same inputs and boundaries, original translated kernels match the four stored RTX outputs. Hardware and batched implementations produce the same differing outputs. Stage snapshots first diverge at encoder 0; the preceding input-stage arenas match. This localizes the fast-body difference, but does not identify one specific instruction or establish a universal error bound.

## Measured Fast-preset error

On the four stored 1080p frames, the actual launch preset produced:

- RGB mean absolute error: approximately 0.0000012–0.0000149.
- RGB PSNR with a fixed peak of 1: approximately 77.9–79.5 dB.
- Largest absolute RGB difference: approximately 0.0918.
- Pixels with an RGB-channel error above 0.01: 55–100 on the last three frames, and 64 on the reset frame, out of 2,073,600 pixels per frame.
- Finite outputs and identical alpha channels on all four frames.

These are linear framebuffer comparisons. Values were not clipped for measurement, and some reference values exceed 1. The high aggregate PSNR must not hide local outliers or be treated as proof of acceptable moving-game quality. No new gameplay FPS or sustained visual-quality claim follows.

The separate default-boundary fast-body diagnostic has slightly different temporal outputs; it is not interchangeable with the live launch preset.

## Probe claim boundaries

The temporal probe records requested kernel settings, input hashes, jitter, motion scale, expected-output hashes, and output hashes. A missing reference output results in `unverified`, false RTX parity, and a nonzero exit status. A truncated reference is an error. The receipt distinguishes logical network stages from unmeasured physical dispatch count; batching must not be mislabeled as fourteen physical dispatches per frame.

For a locally supplied compatible 1080p runtime/capture, the GPU claim-boundary check is:

```sh
python3 scripts/validate-netal-sr-probe-claims.py \
  --probe build/macos_host/dlss-sr-session-temporal-probe \
  --bundle '/path/to/runtime-bundle' --capture '/path/to/capture' \
  --output '/new/local/claim-check'
```

This exercises matching, missing and truncated oracles. It uses private inputs only when explicitly supplied locally; none are included with public source. Shared-lab runs must use the lab's GPU reservation mechanism.

## Remaining acceptance work

The temporal discrepancy beyond 1080p, the precise fast-path numerical mechanism, broader input coverage, HDR/motion/cuts, and sustained SR+FG gameplay still need investigation. The UI policy and both bridge architectures compile and have focused tests, but changing modes during a real game still needs a live transition/recovery test. Keep NR experimental and disabled in ordinary launches.
