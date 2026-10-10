# Performance and correctness

No new gameplay benchmark is claimed by this release cleanup. Historical SR offline kernel timings and NR still-frame timings do not establish the performance of the packaged app in a moving game.

For each supported configuration record:

- Netal revision, model identity, Mac chip/memory, macOS, and CrossOver/D3DMetal/game versions.
- Input/render/output sizes; FG internal size; all SR, FG, and NR switches.
- Warm-up procedure, synchronization, repeat count, background load, and measurement scope.
- Source FPS separately from displayed FPS; median and tail frame times, stalls, memory, and thermal behavior.
- SR-only and SR+FG comparisons against the same scene without Netal.
- Artifacts, fallback counts, and proof that Netal actually served the output.

Byte identity is required only for paths described as exact. Approximate native or MLX NR must report error against the reference and moving-frame visual quality. A finite tensor, a completed command buffer, or a high single-frame PSNR alone is not gameplay acceptance.

The saved MLX 540p repeated-still pipeline result was about 62 ms on M3 Max before game overhead. Keep that scope attached if citing it. It does not establish a 20 ms NR runtime or 60 FPS gameplay.
