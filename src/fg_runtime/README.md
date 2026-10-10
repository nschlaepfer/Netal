# Native Metal frame-generation session

`fg_runtime.h` is the fail-closed C session boundary used by the native bridge.  The caller owns all input/output textures.  A
session owns pipelines, model buffers, tables, scratch resources, temporal
histories and bank selection.

The pinned v1 profile is 640x360:

- current and previous backbuffers: `RGBA8Unorm`, shader-readable;
- motion: `RG16Float`, shader-readable;
- depth: `R32Float`, shader-readable;
- generated target: `RGBA8Unorm`, blit-destination capable.

A reset frame executes selected L3-L11 and produces no generated output.  Each
adjacent non-reset frame executes L12-L87 and writes the caller's target.  The
session retains its previous current-backbuffer object solely to enforce frame
lineage; the selected temporal history remains in the model's internal banks.
Callers must serialize calls on a session; v1 permits one live session per
process so its selected-tail resource cache cannot be cross-wired.

Failures never authorize a partially written target.  Validation happens
before command encoding; runtime errors fault the session and require
invalidation/reset or recreation.  The integration fallback is to present the
caller's current rendered backbuffer.

The implementation and evidence must keep the accepted long16 auxiliary
profile explicit.  It must not claim live HUDless, distortion, subrect,
dynamic-resolution, multi-frame, Netal or performance support until those
paths are independently gated.

`NSFGSessionEvaluate` waits internally. `NSFGSessionEvaluateMetal4` encodes
into a caller-owned Metal 3 or Metal 4 command buffer for sessions created with
`metal4=true`. The caller owns commit, completion, and presentation lifetimes.
The inline API does not itself establish correct live-game presentation or
safe ring reuse; those integration gates remain separate.

The [native offline quality lab](../../docs/NATIVE_FG_OFFLINE_LAB.md) executes
this model on independently rendered color, motion and depth. It keeps the native
model distinct from the window-capture block-motion and Apple comparison paths.

For source compilation and explicit local runtime assembly, see
[`docs/FG_SOURCE_BUILD.md`](../../docs/FG_SOURCE_BUILD.md). The build command
performs no inference. Assembly validates the supported local model buffers
and math tables but does not obtain them or establish redistribution rights.
