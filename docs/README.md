# Netal documentation

Start with [current engine status](STATUS.md),
[supported DLLs and setup](MODEL_SETUP.md), [public release readiness](PUBLIC_READINESS.md),
and [repository rules and CI](REPOSITORY_GOVERNANCE.md).
The many component ledgers, dated experiment reports and older integration notes
describe their recorded checkpoints; they are not all current release promises.
In particular, startup-only packaging tests do not supersede the documented
working Resident Evil SR/FG integration.

- [Boundary kernels and math](SR_BOUNDARY_KERNELS_AND_MATH.md): source builds, scalar conformance and generated arithmetic assets.
- [Source provenance review](SOURCE_PROVENANCE_REVIEW.md): identified compiler licenses and unresolved translated-input rights.
- [Getting started](getting-started.md): build the app and connect a prepared runtime.
- [SR boundary metadata](SR_BOUNDARY_METADATA.md): local DLL-derived templates, generated profiles, and render-size correction.
- [SR arithmetic modes](SR_ARITHMETIC_MODES.md): validated numerical scope and honest mode controls.
- [SR body source build](SR_SOURCE_BUILD.md): generated layouts, model repacking, pipeline recipes, and current regression limits.
- [FG engine source build](FG_SOURCE_BUILD.md): compile shaders and universal host libraries without models or inference.
- [Local SR model setup](MODEL_SETUP.md): build capture helpers and import the supported model locally.
- [Release gate](RELEASE.md): SR + FG scope, experimental NR, and remaining blockers.
- [Architecture](architecture.md): app, game bridge, and runtime boundaries.
- [Menu-bar app](menubar.md): feature state, model integrity, and setup.
- [Runtime bundles](RUNTIME_BUNDLE.md): locally supplied inputs and bootstrap limits.
- [Performance](performance.md): what a valid comparison must record.
- [Roadmap](roadmap.md): the order of release work.

The source candidate includes the complete SR/FG engine implementation. Private journals, captured model data and old Git history are excluded. See [complete source builds](COMPLETE_SOURCE_BUILD.md).
