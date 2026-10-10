# Source release readiness

This is the separate complete-engine source candidate. The original research
repository remains private. Old commits, branches, issue history, LFS payloads,
private captures, model weights and compiled runtime/vendor binaries are not
part of this new source tree.

## Completed technical preparation

The full SR and FG kernels and host implementations compile from this tree.
Both game bridges link against the fresh FG archive. The source-built FG engine
reproduces the 96 recorded frame records; SR assembly from explicit local inputs
matches four retained 1080p RTX reference outputs. See [status](STATUS.md).

The privacy/content checks cover the intended files and fresh Git history.
Five explicitly pinned scalar calibration data files are the only binary data
exceptions. They are separately documented measured arithmetic data, not models.

## Still required before public visibility

The 22 marked runtime-derived shaders have unresolved input distribution rights.
The compiler's license and successful tests do not establish that basis. Resolve
that question or replace/generate the affected implementations from user-supplied
inputs, while preserving the complete engine and measured behavior.

Model/runtime bootstrap requirements remain explicit in [setup](MODEL_SETUP.md)
and [the complete source build](COMPLETE_SOURCE_BUILD.md). A source release is not
a one-click installer, universal game support, or a guarantee of exactness in all
modes and geometries. Branch-rule enforcement must be verified on GitHub.

The user authorized public release after complete-engine preparation. No extra
public approval is required once these conditions are met, but this file does
not itself establish that they are met or grant third-party rights.
