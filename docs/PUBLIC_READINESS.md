# Source release readiness

This is the experimental complete-engine source release. The original research
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

## Open provenance review and release limits

The 22 marked runtime-derived shaders have unresolved input distribution rights.
The compiler's license, successful tests and public availability do not establish
that basis. The review remains open; establishing a distribution basis or
replacing the affected implementations remains source-origin work. Local
generation from user-supplied inputs is a possible future approach, not a
verified feature of this release.

Model/runtime bootstrap requirements remain explicit in [setup](MODEL_SETUP.md)
and [the complete source build](COMPLETE_SOURCE_BUILD.md). A source release is not
a one-click installer, universal game support, or a guarantee of exactness in all
modes and geometries. See [repository governance](REPOSITORY_GOVERNANCE.md) for
the required checks and how to verify branch-rule enforcement on GitHub.

The maintainer authorized publication of the complete source as an experimental
release. This records the release scope; it does not grant third-party rights
or represent a completed legal review.
