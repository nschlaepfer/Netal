# Packages

Reference package manifests validated against
[`spec/manifest/v1`](../spec/manifest/v1/README.md).

| Package | Backend | Distribution | Temporal | Redistributable |
|---|---|---|---|---|
| `org.neuralstudio.test.passthrough` | `test` | bundled | stateless | yes (MIT) |
| `com.nvidia.ngx.dlss-nr` | `ngx` | adapter-located | stateful, soft reset | **no** |

These two exist to keep the manifest schema honest. They differ on both axes
that matter — open versus proprietary provenance, and stateless versus stateful
temporal handling — so a schema change that breaks either shape is caught
immediately.

## Current validation state

The open test package carries the exact reproducible hash of its built adapter
and validates end to end. The NGX manifest deliberately retains an all-zero
placeholder until its NeuralStudio adapter exists; the validator must fail that
package rather than accepting an unverified binary.

Package artifact signatures and provider-runtime signatures are separate trust
domains. A custom NeuralStudio adapter cannot inherit trust from an NVIDIA-
signed runtime that it loads.

Once `ns_adapter_ngx.dll` is built, its packaging step replaces the placeholder
with the real file hash.

## Packaging gate

`com.nvidia.ngx.dlss-nr` declares `provenance.redistribution.allowed: false`.
Any release package that includes it must fail the packaging gate. That is the
mechanical enforcement of the v1 rule that a release contains only artifacts
NeuralStudio is authorized to redistribute, and of the repository rule that
private visibility is not itself a redistribution right.
