# Repository governance and CI

## Current enforcement boundary

The public source repository uses the default-branch policy below. A checked-in configuration alone is not enforcement: inspect [the live repository rules](https://github.com/nschlaepfer/Netal/rules) or the GitHub API to verify that the named ruleset is active. The original research history is held separately in a private repository.

## Default-branch rules

The [ruleset](../.github/rulesets/main.json) requires pull requests, resolved review discussions, a linear history, and successful up-to-date `app`, `repository-policy` and `engine-source` checks from GitHub Actions. It blocks branch deletion and force pushes and has no bypass actors.

The repository currently has one maintainer. The rule requires zero additional peer approvals so the maintainer can merge a self-reviewed PR after checks pass. Contributions still require maintainer review under [CONTRIBUTING.md](../CONTRIBUTING.md). CODEOWNERS requests that review but does not enforce it by itself. With a second trusted maintainer, change the required count to one and enable code-owner approval. No automatic merging is configured.

Preview the rule without changing GitHub:

```sh
python3 scripts/apply-github-ruleset.py --repo nschlaepfer/Netal
```

To apply or update it under maintainer direction:

```sh
python3 scripts/apply-github-ruleset.py --repo nschlaepfer/Netal --apply
```

The helper checks all required CI results, creates or updates only the named ruleset and verifies enforcement. It does not change visibility, buy a plan or alter other rulesets. A failed activation must be reported as blocked, not described as protected.

## CI

- `app`: macOS 26 SDK runner; app tests, synthetic launch/import/runtime checks, offline sequence tests and app build. No game, proprietary model provisioning or performance claim follows.
- `engine-source`: compiles all FG kernels/host archives, SR reference/boundary/texture kernels, and both bridges without models. This is a build check, not an inference test.
- `repository-policy`: Linux runner; validates its boundary tests and inspects changed Git blobs for common restricted payloads, credentials and personal home paths. On PRs the checker comes from the base branch, not the submitted implementation.
- All three workflows use read-only repository permissions, an immutable checkout action SHA, disabled persisted checkout credentials, timeouts and cancellation of superseded runs. No publishing/signing secret is supplied to a PR.
- Dependabot proposes weekly GitHub Actions updates. Pin updates must be reviewed; no automatic approval or merge is configured.

The content guard is incremental. Existing artifacts and history are not retroactively cleared by a green PR. Binary/data provenance, privacy review and model rights require separate review. The [source-origin review](SOURCE_PROVENANCE_REVIEW.md) records the unresolved translated-input questions in this release. Image signatures and pattern scans cannot determine whether an image contains personal or proprietary information.

## PR and issue process

PRs state the problem, result, validation, provenance, compatibility/recovery and AI involvement. Rendering reports include exact hardware, software, DLL/model hashes, geometry and backend. Contributors must explain and verify their own patches. The app bug template, rendering form and feature form separate reproducible issues from setup questions.

The approach is informed by [PCSX2](https://github.com/PCSX2/pcsx2/blob/master/.github/PULL_REQUEST_TEMPLATE.md), [DXVK](https://github.com/doitsujin/dxvk/blob/master/.github/PULL_REQUEST_TEMPLATE/pull_request_template.md) and [MoltenVK](https://github.com/KhronosGroup/MoltenVK/blob/main/.github/workflows/CI.yml), adapted to Netal's single-maintainer research stage. It does not copy their platform matrix, restrictions or support promises.
