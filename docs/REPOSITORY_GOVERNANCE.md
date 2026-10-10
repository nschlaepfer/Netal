# Repository governance and CI

## Current enforcement boundary

This source repository has a prepared default-branch ruleset. A checked-in configuration is not enforcement: GitHub must accept it and report it active. The account's current private-repository plan does not support rulesets; availability must be rechecked when the authorized public release is ready. [GitHub ruleset availability](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/about-rulesets).

## Rules prepared for the default branch

The [ruleset](../.github/rulesets/main.json) requires pull requests, resolved review discussions, a linear history, and successful up-to-date `app`, `repository-policy` and `engine-source` checks from GitHub Actions. It blocks branch deletion and force pushes and has no bypass actors.

The repository currently has one maintainer. The rule requires zero additional peer approvals so the maintainer can merge a self-reviewed PR after checks pass. Contributions still require maintainer review under [CONTRIBUTING.md](../CONTRIBUTING.md). CODEOWNERS requests that review but does not enforce it by itself. With a second trusted maintainer, change the required count to one and enable code-owner approval. No automatic merging is configured.

Preview the rule without changing GitHub:

```sh
python3 scripts/apply-github-ruleset.py --repo nschlaepfer/Netal-public
```

After plan availability and explicit maintainer direction, apply it:

```sh
python3 scripts/apply-github-ruleset.py --repo nschlaepfer/Netal-public --apply
```

The helper checks all required CI results, creates or updates only the named ruleset and verifies enforcement. It does not change visibility, buy a plan or alter other rulesets. A failed activation must be reported as blocked, not described as protected.

## CI

- `app`: macOS 26 SDK runner; app tests, synthetic launch/import/runtime checks, offline sequence tests and app build. No game, proprietary model provisioning or performance claim follows.
- `engine-source`: compiles all FG kernels/host archives, SR reference/boundary/texture kernels, and both bridges without models. This is a build check, not an inference test.
- `repository-policy`: Linux runner; validates its boundary tests and inspects changed Git blobs for common restricted payloads, credentials and personal home paths. On PRs the checker comes from the base branch, not the submitted implementation.
- Both workflows use read-only repository permissions, an immutable checkout action SHA, disabled persisted checkout credentials, timeouts and cancellation of superseded runs. No publishing/signing secret is supplied to a PR.
- Dependabot proposes weekly GitHub Actions updates. Pin updates must be reviewed; no automatic approval or merge is configured.

The content guard is incremental. Existing research artifacts and history are not retroactively cleared by a green PR. Binary/data provenance, complete privacy review and model rights remain explicit public-release gates. Image signatures and pattern scans cannot determine whether an image contains personal or proprietary information.

## PR and issue process

PRs state the problem, result, validation, provenance, compatibility/recovery and AI involvement. Rendering reports include exact hardware, software, DLL/model hashes, geometry and backend. Contributors must explain and verify their own patches. The app bug template, rendering form and feature form separate reproducible issues from setup questions.

The approach is informed by [PCSX2](https://github.com/PCSX2/pcsx2/blob/master/.github/PULL_REQUEST_TEMPLATE.md), [DXVK](https://github.com/doitsujin/dxvk/blob/master/.github/PULL_REQUEST_TEMPLATE/pull_request_template.md) and [MoltenVK](https://github.com/KhronosGroup/MoltenVK/blob/main/.github/workflows/CI.yml), adapted to Netal's single-maintainer research stage. It does not copy their platform matrix, restrictions or support promises.
