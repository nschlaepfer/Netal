# Contributing to Netal

The release scope is SR, FG and the macOS app. NR remains experimental. Successful RE9/CrossOver integration is documented; a complete public runtime, fresh installation and broader compatibility are separate work.

## Pull requests

Open a focused PR with the concrete problem, resulting behavior, rationale and validation. Discuss large architecture or new-model changes in an issue first. Keep unrelated formatting, generated artifacts and optimizations separate. The maintainer reviews every contribution; contributors must be able to explain their code and its origin.

Use the PR template. Report AI assistance and how the implementation was checked. Generated code has the same correctness, provenance and testing requirements as other code. Do not submit unreviewed generated patches, invented test results or summaries you cannot substantiate.

For rendering changes, record hardware/OS/toolkit, game build when applicable, exact model/DLL hash, source and output geometry, arithmetic mode, reference identity, reset/history behavior and actual commands/results. Distinguish synthetic correctness, game generation, presentation, visual quality and performance. Preserve known failures; do not weaken a threshold or treat a missing reference as a pass. Document fallback, unsupported geometry, synchronization/lifetime effects and recovery.

Required CI contexts are `app`, `repository-policy` and `engine-source` once branch rules are enabled. Resolve review discussions and update against the target branch. Prefer squash merging one focused change. Do not force-push or delete the default branch. CODEOWNERS requests maintainer attention; it does not itself enforce approval.

The prepared ruleset requires PRs, passing checks and resolved discussions. It uses zero mandatory peer approvals because there is one maintainer: contributor PRs still need maintainer review, while maintainer-authored PRs document self-review. Increase the required approval count and enable code-owner approval when another trusted maintainer joins. Current GitHub plan restrictions may prevent activation while private; [the governance status](docs/REPOSITORY_GOVERNANCE.md) distinguishes configured files from enforced rules.

## Tests and evidence

```sh
zsh scripts/test-macos-netal-menubar.sh
python3 scripts/test-fg-offline-sequence.py
python3 scripts/test-public-source.py
python3 scripts/test-contribution-content.py
```

Run the focused SR/FG/import/routing tests for the code you change. Build with `zsh scripts/build-macos-netal-menubar.sh`. CI uses synthetic fixtures without proprietary models or game installs. It does not validate game rendering, NVIDIA parity or GPU performance. Hardware-dependent changes need separately recorded local evidence and must keep that limit visible.

## Permitted inputs and source provenance

Users supply their own supported DLL under its applicable terms. The filename alone is not enough: the tested DLL/model identities are pinned in [model setup](docs/MODEL_SETUP.md). A newer or different DLSS DLL is not automatically supported. Do not upload a DLL to report an import failure; report its filename, version and SHA-256.

Do not add weights, captured tensors, vendor binaries/headers, game assets, private captures, signing material or credentials. Keep experiments in ignored local directories. Prefer procedural reproductions and redact logs/images before sharing. Five explicitly pinned, model-independent scalar calibration files are the only binary data exceptions; their hashes and origins are checked. The content-policy check rejects other newly added payloads and common secret/path patterns; it is not a complete privacy or licensing audit.

Imported/adapted material needs a source URL, revision, applicable license and preserved notices. Compiler licensing does not clear translated input programs. Do not claim ownership or a permissive license merely because code was generated. Review the entire history before any repository visibility change; the existing private research tree is not a public export.

## Issue reports and conduct

Search for an existing issue. Use the rendering form for graphics/game failures and include a minimal reproduction, expected/actual behavior and a known-good revision where available. File setup questions with the exact supported input identity and environment. Do not post exploit details or secrets publicly; contact the maintainer privately for sensitive findings.

Keep discussion technical and respectful. Critique code and evidence, not people. Harassment, threats, personal-data disclosure, plagiarism and knowingly false evidence are not accepted. Maintainers may close unsupported, duplicative or abusive submissions with an explanation.

## Reference practices

These rules adapt the shape of [PCSX2's PR template](https://github.com/PCSX2/pcsx2/blob/master/.github/PULL_REQUEST_TEMPLATE.md), its reproducible environment reports and AI disclosure; [DXVK's contributor responsibility](https://github.com/doitsujin/dxvk/blob/master/.github/PULL_REQUEST_TEMPLATE/pull_request_template.md); and [MoltenVK's platform CI](https://github.com/KhronosGroup/MoltenVK/blob/main/.github/workflows/CI.yml). Wording is specific to Netal; no upstream policy is presented as an endorsement.
