#!/usr/bin/env python3
"""Build a content-addressed DLSS Frame Generation intake manifest.

The source descriptor names files beneath a retrieved, job-scoped capture
directory.  This builder refuses path traversal and symlinks, verifies the
capture-side declared size/hash for every artifact, adds descriptor roots, and
emits a manifest suitable for the independent verifier.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any


REPOSITORY = Path(__file__).resolve().parent.parent
SPEC = REPOSITORY / "evidence" / "specs" / "fg-intake-v1" / "manifest.schema.json"
VERIFIER = REPOSITORY / "scripts" / "verify-fg-intake-manifest.py"
SOURCE_SCHEMA = "neuralstudio.fg.intake-source.v1"
MANIFEST_SCHEMA = "neuralstudio.fg.intake-manifest.v1"


class IntakeBuildError(RuntimeError):
    """Raised when the capture bundle cannot be safely manifested."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def digest_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def rooted(value: dict[str, Any], field: str) -> str:
    body = copy.deepcopy(value)
    body.pop(field, None)
    return digest(body)


def require_exact_keys(
    value: Any, required: set[str], label: str, optional: set[str] | None = None
) -> None:
    if not isinstance(value, dict):
        raise IntakeBuildError(f"{label} must be an object")
    optional = optional or set()
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required - optional)
    if missing:
        raise IntakeBuildError(f"{label} missing fields: {','.join(missing)}")
    if extra:
        raise IntakeBuildError(f"{label} unknown fields: {','.join(extra)}")


def safe_artifact_path(bundle_root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise IntakeBuildError("artifact relative_path must be a non-empty string")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise IntakeBuildError(f"unsafe artifact path: {relative}")
    if "\\" in relative:
        raise IntakeBuildError(f"artifact paths must use forward slashes: {relative}")
    root = bundle_root.resolve()
    lexical = root / Path(*pure.parts)
    parent = lexical.parent
    while parent != root:
        if parent.is_symlink():
            raise IntakeBuildError(f"artifact symlinks are forbidden: {relative}")
        parent = parent.parent
    if lexical.is_symlink():
        raise IntakeBuildError(f"artifact symlinks are forbidden: {relative}")
    path = lexical.resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise IntakeBuildError(f"artifact escapes bundle root: {relative}") from error
    if not path.is_file():
        raise IntakeBuildError(f"artifact is missing: {relative}")
    return path


def build_artifact_index(
    entries: Any, bundle_root: Path
) -> list[dict[str, Any]]:
    if not isinstance(entries, list) or not entries:
        raise IntakeBuildError("artifacts must be a non-empty array")
    built: list[dict[str, Any]] = []
    ids: set[str] = set()
    paths: set[str] = set()
    for index, entry in enumerate(entries):
        require_exact_keys(
            entry,
            {
                "id",
                "relative_path",
                "kind",
                "declared_bytes",
                "declared_sha256",
            },
            f"artifacts[{index}]",
        )
        artifact_id = entry["id"]
        relative = entry["relative_path"]
        if not isinstance(artifact_id, str) or not artifact_id:
            raise IntakeBuildError(f"artifacts[{index}].id is invalid")
        if artifact_id in ids:
            raise IntakeBuildError(f"duplicate artifact id: {artifact_id}")
        if relative in paths:
            raise IntakeBuildError(f"duplicate artifact path: {relative}")
        ids.add(artifact_id)
        paths.add(relative)
        path = safe_artifact_path(bundle_root, relative)
        actual_bytes = path.stat().st_size
        actual_sha256 = digest_file(path)
        if entry["declared_bytes"] != actual_bytes:
            raise IntakeBuildError(f"artifact byte count mismatch: {artifact_id}")
        if entry["declared_sha256"] != actual_sha256:
            raise IntakeBuildError(f"artifact SHA-256 mismatch: {artifact_id}")
        built.append(
            {
                "id": artifact_id,
                "relative_path": relative,
                "kind": entry["kind"],
                "bytes": actual_bytes,
                "sha256": actual_sha256,
            }
        )
    return sorted(built, key=lambda item: item["id"])


def _add_descriptor_roots(run: dict[str, Any]) -> None:
    collections = (
        ("loaded_modules", "module_descriptor_sha256"),
        ("resources", "resource_descriptor_sha256"),
        ("allocations", "allocation_descriptor_sha256"),
        ("transfers", "transfer_descriptor_sha256"),
        ("parameter_sets", "parameter_set_sha256"),
        ("selected_launches", "launch_descriptor_sha256"),
    )
    for collection_name, root_name in collections:
        collection = run.get(collection_name)
        if not isinstance(collection, list):
            continue
        parameter_roots = {
            item.get("evaluation_index"): item.get("parameter_set_sha256")
            for item in run.get("parameter_sets", [])
            if isinstance(item, dict) and item.get("phase") == "evaluate"
        }
        for item in collection:
            if not isinstance(item, dict):
                continue
            if collection_name == "selected_launches":
                item["parameter_set_sha256"] = parameter_roots.get(
                    item.get("evaluation_index")
                )
            item.pop(root_name, None)
            item[root_name] = rooted(item, root_name)
    temporal = run.get("temporal")
    if isinstance(temporal, dict) and isinstance(temporal.get("evaluations"), list):
        for evaluation in temporal["evaluations"]:
            if isinstance(evaluation, dict):
                evaluation.pop("temporal_descriptor_sha256", None)
                evaluation["temporal_descriptor_sha256"] = rooted(
                    evaluation, "temporal_descriptor_sha256"
                )
    for field, root_name in (
        ("run_isolation", "isolation_descriptor_sha256"),
        ("control_evidence", "control_descriptor_sha256"),
    ):
        descriptor = run.get(field)
        if isinstance(descriptor, dict):
            descriptor.pop(root_name, None)
            descriptor[root_name] = rooted(descriptor, root_name)


def _artifact_sha_map(manifest: dict[str, Any]) -> dict[str, str]:
    return {
        entry["id"]: entry["sha256"]
        for entry in manifest.get("artifacts", [])
        if isinstance(entry, dict)
        and isinstance(entry.get("id"), str)
        and isinstance(entry.get("sha256"), str)
    }


def _normalize_artifact_references(value: Any, artifacts: dict[str, str]) -> Any:
    """Replace per-run artifact IDs with their content identity."""
    if isinstance(value, list):
        return [_normalize_artifact_references(item, artifacts) for item in value]
    if not isinstance(value, dict):
        return value
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key.endswith("_descriptor_sha256") or key in {
            "run_root_sha256",
            "stream_token_sha256",
            "presentation_time_ns",
            "com_resource_token_sha256",
            "scenario_nonce_sha256",
            "parameter_set_sha256",
        }:
            continue
        if key == "artifact_id" or key.endswith("_artifact_id"):
            if item is None:
                result[("artifact_sha256" if key == "artifact_id" else key.replace("_artifact_id", "_artifact_sha256"))] = None
            elif isinstance(item, str):
                result[("artifact_sha256" if key == "artifact_id" else key.replace("_artifact_id", "_artifact_sha256"))] = artifacts.get(
                    item, f"missing:{item}"
                )
            else:
                result[("artifact_sha256" if key == "artifact_id" else key.replace("_artifact_id", "_artifact_sha256"))] = item
            continue
        if key.endswith("_artifact_ids") and isinstance(item, list):
            result[key.replace("_artifact_ids", "_artifact_sha256s")] = [
                artifacts.get(entry, f"missing:{entry}")
                for entry in item
            ]
            continue
        result[key] = _normalize_artifact_references(item, artifacts)
    return result


def semantic_projection(
    run: dict[str, Any], artifacts: dict[str, str]
) -> dict[str, Any]:
    selected = {
        key: copy.deepcopy(run.get(key))
        for key in (
            "environment",
            "loaded_modules",
            "resources",
            "allocations",
            "transfers",
            "parameter_sets",
            "selected_launches",
            "outputs",
            "temporal",
            "capture_completeness",
            "contamination_guard",
            "run_isolation",
            "control_evidence",
        )
    }
    return _normalize_artifact_references(selected, artifacts)


def recompute_derived_and_roots(manifest: dict[str, Any]) -> None:
    """Refresh all computed roots after assembling or adversarially mutating data."""
    artifacts = _artifact_sha_map(manifest)
    runs = manifest.get("runs", [])
    semantic_roots: list[str] = []
    if isinstance(runs, list):
        for run in runs:
            if not isinstance(run, dict):
                continue
            _add_descriptor_roots(run)
            run.pop("run_root_sha256", None)
            run["run_root_sha256"] = rooted(run, "run_root_sha256")
            semantic_roots.append(digest(semantic_projection(run, artifacts)))
    output_sequence = []
    if runs and isinstance(runs[0], dict):
        for output in runs[0].get("outputs", []):
            if isinstance(output, dict):
                output_sequence.append(
                    {
                        "sequence_index": output.get("sequence_index"),
                        "kind": output.get("kind"),
                        "content_sha256": output.get("content_sha256"),
                        "source_real_output_ids": output.get("source_real_output_ids"),
                    }
                )
    source_descriptor_sha256 = manifest.get("derived", {}).get(
        "source_descriptor_sha256"
    )
    semantic_equal = len(semantic_roots) == 2 and len(set(semantic_roots)) == 1
    manifest["derived"] = {
        "source_descriptor_sha256": source_descriptor_sha256,
        "artifact_index_sha256": digest(manifest.get("artifacts")),
        "job_descriptor_sha256": digest(manifest.get("job")),
        "runtime_descriptor_sha256": digest(manifest.get("runtime")),
        "run_semantic_sha256s": semantic_roots,
        "cross_run_semantic_sha256": semantic_roots[0] if semantic_equal else None,
        "semantic_repeat_equal": semantic_equal,
        "output_sequence_sha256": digest(output_sequence),
    }
    feature_coverage = manifest.get("feature_coverage")
    if isinstance(feature_coverage, dict):
        feature_coverage.pop("coverage_root_sha256", None)
        feature_coverage["coverage_root_sha256"] = rooted(
            feature_coverage, "coverage_root_sha256"
        )
    manifest.pop("manifest_root_sha256", None)
    manifest["manifest_root_sha256"] = rooted(manifest, "manifest_root_sha256")


def build_manifest(source: dict[str, Any], bundle_root: Path) -> dict[str, Any]:
    require_exact_keys(
        source,
        {
            "schema",
            "job",
            "runtime",
            "capture_policy",
            "artifacts",
            "runs",
            "opaque_dependencies",
            "access_issues",
            "claims",
            "feature_coverage",
        },
        "source",
    )
    if source["schema"] != SOURCE_SCHEMA:
        raise IntakeBuildError(f"unsupported source schema: {source['schema']}")
    if not SPEC.is_file() or not VERIFIER.is_file():
        raise IntakeBuildError("FG intake schema or verifier is missing")
    artifacts = build_artifact_index(source["artifacts"], bundle_root)
    manifest = copy.deepcopy(source)
    manifest["schema"] = MANIFEST_SCHEMA
    manifest["artifacts"] = artifacts
    manifest["tooling"] = {
        "builder_sha256": digest_file(Path(__file__).resolve()),
        "verifier_sha256": digest_file(VERIFIER),
        "schema_sha256": digest_file(SPEC),
    }
    manifest["derived"] = {
        "source_descriptor_sha256": digest(source),
    }
    recompute_derived_and_roots(manifest)
    return manifest


def _load_verifier() -> Any:
    spec = importlib.util.spec_from_file_location("fg_intake_verify", VERIFIER)
    if spec is None or spec.loader is None:
        raise IntakeBuildError("cannot load FG intake verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="capture source descriptor JSON")
    parser.add_argument(
        "--bundle-root",
        type=Path,
        help="retrieved capture root (defaults to the source file directory)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_path = args.source.resolve()
    bundle_root = (args.bundle_root or source_path.parent).resolve()
    try:
        source = json.loads(source_path.read_text(encoding="utf-8-sig"))
        manifest = build_manifest(source, bundle_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        verifier = _load_verifier()
        failures, blockers, report = verifier.verify_manifest(manifest, bundle_root)
    except (OSError, json.JSONDecodeError, IntakeBuildError) as error:
        print(f"FG intake build failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures or blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
