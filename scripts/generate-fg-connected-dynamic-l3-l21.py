#!/usr/bin/env python3
"""Generate live L20-L21 extensions from the accepted connected early hosts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def transform(source: str, backend: str) -> str:
    if backend == "mac":
        marker = (
            '      checkpoint("generated-module5", {model_input_bytes,\n'
            '          auxiliary_mask_bytes, auxiliary_vector_bytes});\n\n'
            '      std::size_t input_mismatches = 0;')
        replacement = (
            '      checkpoint("generated-module5", {model_input_bytes,\n'
            '          auxiliary_mask_bytes, auxiliary_vector_bytes});\n'
            '#include "mac_tail.inc"\n\n'
            '      std::size_t input_mismatches = 0;')
        pass_marker = (
            'const bool pass = input_mismatches == 0 && canary_mismatches == 0 &&\n'
            '                        checkpoints.size() == 21;')
        pass_replacement = (
            'const bool pass = input_mismatches == 0 && canary_mismatches == 0 &&\n'
            '                        l20_l21_pass && checkpoints.size() == 23;')
    else:
        marker = (
            '    checkpoint("generated-module5", {\n'
            '        model_input_bytes, auxiliary_mask_bytes, auxiliary_vector_bytes});\n\n'
            '    std::size_t input_mismatches = 0;')
        replacement = (
            '    checkpoint("generated-module5", {\n'
            '        model_input_bytes, auxiliary_mask_bytes, auxiliary_vector_bytes});\n'
            '#include "rtx_tail.inc"\n\n'
            '    std::size_t input_mismatches = 0;')
        pass_marker = (
            'const bool pass = input_mismatches == 0 && canary_mismatches == 0 &&\n'
            '                      allocation37_poison_halfwords == 0 &&\n'
            '                      checkpoints.size() == 21;')
        pass_replacement = (
            'const bool pass = input_mismatches == 0 && canary_mismatches == 0 &&\n'
            '                      allocation37_poison_halfwords == 0 &&\n'
            '                      l20_l21_pass && checkpoints.size() == 23;')
    if source.count(marker) != 1:
        raise RuntimeError(f"{backend}: injection marker drift")
    if source.count(pass_marker) != 1:
        raise RuntimeError(f"{backend}: pass marker drift")
    if source.count("neuralstudio.fg.connected-dynamic-early-run.v1") != 1:
        raise RuntimeError(f"{backend}: schema marker drift")
    if source.count('selected_launch_count\\\": 21') != 1:
        raise RuntimeError(f"{backend}: launch-count marker drift")
    result = source.replace(marker, replacement)
    result = result.replace(pass_marker, pass_replacement)
    result = result.replace(
        "neuralstudio.fg.connected-dynamic-early-run.v1",
        "neuralstudio.fg.connected-dynamic-l3-l21-run.v1")
    result = result.replace('selected_launch_count\\\": 21',
                            'selected_launch_count\\\": 23')
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = args.repository.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = {
        "mac": repository / "src/fg_connected_dynamic_early/macos_connected.mm",
        "rtx": repository / "src/fg_connected_dynamic_early/rtx_connected.cpp",
    }
    generated = {}
    for backend, path in sources.items():
        raw = path.read_bytes()
        result = transform(raw.decode(), backend).encode()
        destination = output / ("generated_macos.mm" if backend == "mac"
                                else "generated_rtx.cpp")
        destination.write_bytes(result)
        generated[backend] = {
            "source": path.relative_to(repository).as_posix(),
            "source_sha256": digest(raw),
            "generated": destination.name,
            "generated_sha256": digest(result),
            "injected_tail": f"{backend}_tail.inc",
        }
    receipt = {
        "schema": "neuralstudio.fg.connected-dynamic-l3-l21-generator.v1",
        "generated": generated,
        "schema_replacements": 2,
        "launch_count_replacements": 2,
        "pass_gate_replacements": 2,
        "live_tail_injections": 2,
        "expected_or_archived_intermediate_embedded": False,
        "result": "pass",
    }
    receipt["receipt_root_sha256"] = digest(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode())
    (output / "generator-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"generator_receipt_root_sha256={receipt['receipt_root_sha256']}")
    print("result=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
