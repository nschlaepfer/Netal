#!/usr/bin/env python3
"""Exercise fail-closed canonical L9 receipt mutations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flip(path: Path, offset: int) -> None:
    data = bytearray(path.read_bytes())
    data[offset] ^= 1
    path.write_bytes(data)


def edit(path: Path, mutate: Callable[[dict], None]) -> None:
    value = json.loads(path.read_text())
    mutate(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtx-root", type=Path, required=True)
    parser.add_argument("--mac-root", type=Path, required=True)
    parser.add_argument("--rig", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parent.parent
    verifier = repository / "scripts/verify-fg-early-l9.py"
    rtx_run = Path("l9-poison-a5-process-a")
    mac_run = Path("l9-poison-5a-process-b")
    cases: list[tuple[str, str, Callable[[Path, Path, Path], None]]] = [
        ("rtx-output", "output receipt hash drift",
         lambda r, m, g: flip(r / rtx_run / "output.bin", 17)),
        ("rtx-input", "generated input drift",
         lambda r, m, g: flip(r / rtx_run / "input-after.bin", 19)),
        ("rtx-dead", "generated input drift",
         lambda r, m, g: flip(r / rtx_run / "dead-mask-after.bin", 23)),
        ("rtx-module", "module hash drift",
         lambda r, m, g: edit(r / rtx_run / "run-receipt.json",
                              lambda v: v.__setitem__("module_sha256", "0" * 64))),
        ("rtx-expected", "forbidden ingress",
         lambda r, m, g: edit(r / rtx_run / "run-receipt.json",
                              lambda v: v.__setitem__("expected_output_opened", True))),
        ("rtx-canary", "integrity gate failed",
         lambda r, m, g: edit(r / rtx_run / "run-receipt.json",
                              lambda v: v.__setitem__("canary_mismatched_bytes", 1))),
        ("mac-output", "output receipt hash drift",
         lambda r, m, g: flip(m / mac_run / "output.bin", 29)),
        ("mac-rom", "RCP ROM drift",
         lambda r, m, g: edit(m / mac_run / "run-receipt.json",
                              lambda v: v.__setitem__("architecture_rom_sha256", "1" * 64))),
        ("mac-rounding", "surface rounding drift",
         lambda r, m, g: edit(m / mac_run / "run-receipt.json",
                              lambda v: v.__setitem__("surface_half_rounding", "nearest"))),
        ("mac-archive", "forbidden ingress",
         lambda r, m, g: edit(m / mac_run / "run-receipt.json",
                              lambda v: v.__setitem__("archived_intermediate_opened", True))),
        ("rig-source", "rig source identity drift",
         lambda r, m, g: edit(g, lambda v: v.__setitem__("source_sha256", "2" * 64))),
        ("address", "address-like value",
         lambda r, m, g: edit(r / rtx_run / "run-receipt.json",
                              lambda v: v.__setitem__("debug", "0x1234567890abcdef"))),
    ]

    def duplicate(rtx: Path, mac: Path, rig: Path) -> None:
        first = json.loads((rtx / "l9-poison-a5-process-a/run-receipt.json").read_text())
        edit(rtx / "l9-poison-a5-process-b/run-receipt.json",
             lambda value: value.__setitem__("process_id", first["process_id"]))

    cases.append(("duplicate-process", "fresh-process gate failed", duplicate))
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    results = []
    with tempfile.TemporaryDirectory(prefix="fg-early-l9-negative-") as temporary:
        for name, expected, mutate in cases:
            case_root = Path(temporary) / name
            rtx = case_root / "rtx"
            mac = case_root / "mac"
            rig = case_root / "rig.json"
            shutil.copytree(args.rtx_root, rtx)
            shutil.copytree(args.mac_root, mac)
            shutil.copy2(args.rig, rig)
            mutate(rtx, mac, rig)
            rtx_receipt = case_root / "rtx.json"
            mac_receipt = case_root / "mac.json"
            final_receipt = case_root / "final.json"
            commands = (
                ["python3", str(verifier), "backend", "--backend", "rtx",
                 "--root", str(rtx), "--output", str(rtx_receipt)],
                ["python3", str(verifier), "backend", "--backend", "mac",
                 "--root", str(mac), "--output", str(mac_receipt)],
                ["python3", str(verifier), "final", "--rtx", str(rtx_receipt),
                 "--mac", str(mac_receipt), "--rig", str(rig),
                 "--output", str(final_receipt)],
            )
            observed_output = ""
            exit_code = 0
            for command in commands:
                process = subprocess.run(
                    command, cwd=repository, env=environment, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    check=False,
                )
                observed_output = process.stdout
                exit_code = process.returncode
                if exit_code != 0:
                    break
            observed = next((line for line in observed_output.splitlines()
                             if line.startswith("verification_error=")),
                            observed_output.strip())
            if exit_code != 2 or expected not in observed or final_receipt.exists():
                raise RuntimeError(
                    f"mutation {name} did not fail closed: "
                    f"exit={exit_code}, output={observed_output!r}"
                )
            results.append({
                "mutation": name,
                "expected_error_fragment": expected,
                "observed_error": observed,
                "exit_code": exit_code,
                "final_receipt_created": final_receipt.exists(),
                "result": "pass",
            })
    audit = {
        "schema": "neuralstudio.fg.early-l9-negative-audit.v1",
        "verifier_sha256": sha(verifier),
        "rig_receipt_sha256": sha(args.rig),
        "mutation_count": len(results),
        "all_rejected": True,
        "mutations": results,
        "result": "pass",
    }
    audit["audit_root_sha256"] = hashlib.sha256(
        json.dumps(audit, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(f"mutation_count={len(results)}")
    print(f"audit_root_sha256={audit['audit_root_sha256']}")
    print("result=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
