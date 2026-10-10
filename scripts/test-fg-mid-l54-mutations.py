#!/usr/bin/env python3
"""Exercise fail-closed mutations for the selected FG launch-54 receipts."""

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


Mutation = Callable[[Path, Path, Path], None]


def sha256(path: Path) -> str:
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
    parser.add_argument("--rig-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parent.parent
    verifier = repository / "scripts/verify-fg-mid-l54.py"
    rtx_run = Path("l54-poison-a5-process-a")
    mac_run = Path("l54-poison-5a-process-b")
    cases: list[tuple[str, str, Mutation]] = []

    cases.append(("rtx-output-byte", "output receipt hash drift",
                  lambda rtx, mac, rig: flip(rtx / rtx_run / "output.bin", 17)))
    cases.append(("rtx-texture-byte", "generated texture drift",
                  lambda rtx, mac, rig: flip(rtx / rtx_run / "texture-a-after.bin", 19)))
    cases.append(("rtx-offset-byte", "generated offset drift",
                  lambda rtx, mac, rig: flip(rtx / rtx_run / "offset-b-after.bin", 23)))

    def duplicate_rtx_process(rtx: Path, mac: Path, rig: Path) -> None:
        source = json.loads((rtx / "l54-poison-a5-process-a/run-receipt.json").read_text())
        edit(rtx / "l54-poison-a5-process-b/run-receipt.json",
             lambda value: value.__setitem__("process_id", source["process_id"]))

    cases.append(("rtx-duplicate-process", "fresh-process gate failed",
                  duplicate_rtx_process))
    cases.append(("rtx-module-hash", "module hash drift",
                  lambda rtx, mac, rig: edit(
                      rtx / rtx_run / "run-receipt.json",
                      lambda value: value.__setitem__("module_sha256", "0" * 64))))
    cases.append(("rtx-expected-ingress", "expected output was opened",
                  lambda rtx, mac, rig: edit(
                      rtx / rtx_run / "run-receipt.json",
                      lambda value: value.__setitem__("expected_output_opened", True))))
    cases.append(("rtx-archive-ingress", "archived intermediate was opened",
                  lambda rtx, mac, rig: edit(
                      rtx / rtx_run / "run-receipt.json",
                      lambda value: value.__setitem__("archived_intermediate_opened", True))))
    cases.append(("rtx-canary", "input/canary integrity failed",
                  lambda rtx, mac, rig: edit(
                      rtx / rtx_run / "run-receipt.json",
                      lambda value: value.__setitem__("canary_mismatched_bytes", 1))))
    cases.append(("rtx-address", "address-like value",
                  lambda rtx, mac, rig: edit(
                      rtx / rtx_run / "run-receipt.json",
                      lambda value: value.__setitem__("debug_value", "0x1234567890abcdef"))))
    cases.append(("mac-output-byte", "output receipt hash drift",
                  lambda rtx, mac, rig: flip(mac / mac_run / "output.bin", 29)))
    cases.append(("mac-point-fraction", "point-coordinate rule drift",
                  lambda rtx, mac, rig: edit(
                      mac / mac_run / "run-receipt.json",
                      lambda value: value.__setitem__(
                          "normalized_point_coordinate_fraction_bits", 11))))
    cases.append(("mac-half-rounding", "half-store rule drift",
                  lambda rtx, mac, rig: edit(
                      mac / mac_run / "run-receipt.json",
                      lambda value: value.__setitem__(
                          "surface_half_rounding", "nearest-even"))))
    cases.append(("mac-rcp-rom", "RCP ROM drift",
                  lambda rtx, mac, rig: edit(
                      mac / mac_run / "run-receipt.json",
                      lambda value: value.__setitem__(
                          "architecture_rom_sha256", "f" * 64))))

    def duplicate_mac_process(rtx: Path, mac: Path, rig: Path) -> None:
        source = json.loads((mac / "l54-poison-5a-process-a/run-receipt.json").read_text())
        edit(mac / "l54-poison-5a-process-b/run-receipt.json",
             lambda value: value.__setitem__("process_id", source["process_id"]))

    cases.append(("mac-duplicate-process", "fresh-process gate failed",
                  duplicate_mac_process))
    cases.append(("mac-result", "result drift",
                  lambda rtx, mac, rig: edit(
                      mac / mac_run / "run-receipt.json",
                      lambda value: value.__setitem__("result", "fail"))))
    cases.append(("rig-module-hash", "RTX rig module hash drift",
                  lambda rtx, mac, rig: edit(
                      rig, lambda value: value.__setitem__("module_sha256", "1" * 64))))
    cases.append(("rig-source-hash", "RTX rig source identity drift",
                  lambda rtx, mac, rig: edit(
                      rig, lambda value: value.__setitem__("source_sha256", "2" * 64))))

    results = []
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryDirectory(prefix="fg-mid-l54-negative-") as temporary:
        temporary_root = Path(temporary)
        for name, expected, mutate in cases:
            case_root = temporary_root / name
            rtx = case_root / "rtx"
            mac = case_root / "mac"
            rig = case_root / "rig-receipt.json"
            shutil.copytree(args.rtx_root, rtx)
            shutil.copytree(args.mac_root, mac)
            shutil.copy2(args.rig_receipt, rig)
            mutate(rtx, mac, rig)
            rtx_receipt = case_root / "rtx-backend.json"
            mac_receipt = case_root / "mac-backend.json"
            final_receipt = case_root / "should-not-exist.json"
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
            observed = next(
                (line for line in observed_output.splitlines()
                 if line.startswith("verification_error=")),
                observed_output.strip(),
            )
            passed = (
                exit_code == 2 and expected in observed and
                not final_receipt.exists()
            )
            if not passed:
                raise RuntimeError(
                    f"mutation {name} did not fail closed: exit={exit_code} "
                    f"output={observed_output!r}"
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
        "schema": "neuralstudio.fg.mid-l54-negative-audit.v1",
        "verifier_sha256": sha256(verifier),
        "rtx_rig_receipt_sha256": sha256(args.rig_receipt),
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
