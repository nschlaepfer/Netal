#!/usr/bin/env python3
"""Exercise fail-closed launch-85 receipt mutations and emit an audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flip(path: Path, offset: int = 0) -> None:
    data = bytearray(path.read_bytes())
    data[offset] ^= 1
    path.write_bytes(data)


def edit_json(path: Path, mutate: Callable[[dict], None]) -> None:
    value = json.loads(path.read_text())
    mutate(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtx-root", type=Path, required=True)
    parser.add_argument("--mac-root", type=Path, required=True)
    parser.add_argument("--rcp-rom", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parent.parent
    verifier = repository / "scripts/verify-fg-postprocess-l85.py"
    cases: list[tuple[str, str, Callable[[Path, Path, Path], None]]] = []

    cases.append(("rtx-output-byte", "output receipt hash drift",
                  lambda rtx, mac, rom: flip(rtx / "poison-a5-process-a/output.bin", 17)))
    cases.append(("rtx-generated-input-byte", "generated input drift",
                  lambda rtx, mac, rom: flip(rtx / "poison-a5-process-a/input-after.bin", 3)))
    cases.append(("rtx-div-record-byte", "probe record hash drift",
                  lambda rtx, mac, rom: flip(rtx / "poison-a5-process-a/div-records.bin", 20)))

    def duplicate_pid(rtx: Path, mac: Path, rom: Path) -> None:
        source = json.loads((rtx / "poison-a5-process-a/run-receipt.json").read_text())
        edit_json(
            rtx / "poison-a5-process-b/run-receipt.json",
            lambda value: value.__setitem__("process_id", source["process_id"]),
        )

    cases.append(("rtx-duplicate-process", "fresh selected process gate failed", duplicate_pid))
    cases.append(("rtx-inactive-level-write", "inactive level was written",
                  lambda rtx, mac, rom: flip(rtx / "poison-5a-process-a/inactive-level2-after.bin", 0)))

    def module_drift(rtx: Path, mac: Path, rom: Path) -> None:
        edit_json(
            rtx / "poison-5a-process-b/run-receipt.json",
            lambda value: value.__setitem__("module_sha256", "0" * 64),
        )

    cases.append(("rtx-module-identity", "module hash drift", module_drift))

    def archived_ingress(rtx: Path, mac: Path, rom: Path) -> None:
        edit_json(
            rtx / "poison-5a-process-b/run-receipt.json",
            lambda value: value.__setitem__("archived_intermediates_consumed", True),
        )

    cases.append(("rtx-archived-ingress", "archived intermediate consumed", archived_ingress))
    cases.append(("mac-output-byte", "output receipt hash drift",
                  lambda rtx, mac, rom: flip(mac / "poison-a5-process-a/output.bin", 19)))

    def erase_strict_counterexample(rtx: Path, mac: Path, rom: Path) -> None:
        for run in (
            "poison-a5-process-a", "poison-a5-process-b",
            "poison-5a-process-a", "poison-5a-process-b",
        ):
            root = mac / run
            exact = (root / "output.bin").read_bytes()
            (root / "strict-division-output.bin").write_bytes(exact)
            edit_json(
                root / "run-receipt.json",
                lambda value, digest=hashlib.sha256(exact).hexdigest():
                    value.__setitem__("strict_division_output_sha256", digest),
            )

    cases.append(("mac-ordinary-division-counterexample", "counterexample vanished",
                  erase_strict_counterexample))
    cases.append(("rcp-rom-byte", "RCP ROM hash drift",
                  lambda rtx, mac, rom: flip(rom, 12345)))

    results = []
    with tempfile.TemporaryDirectory(prefix="fg-postprocess-l85-negative-") as temporary:
        temporary_root = Path(temporary)
        for name, expected, mutate in cases:
            case_root = temporary_root / name
            rtx = case_root / "rtx"
            mac = case_root / "mac"
            rom = case_root / "rcp-rom.bin"
            shutil.copytree(args.rtx_root, rtx)
            shutil.copytree(args.mac_root, mac)
            rom.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.rcp_rom, rom)
            mutate(rtx, mac, rom)
            output = case_root / "should-not-exist.json"
            process = subprocess.run(
                [
                    "python3", str(verifier), "final",
                    "--rtx-root", str(rtx), "--mac-root", str(mac),
                    "--rcp-rom", str(rom), "--output", str(output),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            observed = next(
                (line for line in process.stdout.splitlines()
                 if line.startswith("verification_error=")),
                process.stdout.strip(),
            )
            passed = process.returncode == 2 and expected in observed and not output.exists()
            if not passed:
                raise RuntimeError(
                    f"mutation {name} did not fail closed as expected: "
                    f"exit={process.returncode} output={process.stdout!r}"
                )
            results.append({
                "mutation": name,
                "expected_error_fragment": expected,
                "observed_error": observed,
                "exit_code": process.returncode,
                "receipt_created": output.exists(),
                "result": "pass",
            })

    audit = {
        "schema": "neuralstudio.fg.postprocess-l85-negative-audit.v1",
        "verifier_sha256": sha256(verifier),
        "rtx_backend_receipt_sha256": sha256(args.rtx_root / "backend-receipt.json"),
        "mac_backend_receipt_sha256": sha256(args.mac_root / "backend-receipt.json"),
        "rcp_rom_sha256": sha256(args.rcp_rom),
        "mutation_count": len(results),
        "all_rejected": True,
        "mutations": results,
        "result": "pass",
    }
    identity = dict(audit)
    audit["audit_root_sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(f"mutation_count={len(results)}")
    print(f"audit_root_sha256={audit['audit_root_sha256']}")
    print("result=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
