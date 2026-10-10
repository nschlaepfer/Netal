#!/usr/bin/env python3
"""Prove module23 ABI offset 0 is dead for the selected mode-1 launches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
from pathlib import Path


PTX_SHA = "bf264db5706405b55f13dc7e34aa9c3d83f88a4b9fc1ebc866c32fdf88f27a91"
PARAM_SHA = "afc5eb7348c895a196b685c860392dfecb8b0273b6bf7f204612fe0c3ae9e100"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptx", type=Path, required=True)
    parser.add_argument("--parameter", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ptx_bytes = args.ptx.read_bytes()
    require(sha(ptx_bytes) == PTX_SHA, "module23 PTX identity drift")
    text = ptx_bytes.decode("utf-8-sig").replace("\r\n", "\n")
    first = text.split("Fatbin ptx code:", 2)[1]
    require(first.count(".visible .entry main_kernel(") == 1,
            "module23 first-entry extraction drift")
    require(first.count("ld.param.u64 %rd7, [main_kernel_param_0];") == 1,
            "module23 offset-0 parameter load drift")
    require(len(re.findall(r"%rd7(?![0-9])", first)) == 2,
            "module23 offset-0 value has unexpected uses")
    require(first.count("cvta.to.global.u64 %rd64, %rd7;") == 1,
            "module23 offset-0 dereference root drift")
    require(first.count("ld.param.u32 %r56, [main_kernel_param_0+48];") == 1,
            "module23 mode load drift")
    control = """$L__BB0_64:
bar.sync 0;
setp.eq.s32 %p115, %r56, 0;
@%p115 bra $L__BB0_68;"""
    require(control in first, "module23 mode branch dominance drift")
    start = first.index(control)
    dead_block = first.index("$L__BB0_68:", start)
    dereference = first.index("cvta.to.global.u64 %rd64, %rd7;", dead_block)
    require("bra.uni $L__BB0_70;" in first[start:dead_block],
            "module23 live branch does not bypass dead block")
    require(dereference > dead_block, "module23 dereference placement drift")

    parameters = []
    for path in args.parameter:
        data = path.read_bytes()
        require(len(data) == 72, f"parameter byte count drift: {path}")
        require(sha(data) == PARAM_SHA, f"parameter identity drift: {path}")
        require(struct.unpack_from("<Q", data, 0)[0] == 0,
                f"offset-0 value is not null: {path}")
        require(struct.unpack_from("<I", data, 48)[0] == 1,
                f"selected mode is not 1: {path}")
        parameters.append({
            "source_name": path.parent.parent.name + "/" + path.name,
            "bytes": len(data),
            "raw_sha256": PARAM_SHA,
            "offset_0_value": "null",
            "mode_offset_48": 1,
        })
    require(len(parameters) >= 2, "fresh-process parameter evidence missing")
    result = {
        "schema": "neuralstudio.fg.module23-dead-offset0.v1",
        "module_token": 23,
        "module_sha256": "d787cf7b861295d1b4c6a80cb374230de793aa69f288a7e724bf59278bc4a62d",
        "ptx_sha256": PTX_SHA,
        "parameter_evidence": parameters,
        "selected_launches": [7, 17],
        "packed_parameter_bytes": 72,
        "proof": {
            "offset_0_load_count": 1,
            "offset_0_total_ssa_use_count": 2,
            "offset_0_only_dereference_block": "$L__BB0_68",
            "dereference_block_condition": "mode_at_offset_48 == 0",
            "selected_mode": 1,
            "selected_live_branch_bypasses_dereference": True,
            "selected_offset_0_value": "null",
        },
        "claims": {
            "offset_0_allocation_provenance_required_for_selected_mode": False,
            "offset_0_dead_for_selected_mode": True,
            "module23_numerical_parity": False,
            "connected_early_path": False,
        },
        "address_fields_recorded": False,
        "result": "pass",
    }
    result["receipt_root_sha256"] = sha(json.dumps(
        result, sort_keys=True, separators=(",", ":")).encode())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(f"receipt_root_sha256={result['receipt_root_sha256']}")
    print("result=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
