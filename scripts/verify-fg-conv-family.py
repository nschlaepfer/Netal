#!/usr/bin/env python3
"""Aggregate the independently verified selected FG convolution families."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


ONE_BY_ONE_IDS = {"l22", "l30", "l31", "l34", "l35", "l38", "l39", "l42", "l43", "l46", "l47", "l50", "l51"}
THREE_BY_THREE_IDS = {"l20", "l24", "l26", "l28"}


class VerificationError(RuntimeError): pass
def require(condition: bool, message: str) -> None:
    if not condition: raise VerificationError(message)
def canonical_sha256(value: Any) -> str: return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value=json.loads(path.read_text(encoding="utf-8-sig")); require(isinstance(value,dict),f"not object: {path}"); return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_suffix(path.suffix+".tmp"); temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n"); os.replace(temporary,path)


def validate_family_receipt(receipt: dict[str, Any], schema: str, ids: set[str], label: str) -> None:
    require(receipt.get("schema")==schema,f"{label}: schema"); require(receipt.get("layer_count")==len(ids),f"{label}: count"); require(receipt.get("every_layer_executed_independently") is True and receipt.get("every_layer_rtx_mac_exact") is True,f"{label}: parity"); require(receipt.get("family_inference_from_shape_only") is False,f"{label}: inference"); require(receipt.get("address_fields_recorded") is False and receipt.get("result")=="pass",f"{label}: result")
    without=dict(receipt); root=without.pop("receipt_root_sha256",None); require(root==canonical_sha256(without),f"{label}: root"); members=receipt.get("layers"); require(isinstance(members,list),f"{label}: members"); require({member.get("layer_id") for member in members}==ids,f"{label}: ids"); require(all(isinstance(member.get("receipt_root_sha256"),str) and len(member["receipt_root_sha256"])==64 for member in members),f"{label}: member roots")


def validate(one_path: Path, three_path: Path) -> dict[str, Any]:
    one=load_json(one_path); three=load_json(three_path); validate_family_receipt(one,"neuralstudio.fg.conv-1x1-family.v1",ONE_BY_ONE_IDS,"1x1"); validate_family_receipt(three,"neuralstudio.fg.conv-3x3-family.v1",THREE_BY_THREE_IDS,"3x3")
    members=one["layers"]+three["layers"]; members=sorted(members,key=lambda item:int(item["layer_id"][1:])); ids=[member["layer_id"] for member in members]; require(len(ids)==17 and len(set(ids))==17,"aggregate launch coverage")
    result={"schema":"neuralstudio.fg.conv-family.v1","selected_convolution_launch_count":17,"selected_convolution_launches":ids,"one_by_one_family_root_sha256":one["receipt_root_sha256"],"three_by_three_family_root_sha256":three["receipt_root_sha256"],"per_launch_receipts":[{"layer_id":member["layer_id"],"receipt_root_sha256":member["receipt_root_sha256"],"output_sha256":member["output_sha256"]} for member in members],"every_selected_convolution_executed_independently":True,"every_selected_convolution_rtx_mac_exact":True,"connected_multi_kernel_group_executed":False,"full_frame_parity_claimed":False,"address_fields_recorded":False,"result":"pass"}; result["receipt_root_sha256"]=canonical_sha256(result); return result


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--one-by-one",type=Path,required=True); parser.add_argument("--three-by-three",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
    try: result=validate(args.one_by_one,args.three_by_three); write_json(args.output,result); print(f"receipt_root_sha256={result['receipt_root_sha256']}"); print("result=pass"); return 0
    except (OSError,json.JSONDecodeError,VerificationError) as error: print(f"verification_error={error}",file=sys.stderr); return 1


if __name__=="__main__": raise SystemExit(main())
