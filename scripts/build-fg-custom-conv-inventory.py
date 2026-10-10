#!/usr/bin/env python3
"""Build an address-free inventory of the 25 selected custom convolutions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


class InventoryError(RuntimeError): pass
def require(condition: bool, message: str) -> None:
    if not condition: raise InventoryError(message)
def sha256_file(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()
def canonical_sha256(value: Any) -> str: return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def load_json(path: Path) -> dict[str, Any]:
    value=json.loads(path.read_text(encoding="utf-8-sig")); require(isinstance(value,dict),f"not object: {path}"); return value
def write_json(path: Path,value:dict[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+".tmp");temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n");os.replace(temporary,path)


def build(trace_path: Path, dump_root: Path, provenance_path: Path, manifest_path: Path) -> dict[str, Any]:
    trace_lines=[json.loads(line) for line in trace_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    module_creates={event["module_token"]:event for event in trace_lines if event.get("call")=="NvAPI_D3D12_CreateCuModule" and event.get("status")==0}
    launches=[event for event in trace_lines if event.get("event")=="kernel_launch" and str(event.get("function_name","")).startswith("custom_block")]
    require(len(launches)==25,f"custom launch count {len(launches)} != 25")
    provenance=load_json(provenance_path); comparisons=provenance["canonical_parameters"]["comparisons"]
    by_launch={comparison["launch_ordinal"]+3:comparison for comparison in comparisons if str(comparison.get("function_name","")).startswith("custom_block")}
    require(len(by_launch)==25,"provenance custom launch count mismatch")
    manifest=load_json(manifest_path); manifest_entries={entry["source_token"]:entry for entry in manifest["entries"]}
    rows=[]
    for launch in launches:
        launch_number=int(launch["launch_token"].split("-")[-1]); comparison=by_launch.get(launch_number); require(comparison is not None,f"missing provenance L{launch_number}")
        require(comparison["function_name"]==launch["function_name"],f"function mismatch L{launch_number}"); require(comparison["module_token"]==launch["module_token"],f"module mismatch L{launch_number}"); require(comparison["canonical_equal"] is True,f"nonrepeat parameters L{launch_number}")
        substitutions={entry["parameter_offset"]:entry for entry in comparison["allocation_pointer_substitutions_a"]}; require(0 in substitutions and 8 in substitutions,f"missing model pair L{launch_number}")
        weight_token=int(substitutions[0]["allocation_token"].split("-")[-1]); bias_token=int(substitutions[8]["allocation_token"].split("-")[-1]); weight=manifest_entries.get(weight_token); bias=manifest_entries.get(bias_token); require(weight is not None and bias is not None,f"missing model dump L{launch_number}")
        module=module_creates[launch["module_token"]]; module_path=dump_root/module["module_blob_file"]; require(module_path.is_file(),f"missing module dump L{launch_number}"); require(sha256_file(module_path)==module["module_blob_sha256"].lower(),f"module rehash L{launch_number}")
        other=[]
        for offset,entry in sorted(substitutions.items()):
            if offset in (0,8): continue
            other.append({"parameter_offset":offset,"allocation_token":int(entry["allocation_token"].split("-")[-1]),"allocation_offset":entry["allocation_offset"]})
        rows.append({"launch":launch_number,"launch_token":launch["launch_token"],"function_name":launch["function_name"],"module_token":int(launch["module_token"].split("-")[-1]),"module_bytes":module["module_blob_bytes"],"module_sha256":module["module_blob_sha256"].lower(),"grid":launch["grid"],"block":launch["block"],"dynamic_shared_bytes":launch["dynamic_shared_bytes"],"parameter_bytes":launch["parameter_bytes"],"parameter_sha256":launch["parameter_sha256"].lower(),"weight_token":weight_token,"weight_bytes":weight["logical_bytes"],"weight_sha256":weight["sha256"],"bias_token":bias_token,"bias_bytes":bias["logical_bytes"],"bias_sha256":bias["sha256"],"other_buffer_bindings":other})
    require([row["launch"] for row in rows]==[57,58,59,*range(60,70),*range(72,84)],"custom launch order mismatch")
    require(len({row["weight_token"] for row in rows})==25 and len({row["bias_token"] for row in rows})==25,"model pair tokens not unique")
    result={"schema":"neuralstudio.fg.custom-conv-inventory.v1","source":{"trace_sha256":sha256_file(trace_path),"allocation_provenance_receipt_root_sha256":provenance["receipt_root_sha256"],"default_manifest_sha256":sha256_file(manifest_path)},"custom_convolution_launch_count":25,"model_weight_bias_pair_count":25,"launches":rows,"all_modules_rehashed":True,"all_parameter_blocks_repeat_canonical":True,"all_model_pairs_post_evaluate_0":True,"addresses_recorded":False,"parity_claimed":False}
    result["receipt_root_sha256"]=canonical_sha256(result);return result


def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--trace",type=Path,required=True);parser.add_argument("--dump-root",type=Path,required=True);parser.add_argument("--allocation-provenance",type=Path,required=True);parser.add_argument("--default-manifest",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    try: result=build(args.trace,args.dump_root,args.allocation_provenance,args.default_manifest);write_json(args.output,result);print(f"receipt_root_sha256={result['receipt_root_sha256']}");print("result=pass");return 0
    except (OSError,json.JSONDecodeError,InventoryError,KeyError) as error:print(f"inventory_error={error}",file=sys.stderr);return 1
if __name__=="__main__":raise SystemExit(main())
