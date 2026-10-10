#!/usr/bin/env python3
"""Per-module and aggregate verification for selected FG 3x3 convolutions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MANIFEST_SHA256 = "2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294"
RUN_SCHEMA = "neuralstudio.fg.conv-3x3-run.v1"
BACKEND_SCHEMA = "neuralstudio.fg.conv-3x3-backend.v1"
FINAL_SCHEMA = "neuralstudio.fg.conv-3x3-verification.v1"
FAMILY_SCHEMA = "neuralstudio.fg.conv-3x3-family.v1"
GENERATOR = "splitmix64-fg-conv-3x3-family-v1"
SPATIAL_ORDER = [0, 3, 6, 1, 4, 7, 2, 5, 8]


@dataclass(frozen=True)
class Layer:
    id: str; launch: int; module_token: int; module_bytes: int
    module_sha256: str; weight_token: int; weight_bytes: int
    weight_sha256: str; width: int; height: int; input_channels: int
    output_channels: int; grid: tuple[int, int, int]; block: tuple[int, int, int]
    column_stage_schedule: bool


LAYERS = {
    item.id: item
    for item in (
        Layer("l20",20,28,16512,"7f9c1f110e8ec9347aa968bc9019e5ffc8839057987a2e5676a0f64637c8df2d",63,56320,"44595b02307a6b934a124abe476cfd327440e2acba7f5ed69f73195ca743e3d9",160,96,48,32,(10,8,1),(32,4,1),False),
        Layer("l24",24,31,11168,"e925f51165408fae4657d50247b10304802d46979b863712aeff844a217ade63",67,299008,"4191924ff335d5018987f2526f933cb4cdbb9145411e3ce34f63c8206321be55",40,24,64,128,(3,24,1),(32,1,4),False),
        Layer("l26",26,32,11416,"c74961f6f71f4e7bac5b216cc704b10a0fdd6f0987352fdd315f371467ca7187",69,593920,"c0c5648f9cb839e41d0e786c8034aa0bcc931b6ac84c58cfbde0502dc8a53404",20,12,128,128,(2,6,1),(32,2,4),False),
        Layer("l28",28,33,14752,"6f2ffbe7faacde8119719d92dacde40f21ddd562878381b81654dad7a66775dd",71,1187840,"fe2c3057f2e894219601d3d7200eabfa97d990e08ccb23b43d52dfcba2d2fb93",10,6,128,256,(1,6,2),(32,1,8),True),
    )
}


class VerificationError(RuntimeError): pass
def require(condition: bool, message: str) -> None:
    if not condition: raise VerificationError(message)
def sha256_bytes(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def canonical_sha256(value: Any) -> str: return sha256_bytes(json.dumps(value,sort_keys=True,separators=(",",":")).encode())


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig")); require(isinstance(value,dict),f"not object: {path}"); return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_suffix(path.suffix+".tmp"); temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n"); os.replace(temporary,path)


def require_address_free(value: Any, label: str) -> None:
    def visit(node: Any, location: str) -> None:
        if isinstance(node,dict):
            for key,child in node.items():
                if key != "address_fields_recorded": require(not any(part in key.lower() for part in ("pointer","virtual_address","device_address","gpu_address")),f"{label}: address key {location}.{key}")
                visit(child,f"{location}.{key}")
        elif isinstance(node,list):
            for index,child in enumerate(node): visit(child,f"{location}[{index}]")
        elif isinstance(node,str): require(re.search(r"0x[0-9a-fA-F]{12,16}(?:\b|$)",node) is None,f"{label}: address text {location}")
    visit(value,label)


def splitmix64(value: int) -> int:
    mask=(1<<64)-1; value=(value+0x9E3779B97F4A7C15)&mask; value=((value^(value>>30))*0xBF58476D1CE4E5B9)&mask; value=((value^(value>>27))*0x94D049BB133111EB)&mask; return (value^(value>>31))&mask


def generated_input(config: Layer) -> bytes:
    values=(0x0000,0x2800,0xA800,0x2C00,0xAC00,0x3000,0xB000,0x3400,0xB400,0x3800,0xB800)
    elements=config.width*config.height*config.input_channels; result=bytearray(elements*2)
    seed=0xD6E8FEB86659FD93^(config.launch*0xA5A3564E27F8862B); step=0x9E3779B97F4A7C15; mask=(1<<64)-1
    for index in range(elements): struct.pack_into("<H",result,index*2,values[splitmix64((seed+index*step)&mask)%len(values)])
    return bytes(result)


def validate_backend(backend: str, config: Layer, root: Path) -> dict[str, Any]:
    expected_backend="rtx-cuda-driver" if backend=="rtx" else "mac-metal"; input_sha=sha256_bytes(generated_input(config)); output_bytes=config.width*config.height*config.output_channels*2; receipts=[]; outputs=[]
    for poison in (0xA5,0x5A):
        for ordinal in ("a","b"):
            run_id=f"{config.id}-poison-{poison:02x}-process-{ordinal}"; run_root=root/run_id; receipt=load_json(run_root/"run-receipt.json"); output=(run_root/"output.bin").read_bytes()
            require(receipt.get("schema")==RUN_SCHEMA and receipt.get("backend")==expected_backend,f"{run_id}: identity")
            require(receipt.get("layer_id")==config.id and receipt.get("run_id")==run_id and receipt.get("poison_byte")==poison,f"{run_id}: run")
            require(isinstance(receipt.get("process_id"),int),f"{run_id}: pid")
            require(receipt.get("module_token")==config.module_token and receipt.get("module_bytes")==config.module_bytes and receipt.get("module_sha256")==config.module_sha256,f"{run_id}: module")
            require(receipt.get("trace_launch_token")==f"launch-{config.launch:06d}" and receipt.get("packed_parameter_bytes")==48,f"{run_id}: launch")
            require(receipt.get("abi_scalars")==[config.width,config.height,config.width+2,config.height+2,config.width,config.height],f"{run_id}: ABI")
            require(receipt.get("input_shape_nhwc")==[1,config.height,config.width,config.input_channels] and receipt.get("output_shape_nhwc")==[1,config.height,config.width,config.output_channels],f"{run_id}: shape")
            expected_schedule = "spatial-column-outer/channel-stage-inner/vertical-triplet" if config.column_stage_schedule else "channel-stage-outer/column-major-spatial"
            require(receipt.get("spatial_order")==SPATIAL_ORDER and receipt.get("accumulation_schedule")==expected_schedule and receipt.get("leaky_relu") is True,f"{run_id}: order/activation")
            require(receipt.get("weight_source_token")==config.weight_token and receipt.get("weight_capture_manifest_sha256")==MANIFEST_SHA256,f"{run_id}: weight provenance")
            require(receipt.get("weight_pre_sha256")==config.weight_sha256==receipt.get("weight_post_sha256"),f"{run_id}: weights")
            require(receipt.get("input_generator")==GENERATOR and receipt.get("input_pre_sha256")==input_sha==receipt.get("input_post_sha256"),f"{run_id}: input")
            require(len(output)==output_bytes and receipt.get("output_bytes")==output_bytes and receipt.get("output_sha256")==sha256_bytes(output),f"{run_id}: output")
            require(receipt.get("input_mismatched_bytes")==0 and receipt.get("canary_mismatched_bytes")==0,f"{run_id}: mutation/canary")
            require(receipt.get("per_run_write_coverage_conclusive") is False and receipt.get("external_weight_file_opened") is True and receipt.get("expected_output_file_opened") is False,f"{run_id}: provenance")
            require(receipt.get("address_fields_recorded") is False and receipt.get("result")=="pass",f"{run_id}: result")
            if backend=="rtx": require(receipt.get("grid")==list(config.grid) and receipt.get("block")==list(config.block) and receipt.get("function_name")=="k_conv_fp16_nhwc",f"{run_id}: RTX geometry")
            else:
                require(receipt.get("selected_rtx_grid")==list(config.grid) and receipt.get("selected_rtx_block")==list(config.block),f"{run_id}: selected geometry")
                require(receipt.get("metal_dispatch_threads")==[config.width,config.height,config.output_channels] and receipt.get("metal_threads_per_threadgroup")==[8,4,1] and receipt.get("metal_function")=="fg_conv_3x3",f"{run_id}: Metal geometry")
            require_address_free(receipt,run_id); receipts.append(receipt); outputs.append(output)
    require(len({r["process_id"] for r in receipts})==4,f"{config.id}: fresh process"); require(len({sha256_bytes(o) for o in outputs})==1,f"{config.id}: poison/process")
    mutated=bytearray(outputs[0]); mutated[config.launch%len(mutated)]^=1; require(sha256_bytes(mutated)!=receipts[0]["output_sha256"],f"{config.id}: mutation")
    result={"schema":BACKEND_SCHEMA,"backend":expected_backend,"layer_id":config.id,"trace_launch_token":f"launch-{config.launch:06d}","module_token":config.module_token,"module_sha256":config.module_sha256,"weight_source_token":config.weight_token,"weight_sha256":config.weight_sha256,"weight_capture_manifest_sha256":MANIFEST_SHA256,"input_generator":GENERATOR,"input_sha256":input_sha,"output_sha256":sha256_bytes(outputs[0]),"output_bytes":output_bytes,"fresh_process_count":4,"two_process_two_poison_invariant":True,"full_write_coverage_from_two_poison_invariance":True,"input_weight_immutability":True,"canary_integrity":True,"expected_output_fed_to_execution":False,"output_mutation_rejected":True,"spatial_order_validated":SPATIAL_ORDER,"accumulation_schedule_validated":("spatial-column-outer/channel-stage-inner/vertical-triplet" if config.column_stage_schedule else "channel-stage-outer/column-major-spatial"),"address_fields_recorded":False,"result":"pass"}
    result["receipt_root_sha256"]=canonical_sha256(result); return result


def validate_final(rtx_path: Path, mac_path: Path, config: Layer) -> dict[str, Any]:
    rtx=load_json(rtx_path); mac=load_json(mac_path)
    for receipt,backend in ((rtx,"rtx-cuda-driver"),(mac,"mac-metal")):
        require(receipt.get("schema")==BACKEND_SCHEMA and receipt.get("backend")==backend,f"{config.id}/{backend}: identity"); without=dict(receipt); root=without.pop("receipt_root_sha256",None); require(root==canonical_sha256(without),f"{config.id}/{backend}: root"); require_address_free(receipt,f"{config.id}/{backend}")
    require(rtx["layer_id"]==mac["layer_id"]==config.id and rtx["output_sha256"]==mac["output_sha256"],f"{config.id}: parity")
    result={"schema":FINAL_SCHEMA,"layer_id":config.id,"trace_launch_token":f"launch-{config.launch:06d}","module_sha256":config.module_sha256,"weight_sha256":config.weight_sha256,"input_sha256":rtx["input_sha256"],"output_sha256":rtx["output_sha256"],"output_bytes":rtx["output_bytes"],"rtx_backend_root_sha256":rtx["receipt_root_sha256"],"mac_backend_root_sha256":mac["receipt_root_sha256"],"rtx_mac_differing_bytes":0,"rtx_mac_exact":True,"scope":"one selected 3x3 convolution; no family or multi-kernel group claim","address_fields_recorded":False,"result":"pass"}; result["receipt_root_sha256"]=canonical_sha256(result); return result


def validate_family(receipt_dir: Path, layer_ids: list[str]) -> dict[str, Any]:
    require(len(layer_ids)==len(set(layer_ids)),"duplicate layer"); members=[]
    for layer_id in layer_ids:
        config=LAYERS[layer_id]; receipt=load_json(receipt_dir/f"{layer_id}-verification.json"); require(receipt.get("schema")==FINAL_SCHEMA and receipt.get("layer_id")==layer_id and receipt.get("rtx_mac_exact") is True,f"{layer_id}: final"); without=dict(receipt); root=without.pop("receipt_root_sha256",None); require(root==canonical_sha256(without) and receipt.get("module_sha256")==config.module_sha256,f"{layer_id}: root/module"); members.append({"layer_id":layer_id,"receipt_root_sha256":root,"output_sha256":receipt["output_sha256"]})
    result={"schema":FAMILY_SCHEMA,"layer_count":len(members),"layers":members,"every_layer_executed_independently":True,"every_layer_rtx_mac_exact":True,"family_inference_from_shape_only":False,"address_fields_recorded":False,"result":"pass"}; result["receipt_root_sha256"]=canonical_sha256(result); return result


def main() -> int:
    parser=argparse.ArgumentParser(); subs=parser.add_subparsers(dest="command",required=True)
    backend=subs.add_parser("backend"); backend.add_argument("--backend",choices=("rtx","mac"),required=True); backend.add_argument("--layer",choices=tuple(LAYERS),required=True); backend.add_argument("--root",type=Path,required=True); backend.add_argument("--output",type=Path,required=True)
    final=subs.add_parser("final"); final.add_argument("--layer",choices=tuple(LAYERS),required=True); final.add_argument("--rtx",type=Path,required=True); final.add_argument("--mac",type=Path,required=True); final.add_argument("--output",type=Path,required=True)
    family=subs.add_parser("family"); family.add_argument("--receipt-dir",type=Path,required=True); family.add_argument("--layers",nargs="+",choices=tuple(LAYERS),required=True); family.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    try:
        if args.command=="backend": result=validate_backend(args.backend,LAYERS[args.layer],args.root)
        elif args.command=="final": result=validate_final(args.rtx,args.mac,LAYERS[args.layer])
        else: result=validate_family(args.receipt_dir,args.layers)
        write_json(args.output,result); print(f"receipt_root_sha256={result['receipt_root_sha256']}"); print("result=pass"); return 0
    except (OSError,json.JSONDecodeError,KeyError,VerificationError) as error: print(f"verification_error={error}",file=sys.stderr); return 1


if __name__=="__main__": raise SystemExit(main())
