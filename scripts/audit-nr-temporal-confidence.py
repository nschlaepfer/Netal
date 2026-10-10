#!/usr/bin/env python3
"""Resolve raw post155 +104 using this capture's allocations, not replay IDs."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();repo=Path(__file__).resolve().parents[1]
    capdir=repo/'build/nr_reset_sequence/pilot-reset10-compact-v1';argsdir=repo/'build/nr_reset_sequence/pilot-reset10-args-v1';cap=read(capdir/'capture-receipt.json');delta=read(argsdir/'comparison.json');body=dict(delta);root=body.pop('receipt_root_sha256');assert canonical(body)==root
    assert sha(capdir/'capture-receipt.json')==delta['capture_receipt_sha256']=='8476f0f2d226dfd5245324529196b873eeec3848efa242e92af55042008562b9'
    assert sha(capdir/'rtx-cuda-trace.jsonl')==cap['cuda_trace_sha256'] and sha(capdir/'artifact-inventory.json')==cap['artifact_inventory_sha256']
    for n,h in delta['files'].items():assert sha(argsdir/n)==h
    b0=(argsdir/'arguments-0155.bin').read_bytes();b1=(argsdir/'arguments-0311.bin').read_bytes();pointer=struct.unpack_from('<Q',b0,104)[0];assert struct.unpack_from('<Q',b1,104)[0]==pointer
    events=[json.loads(s) for s in (capdir/'rtx-cuda-trace.jsonl').read_text(encoding='utf-8-sig').splitlines()];alloc=[e for e in events if e.get('call')=='cuMemAlloc' and int(e['pointer'],16)<=pointer<int(e['pointer'],16)+e['bytes']];assert len(alloc)==1
    allocation=alloc[0];offset=pointer-int(allocation['pointer'],16);transfer=[e for e in events if e.get('call')=='cuMemcpyHtoDAsync' and e['allocation_id']==allocation['allocation_id'] and e['byte_offset']<=offset<e['byte_offset']+e['bytes']];assert len(transfer)==1;t=transfer[0]
    name=f'cuda-dumps/transfer-{t["transfer_id"]:04d}.bin';files={x['path']:x for x in read(capdir/'artifact-inventory.json')['files']};local=repo.parent/'work/rtx-oracle-fp8/capture'/name
    assert sha(local)==files[name]['sha256'] and local.stat().st_size==files[name]['bytes']==t['bytes']==2
    data=local.read_bytes();assert offset==t['byte_offset'] and data.hex()=='eb39';half=struct.unpack('<e',data)[0]
    textures={e['object']:e['array_id'] for e in events if e.get('call')=='cuTexObjectCreate'};store=struct.unpack_from('<Q',b0,16)[0];assert store==4 and textures[store]==3 and textures[1]==1 and textures[6]==5
    unchanged={str(off):{'u32':struct.unpack_from('<I',b0,off)[0],'hex_le':b0[off:off+4].hex()} for off in (52,112,164,168)}
    assert all(b0[off:off+4]==b1[off:off+4] for off in (52,112,164,168)) and unchanged['112']['u32']==1
    r={'schema':'neuralstudio.nr.temporal-confidence-binding.v1','result':'pass_capture_binding','capture_receipt_sha256':delta['capture_receipt_sha256'],'argument_delta_root':root,'post155_argument_offset':104,'cuda_allocation_id':allocation['allocation_id'],'allocation_offset':offset,'transfer_id':t['transfer_id'],'record_bytes':2,'record_sha256':sha(local),'half_bits':'0x39eb','half_value':half,'unchanged_between_evaluations':True,'post155_store_handle':store,'post155_store_array_id':3,'history_read_array_id':1,'motion_array_id':5,'interpretation':'This capture binds a two-byte uploaded constant, not the one-byte CUDA dummy allocation. Replay-local pointer IDs must not be interpreted as CUDA allocation IDs. Caller output array3 and internal history array1 are distinct resources. Kernel156 handles/history content still need same-capture verification.','gpu_executed':False,'verifier_sha256':sha(Path(__file__))}
    r['post155_unchanged_field_words']=unchanged
    r['template_caution']='The live first/second records both have +112 u32=1; +164/+168 already contain inverse dimensions. Do not populate only the changed offsets on an older replay template with different inactive-branch fields.'
    r['receipt_root_sha256']=canonical(r);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
