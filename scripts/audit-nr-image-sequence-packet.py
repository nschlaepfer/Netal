#!/usr/bin/env python3
"""Bind downloaded original input and comparison-only finals to repeated RTX runs."""
import argparse
import hashlib
import json
from pathlib import Path
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def require(v,label):
    if not v:raise ValueError(label)
def sealed(p):
    r=read(p);b=dict(r);h=b.pop('receipt_root_sha256');require(canonical(b)==h,'seal '+str(p));return r
def audit(packet):
    repo=Path(__file__).resolve().parents[1];matrix=sealed(repo/'build/nr_image_sequence/matrix-audit-v1.json')
    require(matrix['color_cases']==[0,2,1] and matrix['result']=='pass_bounded_RTX_reference' and matrix['fresh_processes']==4,'accepted per-case matrix')
    transport=sealed(packet/'transport-manifest.json');retrieval=sealed(packet/'retrieval.json');d=packet/'capture'
    require(retrieval['manifest_root']==transport['receipt_root_sha256'] and retrieval['result']=='pass_transport_only' and not retrieval['inference_acceptance'],'verified transport only')
    require(transport['files']==retrieval['files'],'transport inventory')
    for n,meta in transport['files'].items():require((d/n).stat().st_size==meta['bytes'] and sha(d/n)==meta['sha256'],'downloaded payload '+n)
    ref=sealed(d/'reference.json');cap=read(d/'capture-receipt.json');inventory=read(d/'artifact-inventory.json');files={x['path']:x for x in inventory['files']}
    require(ref['receipt_root_sha256']==matrix['runs'][0]['reference_root'] and ref['color_cases']==[0,2,1] and ref['image_sequence']=='mixed','matrix reference binding')
    require(sha(d/'capture-receipt.json')==ref['capture_receipt_sha256']==matrix['runs'][0]['native_capture_sha256'],'native capture binding')
    require(sha(d/'artifact-inventory.json')==ref['inventory_sha256']==cap['artifact_inventory_sha256'],'execution-time inventory')
    original_inputs=[]
    for i,n in enumerate(('input-color.rgba16le','frame-1-input-color.rgba16le','frame-2-input-color.rgba16le')):
        p=d/n;require(p.stat().st_size==16588800 and sha(p)==matrix['input_hashes'][i]==files[n]['sha256'],'per-frame original input')
        original_inputs.append({'frame':i,'case':matrix['color_cases'][i],'file':str(p.resolve()),'sha256':sha(p),'role':'original synthetic input; allowed runtime input'})
    motion=d/'input-motion.rg16le';require(motion.stat().st_size==8294400 and sha(motion)=='788ae0147bdf979a6575938ca2d7d4403788588f7be2010f03776c968fd1ab49','zero motion')
    outputs=[]
    for i,n in enumerate(('output.rgba16le','frame-1-output.rgba16le','frame-2-output.rgba16le')):
        p=d/n;require(p.stat().st_size==16588800 and sha(p)==matrix['outputs'][i]==files[n]['sha256'],'actual reference output '+n)
        outputs.append({'frame':i,'file':str(p.resolve()),'sha256':sha(p),'role':'CPU comparison only; never feed to live runtime/history'})
    return {'schema':'neuralstudio.nr.image-sequence-packet-audit.v1','result':'pass_verified_reference_packet','color_cases':[0,2,1],'matrix_root':matrix['receipt_root_sha256'],'transport_root':transport['receipt_root_sha256'],'original_inputs':original_inputs,'original_motion':{'file':str(motion.resolve()),'sha256':sha(motion)},'comparison_only_outputs':outputs,'boundaries_comparison_only':matrix['boundary_ports'],'model_executed':False,'mac_parity':False,'full_goal_complete':False,'verifier_sha256':sha(Path(__file__))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'preserve prior audit')
    r=audit(a.packet.resolve());r['receipt_root_sha256']=canonical(r);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='boundaries_comparison_only'},indent=2))
if __name__=='__main__':main()
