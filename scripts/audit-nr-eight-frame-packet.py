#!/usr/bin/env python3
"""Bind the new fourth-frame reference and reuse already verified comparison files."""
import argparse
import hashlib
import json
from pathlib import Path
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def require(ok,label):
    if not ok:raise ValueError(label)
def sealed(p):
    r=read(p);b=dict(r);h=b.pop('receipt_root_sha256');require(canonical(b)==h,'seal');return r
def audit(packet):
    repo=Path(__file__).resolve().parents[1];matrix=sealed(repo/'build/nr_eight_frame/matrix-audit-v1.json');manifest=sealed(packet/'transport-manifest.json');transport=sealed(packet/'retrieval.json');d=packet/'capture'
    require(matrix['receipt_root_sha256']=='f2fe8f4e02f4c40972b41643d482b37ef8e4ab87c8301593b6f9d9e8bbdd84c2','eight-frame matrix')
    require(transport['manifest_root']==manifest['receipt_root_sha256'] and transport['result']=='pass_transport_only' and not transport['inference_acceptance'],'transport only')
    for n,m in manifest['files'].items():require((d/n).stat().st_size==m['bytes'] and sha(d/n)==m['sha256'],'transferred payload')
    reference=sealed(d/'reference.json');require(reference['receipt_root_sha256']==matrix['runs'][0]['reference_root'] and sha(d/'capture-receipt.json')==matrix['runs'][0]['capture_sha256'],'new reference binds to executed capture')
    paths=[repo.parent/'work/rtx-oracle-fp8/capture/output.rgba16le',repo/'build/nr_temporal_post_input/packet-v1/capture/expected-output.rgba16le',repo/'build/nr_three_frame_history/receipts-v1/temporal-reset100-a5-a/frame-2-output.rgba16le',d/'frame-3-output.rgba16le']
    outputs=[]
    for i in range(8):
        path=paths[i%4];require(path.stat().st_size==16588800 and sha(path)==matrix['outputs'][i],'all eight reference identities')
        outputs.append({'frame':i,'file':str(path.resolve()),'sha256':sha(path),'role':'CPU comparison only, never runtime/history input'})
    return {'schema':'neuralstudio.nr.eight-frame-packet.v1','result':'pass_verified_references','matrix_root':matrix['receipt_root_sha256'],'transport_root':manifest['receipt_root_sha256'],'reset_mask':matrix['reset_mask'],'outputs':outputs,'only_new_output_downloaded':'frame-3-output.rgba16le; frame7 has the same current RTX hash','model_executed':False,'mac_parity':False,'full_goal_complete':False,'verifier_sha256':sha(Path(__file__))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'preserve prior packet audit')
    r=audit(a.packet.resolve());r['receipt_root_sha256']=canonical(r);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='outputs'},indent=2))
if __name__=='__main__':main()
