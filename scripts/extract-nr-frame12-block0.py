#!/usr/bin/env python3
"""Extract a comparison-only Block0 encoder tensor from existing RTX snapshots."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--job-root',type=Path,required=True);a=p.parse_args();job=a.job_root.resolve()
    assert job.drive.upper()=='D:' and job.parent.name=='CodexLab' and job.name=='nr-color-reset-sixteen-20260906-v1'
    label='continuous-a5-a';capture=job/label;out=job/'frame12-block0-v2';assert not out.exists();out.mkdir()
    ref=read(job/(label+'-reference.json'));cap=read(capture/'capture-receipt.json');inv=read(capture/'artifact-inventory.json');files={f['path']:f for f in inv['files']}
    assert sha(capture/'capture-receipt.json')==ref['capture_receipt_sha256'] and sha(capture/'artifact-inventory.json')==ref['inventory_sha256']==cap['artifact_inventory_sha256']
    parser=job/'build-rtx-oracle-manifest.py';assert sha(parser)==ref['parser_sha256'];s=importlib.util.spec_from_file_location('delta',parser);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    state=None;boundaries=[]
    for launch in cap['snapshot_launch_ids']:
        if launch>1874:break
        n=f'pool-snapshots/snapshot-{launch:04d}.delta';path=capture/n;assert sha(path)==files[n]['sha256']
        state,info=m.parse_snapshot_delta(path,launch,state);boundaries.append(info)
    assert boundaries[-1]['launch_id']=='launch_1874' and len(boundaries)==25
    data=bytes(memoryview(state)[71166976:71166976+17694720]);h=hashlib.sha256(data).hexdigest();assert h==ref['boundaries'][24]['ports']['block0-encoder']
    target=out/'rtx-block0-encoder.e4m3';target.write_bytes(data)
    r={'schema':'neuralstudio.nr.frame12-block0-diagnostic.v1','capture_reference_root':ref['receipt_root_sha256'],'frame':12,'launch':1874,'tensor_bytes':len(data),'tensor_sha256':h,'source_snapshots':len(boundaries),'model_executed':False,'role':'archived comparison-only component tensor; NEVER feed into live full-model inference','extractor_sha256':sha(Path(__file__))};r['receipt_root_sha256']=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest();receipt=out/'extraction.json';receipt.write_text(json.dumps(r,indent=2)+'\n')
    ps=importlib.util.spec_from_file_location('pack',job/'pack-nr-reset-retrieval.py');pm=importlib.util.module_from_spec(ps);ps.loader.exec_module(pm)
    packed=pm.pack({'rtx-block0-encoder.e4m3':target,'extraction.json':receipt},job/'transport-v1','frame12-block0-v2')
    print(json.dumps({'extraction_root':r['receipt_root_sha256'],'tensor_sha256':h,'transport_root':packed['receipt_root_sha256'],'archive_bytes':packed['archive']['bytes'],'parts':len(packed['parts'])}))
if __name__=='__main__':main()
