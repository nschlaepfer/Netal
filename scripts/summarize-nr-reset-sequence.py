#!/usr/bin/env python3
"""Seal four fresh two-evaluation RTX observations without broadening their scope."""
import argparse
import hashlib
import json
from pathlib import Path
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--captures',type=Path,nargs=4,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    rows=[];pids=set();sequence=None;outputs=None;source=None
    for d,poison in zip(a.captures,('a5','a5','5a','5a')):
        r=json.loads((d/'verification.json').read_text());b=dict(r);h=b.pop('receipt_root_sha256');assert canonical(b)==h and r['result']=='audited_observation'
        assert r['pid'] not in pids and r['poison']==poison;pids.add(r['pid'])
        assert r['resets'] in ([1,1],[1,0]) and (sequence is None or sequence==r['resets']);sequence=r['resets']
        assert outputs is None or outputs==r['outputs'];outputs=r['outputs']
        assert source is None or source==r['source_root'];source=r['source_root']
        assert r['input_arrays_rehashed_locally'] and r['graph_equal'] and r['address_free_graph_roots']==['0e7a4f6ba72bde69ec42eea4968ebee4d359f551d2961dd03945f7a7c25a2d61']*2
        assert sha(d/'capture-receipt.json')==r['capture_receipt_sha256'] and sha(d/'job-receipt.json')==r['job_receipt_sha256']
        for name,expected in zip(('output.rgba16le','frame-1-output.rgba16le'),outputs):assert (d/name).stat().st_size==16588800 and sha(d/name)==expected
        rows.append({'directory':str(d.resolve()),'pid':r['pid'],'poison':poison,'observation_root':h,'same_process_argument_hash_differences':r['same_process_argument_hash_differences']})
    r={'schema':'neuralstudio.nr.reset-sequence-matrix.v1','result':'pass_observed_repeatability','resets':sequence,'source_root':source,'runs':rows,'fresh_processes':4,'evaluations':8,'outputs':outputs,'output_poison_invariance':True,'second_equals_first':outputs[0]==outputs[1],'mac_temporal_parity':False,'full_goal_complete':False,'scope':'One fixed original image, two calls per instance, four fresh RTX processes and both output poisons. Unmodified selected graph/weights; locally rehashed inputs before/after. Not changing-image history, disocclusion, descriptor or arbitrary-length temporal validation.'}
    r['receipt_root_sha256']=canonical(r);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
