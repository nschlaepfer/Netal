#!/usr/bin/env python3
"""Verify the exact diagnostic post-stage packet; never promote it as live input."""
import argparse
import hashlib
import json
from pathlib import Path
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def audit(d):
    r=read(d/'reference.json');body=dict(r);root=body.pop('receipt_root_sha256');assert canonical(body)==root=='c0b0b946c44bf01dadba4b1a7b29a4cd1976355ec91a674337438db5fbae652e'
    assert r['result']=='pass_comparison_reference' and r['live_runtime_input_forbidden'] is True and r['gpu_executed_by_extractor'] is False
    assert r['parent_reference_root']=='3d1aa05303da317a88ae46353743a6ade67017f4b9ad97d31398283d413ef6f8'
    cap=read(d/'capture-receipt.json');build=read(d/'build-receipt.json');job=read(d/'job-receipt.json');inv=read(d/'artifact-inventory.json');files={x['path']:x for x in inv['files']}
    assert sha(d/'capture-receipt.json')==r['capture_receipt_sha256']==job['capture_receipt_sha256']
    assert sha(d/'artifact-inventory.json')==r['inventory_sha256']==cap['artifact_inventory_sha256'] and sha(d/'build-receipt.json')==job['build_receipt_sha256']
    assert build['source_root']==r['source_root']=='1dfae0912b42f835caa09c86aea5a65a917087006e88d3b40bb78e3a02c3e6fa' and build['exit_code']==job['exit_code']==0
    assert cap['runtime_file_sha256']=='e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e' and cap['analysis_runtime'] is None
    sizes={'post-input-pool.bin':212519936,'params-311.bin':184,'params-312.bin':72,'input-color.rgba16le':16588800,'motion.rg16le':8294400,'history.rgba16le':16588800,'expected-output.rgba16le':16588800}
    assert set(r['files'])==set(sizes)
    for n,size in sizes.items():assert (d/n).stat().st_size==r['files'][n]['bytes']==size and sha(d/n)==r['files'][n]['sha256'],n
    for local,source in {'params-311.bin':'cuda-dumps/arguments-0311.bin','params-312.bin':'cuda-dumps/arguments-0312.bin','input-color.rgba16le':'input-color.rgba16le','motion.rg16le':'input-motion.rg16le','history.rgba16le':'frame-1-history-before.rgba16le','expected-output.rgba16le':'frame-1-output.rgba16le'}.items():assert sha(d/local)==files[source]['sha256']
    assert cap['snapshot_launch_ids']==[2,24,158,180,310] and cap['counts']['snapshots']==5 and cap['counts']['launches']==312
    for n,info in zip((2,24,158,180,310),r['snapshots']):assert info['delta_sha256']==files[f'pool-snapshots/snapshot-{n:04d}.delta']['sha256']
    assert sha(d/'expected-output.rgba16le')==cap['second_output_sha256']==r['outputs'][1]
    return {'schema':'neuralstudio.nr.diagnostic-post-packet-audit.v1','result':'pass_diagnostic_post_operand','reference_root':root,'actual_payload_bytes_checked':sum(sizes.values()),'post_input_pool_sha256':sha(d/'post-input-pool.bin'),'live_whole_model_acceptance':False,'archived_component_inputs':True,'scope':'Verified diagnostic post155 input from snapshot310, plus captured history/motion/parameters and comparison output. This packet cannot establish a live native temporal sequence.','verifier_sha256':sha(Path(__file__))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();r=audit(a.capture);r['receipt_root_sha256']=canonical(r);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
