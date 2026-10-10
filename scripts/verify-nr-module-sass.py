#!/usr/bin/env python3
"""Hash-check retained driver-JIT artifacts and expose bounded UTF-8 kernels."""
import argparse
import hashlib
import json
from pathlib import Path
import re
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('results',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    assert not a.output.exists(),'preserve results';a.output.mkdir(parents=True)
    inputs=json.loads((a.source/'inputs.json').read_text());r=json.loads((a.results/'receipt.json').read_text(encoding='utf-8-sig'))
    root=dict(inputs);claimed=root.pop('receipt_root_sha256');assert hashlib.sha256(json.dumps(root,sort_keys=True,separators=(',',':')).encode()).hexdigest()==claimed
    assert r['inputs_root_sha256']==claimed and r['gpu_executed'] is False
    for name,h in inputs['files'].items():assert sha(a.source/name)==h,name
    for item in r['artifacts']:
        path=a.results/'output'/item['name'];assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],item['name']
    requested={'cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8',
               'cc_tinlayout_fused_post_block_swin_1h_32_fp8',
               'cc_tinlayout_fused_swin_1h_32_1_upsample_tilesync_fp8',
               'cc_tinlayout_fused_swin_1h_32_1_chained_fp8',
               'cc_tinlayout_fused_swin_1h_32_1_outview_fp8'}
    sections=[]
    for mod in r['module_results']:
        assert mod['driver_jit_exit']==0,mod
        path=a.results/'output'/(mod['module']+'.sass.txt');text=path.read_text(encoding='utf-16')
        matches=list(re.finditer(r'^\s*Function : (\S+)\s*$',text,re.M))
        for i,m in enumerate(matches):
            name=m[1]
            if name not in requested:continue
            block=text[m.start():matches[i+1].start() if i+1<len(matches) else len(text)]
            file=a.output/(name+'.sass.txt');file.write_text(block)
            arithmetic=[line for line in block.splitlines() if re.search(r'\b(?:HFMA2|HADD2|HMUL2|QMMA|HMMA|MUFU)(?:\.|\s)',line)]
            ops=a.output/(name+'.arithmetic.txt');ops.write_text('\n'.join(arithmetic)+'\n')
            sections.append({'kernel':name,'module':mod['module'],'sass_sha256':sha(file),'arithmetic_sha256':sha(ops),'arithmetic_instructions':len(arithmetic)})
    found={x['kernel'] for x in sections};assert requested-found <= {'cc_tinlayout_fused_swin_1h_32_1_outview_fp8'},requested-found
    result={'schema':'neuralstudio.nr.selected-driver-sass-verification.v1','input_root_sha256':claimed,'job_receipt_sha256':sha(a.results/'receipt.json'),
            'sections':sections,'compatibility_build_results':r['module_results'],'original_ptx_modified_for_driver_jit':False,
            'gpu_kernels_executed':False,'capture_live_jit_identity_proven':False,'result':'pass',
            'scope':'Original PTX compiled with installed driver via cuLink; exposes contraction instructions. Equivalence to archived live code and numerical model output must be separately checked.'}
    result['receipt_root_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (a.output/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
