#!/usr/bin/env python3
"""Verify complete binary16-derived EX2 observations and build a checked math table."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def audit(captures):
    rows=[];pids=set();base=None;digests=[];source='29c5b8b2314307aec18a2d14514ffdcfbbf31419f4832704a9e2120fe0f3cd50'
    for d,poison in zip(captures,(165,165,90,90)):
        label=d.name;job=read(d/(label+'-receipt.json'));run=read(d/'run.json');build=read(d/'build.json')
        assert job['exit_code']==build['exit_code']==0 and job['source_sha256']==build['source_sha256']==source and job['binary_sha256']==build['binary_sha256']
        for n in ('run.json','values.bin'):assert sha(d/n)==job['files'][n]['sha256'] and (d/n).stat().st_size==job['files'][n]['bytes']
        assert run['schema']=='neuralstudio.nr.ex2-scaled-half-oracle.v1' and run['result']=='pass' and run['rows']==65536 and run['scale_bits']=='0xbfb8aa3b'
        assert run['coverage_once'] and run['canaries_intact'] and run['model_executed'] is False and run['poison']==poison and run['pid']>0 and run['pid'] not in pids;pids.add(run['pid'])
        a=np.fromfile(d/'values.bin',dtype='<u4').reshape(65536,3);assert np.array_equal(a[:,0],np.arange(65536,dtype=np.uint32))
        if base is None:base=a.copy()
        else:assert np.array_equal(a,base),'fresh-process/poison mismatch'
        digests.append(job['binary_sha256']);rows.append({'pid':run['pid'],'poison':poison,'values_sha256':sha(d/'values.bin'),'job_receipt_sha256':sha(d/(label+'-receipt.json'))})
    assert len(captures)==4 and len(set(digests))==1
    hb=np.arange(65536,dtype=np.uint16);finite=(hb&0x7c00)!=0x7c00;alpha=hb.view(np.float16).astype(np.float64)
    scale=struct.unpack('<f',bytes.fromhex('3baab8bf'))[0]
    with np.errstate(invalid='ignore',over='ignore'):arg=(alpha*float(scale)).astype(np.float32).view(np.uint32)
    assert np.array_equal(arg[finite],base[:,1][finite]),'finite argument construction'
    assert int(base[0xbb80,1])==0x3fad1f97 and int(base[0xbb80,2])==0x40236e03
    report={'schema':'neuralstudio.nr.ex2-scaled-half-domain.v1','result':'pass_complete_half_domain_observation','runs':rows,'source_sha256':source,'binary_sha256':digests[0],'domain_rows':65536,'finite_argument_rows_verified':int(finite.sum()),'scale_bits':'0xbfb8aa3b','known_counterexample':{'half_bits':'0xbb80','argument_bits':'0x3fad1f97','rtx_ex2_bits':'0x40236e03'},'index':'original binary16 operand bits, never a pixel/frame/model-output index','integration_contract':'Keep the original half alpha operand. Require the computed fp32 argument bits to equal the table argument word, then return the recorded fp32 EX2 word. Restrict promoted use to verified finite-half inputs; do not infer a general-f32 EX2 implementation or recover the index approximately.','model_executed':False,'full_model_parity':False,'nonfinite_rows':'Observed and repeated, but no cross-backend NaN-payload/exception contract promoted.','verifier_sha256':sha(Path(__file__))}
    return base[:,1:3].astype('<u4').tobytes(),report
def main():
    p=argparse.ArgumentParser();p.add_argument('--captures',type=Path,nargs=4,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();table,r=audit(a.captures);a.output.mkdir(parents=True)
    path=a.output/'ex2-scaled-half.u32x2';path.write_bytes(table);r['table']={'file':path.name,'bytes':len(table),'sha256':sha(path),'layout':'65536 little-endian uint2 pairs: exact argument bits, exact RTX EX2 bits'};r['receipt_root_sha256']=canonical(r);(a.output/'verification.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
