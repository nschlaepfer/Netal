#!/usr/bin/env python3
"""Audit primitive bits and compare two Q8 hypotheses on a synthetic step texture."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sample(q,mode):
    f=np.float32;u,v=q;tx=f(f(u)*f(1920)-f(.5));ty=f(f(v)*f(1080)-f(.5));x0=int(np.floor(tx));y0=int(np.floor(ty));fx=f(tx-f(x0));fy=f(ty-f(y0));quant=np.floor if mode==0 else np.rint;fx=f(quant(f(fx*256))/256);fy=f(quant(f(fy*256))/256)
    def texel(x,y):
        x=min(max(x,0),1919);y=min(max(y,0),1079);return np.array([x>=384,y>=1,(x+y)&1,1],dtype=np.float32)
    a,b,c,d=texel(x0,y0),texel(x0+1,y0),texel(x0,y0+1),texel(x0+1,y0+1);top=a+(b-a)*fx;bottom=c+(d-c)*fx;return top+(bottom-top)*fy
def audit(d,label):
    job=read(d/(label+'-receipt.json'));run=read(d/'run.json');build=read(d/'build.json')
    assert job['exit_code']==build['exit_code']==0 and run['result']=='pass' and run['exact_write_coverage'] and run['input_immutable']
    assert job['source_sha256']==build['source_sha256']=='996d73ff96ca81d6ee42fdda82876ee7ed8c9256db0669becd34dc4ad2ed7942' and job['binary_sha256']==build['binary_sha256']
    for n in ('run.json','coord-1920.bin','coord-1080.bin','queries.bin','samples.bin'):assert sha(d/n)==job['files'][n]['sha256'] and (d/n).stat().st_size==job['files'][n]['bytes']
    image=np.empty((1080,1920,4),dtype='<u2');x=np.arange(1920)[None,:];y=np.arange(1080)[:,None]
    image[:,:,0]=np.where(x>=384,0x3c00,0);image[:,:,1]=np.where(y>=1,0x3c00,0);image[:,:,2]=np.where((x+y)&1,0x3c00,0);image[:,:,3]=0x3c00
    image_hash=hashlib.sha256(image.tobytes()).hexdigest();assert image_hash==job['files']['input.rgba16le']['sha256']==job['files']['input-after.rgba16le']['sha256']
    rows=[]
    for width in (1920,1080):
        a=np.fromfile(d/f'coord-{width}.bin',dtype='<u4').reshape(width,8)
        assert np.all(a[:,0].view(np.float32)==np.arange(width,dtype=np.float32)+np.float32(.5)) and np.all(a[:,1].view(np.float32)==width)
        rows.append({'dimension':width,'rcp_bits':f'{int(a[0,2]):08x}','div_vs_mul_rcp':int(np.sum(a[:,3]!=a[:,4])),'div_vs_ieee':int(np.sum(a[:,3]!=a[:,6])),'div_vs_mul_rz':int(np.sum(a[:,3]!=a[:,7])),'noncentre_roundtrips':int(np.sum(a[:,0]!=a[:,5])),'requested_row':[f'{int(v):08x}' for v in a[383 if width==1920 else 1]]})
    queries=np.fromfile(d/'queries.bin',dtype='<f4').reshape(-1,2);actual=np.fromfile(d/'samples.bin',dtype='<u4').reshape(-1,4);assert len(queries)==len(actual)==run['sample_queries']==303
    variants=[]
    for mode in (0,1):
        expected=np.array([sample(q,mode) for q in queries],dtype=np.float32).view(np.uint32);bad=np.argwhere(actual!=expected)
        variants.append({'mode':'truncate' if mode==0 else 'nearest_even','unequal_values':len(bad),'total_values':actual.size,'first_counterexamples':[{'query':int(i),'channel':int(k),'actual_bits':f'{int(actual[i,k]):08x}','candidate_bits':f'{int(expected[i,k]):08x}'} for i,k in bad[:8]]})
    return {'schema':'neuralstudio.nr.coordinate-texture-micro-audit.v1','result':'audited_primitive_observation','pid':run['pid'],'poison':run['poison'],'coordinate_results':rows,'sampler_candidates':variants,'input_recipe_sha256':image_hash,'source_sha256':job['source_sha256'],'binary_sha256':job['binary_sha256'],'job_receipt_sha256':sha(d/(label+'-receipt.json')),'input_arrays_downloaded':False,'scope':'Full3000-coordinate RTX arithmetic census and303 known-step texture queries; recipe regenerated on CPU and matches execution-time before/after hashes. The 0/1 texture discriminates fraction rules but does not generally establish arbitrary-half blend-order precision. Not model or Mac temporal acceptance.','verifier_sha256':sha(Path(__file__))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--label',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();r=audit(a.capture,a.label);r['receipt_root_sha256']=canonical(r);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
