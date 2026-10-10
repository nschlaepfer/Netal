#!/usr/bin/env python3
"""Independent all-output quantization-cell check; not an enclosure proof.

Uses an enumerated half lattice and nearest-even FP8 neighbors, not the
prototype's endpoint rounding functions. No GPU and no prototype modifications.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def check(ok,label):
    if not ok:raise ValueError(label)
def audit(folder,expected):
    rec=json.loads((folder/'census.json').read_text());b=dict(rec);h=b.pop('receipt_root_sha256');check(canonical(b)==h,'census seal')
    before={str(p):sha(p) for p in (folder/'census.json',folder/'per-output.npz',expected)}
    arrays=np.load(folder/'per-output.npz',allow_pickle=False);P=arrays['P_L'];B=arrays['B_L'];mask=arrays['certified'];code=arrays['code']
    version2='seal' in rec
    if version2:
        seal=rec['seal'];check(seal['inputs_unchanged'] is True and seal['inputs_sha256_pre_run']==seal['inputs_sha256_post_run'],'input pre/post identity')
        check(sha(Path(seal['script']))==seal['script_sha256'],'census source identity')
        check(sha(folder/'per-output.npz')==seal['per_output_archive_sha256'],'sealed per-output archive')
        for key,h in seal['inputs_sha256_pre_run'].items():
            path=Path(rec['inputs'][key]['file']) if key in rec['inputs'] else Path(key)
            check(sha(path)==h,'current input identity')
    check(P.shape==B.shape==mask.shape==code.shape==(640,1024),'complete real projection')
    check(np.all(B>=0),'nonnegative primary radii')
    # Independent logical output address map from the reviewed Metal contract.
    t=np.arange(640,dtype=np.int64)[:,None];n=np.arange(1024,dtype=np.int64)[None,:]
    c=(n&~31)+(n&16)+(n&1)+2*(n&6)+((n&8)//4)
    lane=4*(t%8)+(c%16)//4;reg=(t//8)%2+2*((c%32)//16)
    idx=(t//16)*16384+(c//32)*512+lane*16+reg*4+c%4
    check(np.all(np.bincount(idx.ravel(),minlength=655360)==1),'layout bijection')
    golden=np.fromfile(expected,dtype='u1')[idx];check(np.array_equal(code,golden),'all exact-reference FP8 bytes')
    # Enumerate every finite half as an exact integer multiple of2^-30.
    bits=np.arange(65536,dtype=np.int64);e=(bits>>10)&31;bits=bits[e!=31];e=(bits>>10)&31;f=bits&1023
    mag=np.where(e==0,f<<6,(f+1024)<<(e+5));values=np.where((bits&32768)!=0,-mag,mag)
    order=np.argsort(values,kind='stable');values=values[order];bits=bits[order];mag=np.abs(values)
    # Independently derive nearest-even finite E4M3 encoding for each half.
    q=np.arange(127,dtype=np.int64);qe=q>>3;qf=q&7
    qvalue=np.where(qe==0,qf<<21,(8+qf)<<(qe+20))
    upper=np.minimum(np.searchsorted(qvalue,mag),126);lower=np.maximum(upper-1,0)
    dl=mag-qvalue[lower];du=np.abs(qvalue[upper]-mag)
    choose_upper=(du<dl)|((du==dl)&((upper&1)==0));mcode=np.where(choose_upper,upper,lower)
    hcode=(mcode|np.where((bits&32768)!=0,128,0)).astype('u1')
    lo=P-B;hi=P+B;i=np.searchsorted(values,lo,side='left');j=np.searchsorted(values,hi,side='right')-1
    check(np.all(~mask|((i<=j)&(i>=0)&(j<len(values)))),'certified interval contains valid half states')
    check(np.all(~mask|((lo>0)|(hi<0))),'signed-zero exclusion')
    check(np.all(~mask|(np.maximum(np.abs(lo),np.abs(hi))<(448<<30))),'saturation exclusion')
    low=hcode[np.clip(i,0,len(hcode)-1)];high=hcode[np.clip(j,0,len(hcode)-1)]
    check(np.all(~mask|((low==high)&(low==golden))),'independent quantization cells')
    enclosure_checked=False
    if version2:
        hb=arrays['T3_bits'].astype(np.int64);he=(hb>>10)&31;hf=hb&1023
        check(hb.shape==P.shape and np.all((hb>=0)&(hb<65536)) and np.all(he!=31),'finite retained terminal-half reference')
        hm=np.where(he==0,hf<<6,(hf+1024)<<(he+5));hv=np.where((hb&32768)!=0,-hm,hm)
        check(np.all((hv>=lo)&(hv<=hi)),'all retained reference halves inside primary enclosures')
        codec=np.zeros(65536,dtype='u1');codec[bits]=hcode;check(np.array_equal(codec[hb],golden),'terminal half reference codec')
        z=(lo<=0)&(hi>=0);sat=np.maximum(np.abs(lo),np.abs(hi))>=(448<<30)
        eligible=~z&~sat&~arrays['overflow']&~arrays['nonfinite'];empty=i>j
        check(np.array_equal(z,arrays['contains_zero']) and np.array_equal(sat,arrays['saturation']) and np.array_equal(eligible,arrays['eligible']) and np.array_equal(empty,arrays['empty_interval']),'saved primary eligibility masks')
        recomputed=eligible&~empty&(low==high);check(np.array_equal(mask,recomputed),'full primary classification')
        check(np.all(arrays['Bfp32_L']>=B),'modeled radius cannot narrow primary interval')
        enclosure_checked=True
    groups=mask.reshape(40,16,16,64).all(axis=(1,3));tiles=mask.reshape(40,16,128,8).all(axis=(1,3))
    check(int(mask.sum())==rec['certified'] and int(groups.sum())==rec['clustering']['workgroups_16x64_fully_certified'] and int(tiles.sum())==rec['clustering']['tiles_16x8_fully_certified'],'reported census/clustering')
    for p,h in before.items():check(sha(Path(p))==h,'audit input drift')
    return {'schema':'neuralstudio.nr.projection-cells-independent-audit.v1','result':'quantization_cells_and_reference_pass','census_root':rec['receipt_root_sha256'],'input_hashes':before,'outputs_checked':640*1024,'certified':int(mask.sum()),'certified_fraction':float(mask.mean()),'fully_certified_workgroups':int(groups.sum()),'fully_certified_instruction_tiles':int(tiles.sum()),'retained_terminal_half_enclosures_checked':enclosure_checked,'enclosure_proof_accepted':False,'modeled_fp32_accepted':False,'metal_speedup':False,'limits':['Independently validates primary certificates against finite-half cells and real FP8 reference bytes; v2 additionally checks retained T3 enclosures and masks.','This is a data/reproduction audit, not a universal proof of interval derivation or a verified Metal predictor contract.','No complete-group GPU cost has been measured.'],'verifier_sha256':sha(Path(__file__))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--census-dir',required=True,type=Path);p.add_argument('--expected',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();check(not a.output.exists(),'preserve audit')
    try:r=audit(a.census_dir,a.expected)
    except Exception as e:r={'schema':'neuralstudio.nr.projection-cells-independent-audit.v1','result':'fail','error':str(e)}
    r['receipt_root_sha256']=canonical(r);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
    if r['result']=='fail':raise SystemExit(3)
if __name__=='__main__':main()
