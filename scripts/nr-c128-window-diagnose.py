#!/usr/bin/env python3
"""Isolate the Block11 counterexample and compare compact arithmetic precisions.

CPU diagnostic only. Runtime inputs are the computed Block10 output and original
weights; RTX/Metal Block11 bytes are opened only after producing each candidate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('preserve diagnostic')
    repo=Path(__file__).resolve().parents[1];snapshot=repo/'build/nr_c128_counterexample/source-v1'
    sys.path.insert(0,str(snapshot/'scripts'));from nr_fp8 import E4M3,encode_e4m3,weight_index
    prior=(repo/'build/nr_encoder14/macos-v2/a5-a/block10.e4m3').read_bytes()
    if sha(prior)!='7723c35dc91e13a4811a50c1c23ee3c9595f0ef367a6f472b34ebf2f9da8325f':raise ValueError('live producer drift')
    wraw=(repo/'build/nr_encoder14/reference-v1/weights/block11.bin').read_bytes()
    if sha(wraw)!='dd5440dccd5fa2be9b63421d2a67715b8c41abf783464601a78bf138f479a40f':raise ValueError('weight drift')
    def gather(data):
        out=bytearray()
        for tile in range(4):
            for p in range(16):
                x=60+tile//2*4+p%4;y=8+tile%2*4+p//4
                row=((y//8)*30+x//8)*64+((x%8)//4)*32+((y%8)//4)*16+(y%4)*4+x%4
                out.extend(data[row*128:row*128+128])
        return bytes(out)
    xraw=gather(prior);x=np.frombuffer(xraw,dtype=np.uint8).reshape(64,128);w=np.frombuffer(wraw,dtype=np.uint8)
    h=lambda x:np.asarray(x,dtype=np.float16)
    f=lambda x:np.asarray(x,dtype=np.float32)
    q=lambda x:encode_e4m3(f(x))
    dec=lambda x:E4M3[x].astype(np.float32)
    c=np.arange(128);perm=(c&1)+((c&12)>>1)+((c&2)<<2)+(c&16)+(c&~31)
    inverse=(c&1)+((c&6)<<1)+((c&8)>>2)+(c&16)+(c&~31)
    perm32=perm[:32]
    rsq=np.frombuffer((snapshot/'build/nr_vit/tables/rtx-rsqrt-half.bin').read_bytes(),dtype='<u2').view(np.float16)
    rcp=np.frombuffer((snapshot/'build/nr_vit/tables/rtx-rcp-half.bin').read_bytes(),dtype='<u2').view(np.float16)
    def table(values,t):
        bits=h(values).view(np.uint16)
        if np.any(bits<1) or np.any(bits>0x7bff):raise ValueError('lookup outside proved half domain')
        return t[bits.astype(np.int32)-1]
    def matrix(offset,K,N):
        kk=np.arange(K)[:,None];nn=np.arange(N)[None,:]
        return dec(w[offset+(kk//32)*(N*32)+weight_index(kk%32,nn)])
    def dot(X,W,C=None):
        acc=np.zeros((len(X),W.shape[1]),np.float16) if C is None else h(C)
        for s in range(X.shape[1]//32):
            v=f(acc)
            for k in range(32):v=f(v+X[:,s*32+k,None]*W[None,s*32+k,:])
            acc=h(v)
        return acc
    aa=np.arange(64)[:,None];bb=np.arange(64)[None,:]
    qt=aa//16;kt=bb//16;qt=(qt%2)*2+qt//2;kt=(kt%2)*2+kt//2
    biasidx=qt*2048+kt*512+((aa%8)*4+(bb%8)//2)*16+((bb%16)//8)*8+((aa%16)//8*2+bb%2)*2
    def scales(off,N):return np.frombuffer(wraw[off:off+2*N],dtype='<f2')
    a.output.mkdir(parents=True);(a.output/'input.e4m3').write_bytes(xraw);(a.output/'weight.bin').write_bytes(wraw)
    results=[]
    for mode in ('float32_fma','exact_half_fma'):
        dt=np.float32 if mode=='float32_fma' else np.float64
        hfma=lambda a,b,c:h(np.asarray(a,dtype=dt)*np.asarray(b,dtype=dt)+np.asarray(c,dtype=dt))
        stage={};exp=[]
        for g in range(4):exp.append(dot(dec(x),matrix(g*16384,128,128)))
        expanded=np.concatenate(exp,axis=1);stage['expansion']=expanded
        pp=np.arange(512);hp=(pp//32)*32+perm32[pp%32]
        value=expanded[:,hp];clip=np.clip(value,-4,4)
        t=hfma(np.abs(clip),h(-0.055908203125),h(0.447265625));t=hfma(clip,t,h(0.89453125))
        activation=q(h(f(value)*f(t)));stage['activation']=activation
        mid=np.concatenate([dot(dec(activation[:,g*128:(g+1)*128]),matrix(65536+g*4096,128,32)) for g in range(4)],axis=1);stage['mid']=mid
        ffn=dot(dec(q(mid[:,perm])),matrix(81920,128,128),h(dec(x[:,inverse])*f(scales(98320,128))));stage['ffn']=ffn
        qkv=dot(dec(q(ffn[:,perm])),matrix(98592,128,384));stage['qkv']=qkv
        normalized=np.empty((64,4,2,32),np.uint8)
        for head in range(4):
            for component in range(2):
                v=qkv[:,head*96+component*32:head*96+component*32+32]
                P=h(f(hfma(v[:,8:16],v[:,8:16],h(f(v[:,24:32])*f(v[:,24:32]))))+f(hfma(v[:,:8],v[:,:8],h(f(v[:,16:24])*f(v[:,16:24])))))
                lo=h(f(h(f(P[:,0])+f(P[:,4])))+f(h(f(P[:,2])+f(P[:,6]))));hi=h(f(h(f(P[:,1])+f(P[:,5])))+f(h(f(P[:,3])+f(P[:,7]))))
                den=np.maximum(h(f(lo)+f(hi)),h(6.198883056640625e-05));inv=table(den,rsq)
                value=h(f(v[:,perm32])*f(inv[:,None]))
                if component==0:value=h(f(value)*f(h(np.frombuffer(wraw[180512:180528],dtype='<f4')[head])))
                normalized[:,head,component,:]=q(value)
        stage['normalized']=normalized;all_scores=[];all_probs=[];all_values=[]
        for head in range(4):
            bias=np.frombuffer(wraw,dtype='<f2')[(147744+head*8192+biasidx)//2]
            scores=dot(dec(normalized[:,head,0]),dec(normalized[:,head,1]).T,bias);all_scores.append(scores)
            v=np.clip(hfma(scores,h(0.044921875),h(1.30078125)),h(1.03125),h(1.5693359375))
            e=((v.view(np.uint16).astype(np.uint32)<<5)+0x8000).astype(np.uint16).view(np.float16)
            S=[]
            for i in range(8):
                v=h(f(h(f(e[:,i])+f(e[:,i+8])))+f(h(f(e[:,i+32])+f(e[:,i+40]))))
                v=h(f(v)+f(h(f(e[:,i+16])+f(e[:,i+24]))));S.append(h(f(v)+f(h(f(e[:,i+48])+f(e[:,i+56])))))
            lo=h(f(h(f(h(f(S[0])+f(S[2])))+f(S[4])))+f(S[6]));hi=h(f(h(f(h(f(S[1])+f(S[3])))+f(S[5])))+f(S[7]))
            den=np.maximum(h(f(lo)+f(hi)),h(6.198883056640625e-05));prob=q(h(f(e)*f(table(den,rcp)[:,None])));all_probs.append(prob)
            order=np.array([chunk*16+(32 if k>=16 else 0)+int(perm32[k%16]) for chunk in range(2) for k in range(32)])
            all_values.append(dot(dec(prob[:,order]),dec(q(qkv[:,head*96+64:head*96+96]))[order]))
        stage['scores']=np.stack(all_scores,axis=1);stage['probabilities']=np.stack(all_probs,axis=1)
        values=np.concatenate(all_values,axis=1);stage['values']=values
        projection=dot(dec(q(values[:,perm])),matrix(180528,128,128),h(dec(q(ffn))*f(scales(196912,128))));stage['projection']=projection
        out=q(projection[:,perm]);stage['output']=out
        path=a.output/mode;path.mkdir()
        for name,data in stage.items():(path/(name+('.f16' if data.dtype==np.float16 else '.e4m3'))).write_bytes(data.tobytes())
        expected=gather((repo/'build/nr_encoder14/reference-v1/block11.e4m3').read_bytes());mac=gather((repo/'build/nr_encoder14/macos-v2/a5-a/block11.e4m3').read_bytes())
        result={'mode':mode,'rtx_mismatches':sum(x!=y for x,y in zip(out.tobytes(),expected)),'mac_mismatches':sum(x!=y for x,y in zip(out.tobytes(),mac)),'stage_sha256':{name:sha(data.tobytes()) for name,data in stage.items()}}
        results.append(result);print(mode,'RTX',result['rtx_mismatches'],'Mac',result['mac_mismatches'],flush=True)
    receipt={'schema':'neuralstudio.nr.c128-window-numeric-diagnostic.v1','cta':[8,1,0],'origin':[60,8],'input_sha256':sha(xraw),'weight_sha256':sha(wraw),'results':results,'full_model_inference':False}
    receipt['receipt_root_sha256']=sha(json.dumps(receipt,sort_keys=True,separators=(',',':')).encode());(a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
