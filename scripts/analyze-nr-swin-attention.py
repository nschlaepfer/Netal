#!/usr/bin/env python3
"""Diagnose Swin cosine-attention tensors from live native QKV outputs."""
import argparse
import json
import math
from pathlib import Path
import struct
from nr_swin_reference import load_cta,load_module


def norm32(values,h):
    lanes=[]
    for p in range(8):
        a=h(values[8+p]*values[8+p]+h(values[24+p]*values[24+p]))
        b=h(values[p]*values[p]+h(values[16+p]*values[16+p]))
        lanes.append(h(a+b))
    low=h(h(lanes[0]+lanes[4])+h(lanes[2]+lanes[6]))
    high=h(h(lanes[1]+lanes[5])+h(lanes[3]+lanes[7]))
    return max(h(low+high),6.198883056640625e-05)


def bias_offset(query,key):
    qt,kt=query//16,key//16
    return (11360+((qt&1)*2+(qt>>1))*2048+((kt&1)*2+(kt>>1))*512+
            ((query%8)*4+(key%8)//2)*16+((key%16)//8)*8+
            ((query%16)//8*2+key%2)*2)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--qkv-attempt',type=Path,required=True)
    p.add_argument('--fixtures',type=Path,default=Path('build/nr_swin_mlp/fixtures'))
    p.add_argument('--cta',type=int,default=0)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--export',type=Path)
    a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    if a.output.exists():raise ValueError('preserve prior analysis')
    m=load_module('nr_mlp',repo/'scripts/analyze-nr-swin-mlp.py')
    q,cta,records,pins=load_cta(repo,a.workspace,a.cta)
    data=(a.qkv_attempt/f'cta{a.cta}-a5-a/qkv.f16').read_bytes()
    values=struct.unpack('<6144e',data)
    weights=(a.fixtures/f'cta{a.cta}/inputs/weight.bin').read_bytes()
    scale=m.half(struct.unpack_from('<f',weights,19552)[0])
    decode=[q.decode_e4m3(i) for i in range(256)]
    qs=[];ks=[];sums=[]
    for row in range(64):
        vectors=[]
        row_sums=[]
        for component in range(2):
            v=values[row*96+component*32:row*96+component*32+32]
            norm=norm32(v,m.half);row_sums.append(norm)
            factor=m.half(1/math.sqrt(norm))
            normalized=[m.half(x*factor) for x in v]
            if component==0:normalized=[m.half(x*scale) for x in normalized]
            vectors.append(bytes(m.fp8(normalized[m.dense_from_channel(k)],decode) for k in range(32)))
        qs.append(vectors[0]);ks.append(vectors[1]);sums.append(row_sums)
    biases={}
    for base in range(11360,19552,512):
        for fragment in range(2):
            v=b''.join(weights[base+((row%8)*4+col//2)*16+fragment*8+(row//8*2+col%2)*2:
                              base+((row%8)*4+col//2)*16+fragment*8+(row//8*2+col%2)*2+2]
                        for row in range(16) for col in range(8))
            biases.setdefault(v,[]).append((base,fragment))
    results=[];mismatches={'query':0,'key':0,'scores':0};failures=[]
    expected_norm=[None]*4096;expected_scores=[None]*4096;score_coverage=[0]*4096
    for index,r in enumerate(records):
        if r['c'] not in biases:continue
        query_rows=[];key_rows=[]
        for row in range(16):
            v=r['a'][row*32:row*32+32]
            errors=[sum(x!=y for x,y in zip(v,computed)) for computed in qs]
            selected=errors.index(min(errors));query_rows.append(selected)
            mismatches['query']+=errors[selected]
        for col in range(8):
            v=bytes(r['b'][k*8+col] for k in range(32))
            errors=[sum(x!=y for x,y in zip(v,computed)) for computed in ks]
            selected=errors.index(min(errors));key_rows.append(selected)
            mismatches['key']+=errors[selected]
        c=struct.unpack('<128e',r['c']);d=struct.unpack('<128H',r['d']);computed=[]
        for row in range(16):
            for col in range(8):
                x,y=qs[query_rows[row]],ks[key_rows[col]]
                offset=bias_offset(query_rows[row],key_rows[col])
                if weights[offset:offset+2]!=r['c'][(row*8+col)*2:(row*8+col+1)*2]:
                    raise ValueError('bias permutation mismatch')
                value=c[row*8+col]+sum(decode[x[k]]*decode[y[k]] for k in range(32))
                computed.append(struct.unpack('<H',struct.pack('<e',value))[0])
                position=query_rows[row]*64+key_rows[col]
                expected_scores[position]=d[row*8+col];score_coverage[position]+=1
            for k in range(32):
                position=query_rows[row]*64+k;v=r['a'][row*32+k]
                if expected_norm[position] not in (None,v):raise ValueError('query observation inconsistent')
                expected_norm[position]=v
        for col in range(8):
            for k in range(32):
                position=key_rows[col]*64+32+k;v=r['b'][k*8+col]
                if expected_norm[position] not in (None,v):raise ValueError('key observation inconsistent')
                expected_norm[position]=v
        mismatch=sum(x!=y for x,y in zip(computed,d));mismatches['scores']+=mismatch
        if mismatch and len(failures)<5:failures.append({'op':index,'differing_halves':mismatch})
        results.append({'op':index,'bias_candidates':biases[r['c']],
                        'query_rows':query_rows,'key_rows':key_rows,'score_half_mismatches':mismatch})
    result={'schema':'neuralstudio.nr.swin32-attention-semantic.v1','cta':cta,
            'score_operations':len(results),'query_scale_f16':scale,'sum_squares':sums,
            'mismatches':mismatches,'first_failures':failures,'operations':results,
            'full_model_inference':False,'result':'pass' if len(results)==32 and not any(mismatches.values()) else 'fail'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    if a.export:
        if a.export.exists():raise ValueError('preserve prior reference')
        if result['result']!='pass' or set(score_coverage)!={1} or None in expected_norm:
            raise ValueError('incomplete attention reference')
        a.export.mkdir(parents=True)
        (a.export/'normalized.e4m3').write_bytes(bytes(expected_norm))
        (a.export/'scores.f16').write_bytes(struct.pack('<4096H',*expected_scores))
        provenance={'cta':cta,'source_observations':pins,'qkv_native_output_sha256':q.sha(data),
                    'comparison_only':True,'normalized_sha256':q.sha(bytes(expected_norm)),
                    'scores_sha256':q.sha(struct.pack('<4096H',*expected_scores)),
                    'result':'pass'}
        (a.export/'receipt.json').write_text(json.dumps(provenance,indent=2,sort_keys=True)+'\n')
    print({k:v for k,v in result.items() if k not in ('operations','sum_squares')})


if __name__=='__main__':main()
