#!/usr/bin/env python3
"""Recover probability, value and output projection tensor semantics."""
import argparse
import itertools
import json
from pathlib import Path
import struct
from nr_swin_reference import load_cta,load_module


def probability_row(scores,m,decode):
    numerators=[]
    for score in scores:
        value=m.half(score*0.044921875+1.30078125)
        value=max(1.03125,min(1.5693359375,value))
        bits=struct.unpack('<H',struct.pack('<e',value))[0]
        numerators.append(struct.unpack('<e',struct.pack('<H',((bits<<5)+0x8000)&65535))[0])
    h=m.half
    s=[]
    for p in range(8):
        value=h(h(numerators[p]+numerators[p+8])+h(numerators[p+32]+numerators[p+40]))
        value=h(value+h(numerators[p+16]+numerators[p+24]))
        s.append(h(value+h(numerators[p+48]+numerators[p+56])))
    lo=h(h(h(s[0]+s[2])+s[4])+s[6]);hi=h(h(h(s[1]+s[3])+s[5])+s[7])
    denominator=max(h(lo+hi),6.198883056640625e-05)
    inv=h(1/denominator)
    return bytes(m.fp8(h(x*inv),decode) for x in numerators),denominator


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--cta',type=int,default=0);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--export',type=Path)
    a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    if a.output.exists():raise ValueError('preserve analysis')
    q,cta,records,pins=load_cta(repo,a.workspace,a.cta)
    m=load_module('nr_mlp',repo/'scripts/analyze-nr-swin-mlp.py')
    root=repo/f'build/nr_swin_attention/macos-v2/cta{a.cta}-a5-a'
    scores=struct.unpack('<4096e',(root/'scores.f16').read_bytes())
    qkv=struct.unpack('<6144e',(root/'qkv.f16').read_bytes())
    partial=struct.unpack('<8192e',(root/'partial.f16').read_bytes())
    weights=(repo/f'build/nr_swin_mlp/fixtures/cta{a.cta}/inputs/weight.bin').read_bytes()
    decode=[q.decode_e4m3(i) for i in range(256)]
    probs=[];denoms=[]
    for row in range(64):
        ps,denom=probability_row(scores[row*64:row*64+64],m,decode)
        probs.append(ps);denoms.append(denom)
    qkv_orders={x['qmmaa_order'] for x in json.loads((repo/f'build/nr_swin_qkv/reference-v1/cta{a.cta}/receipt.json').read_text())['operations']}
    attn_path=repo/('build/nr_swin_attention/semantic-v2.json' if a.cta==0 else f'build/nr_swin_attention/semantic-cta{a.cta}-export.json')
    score_orders={x['op'] for x in json.loads(attn_path.read_text())['operations']}
    projection_b={bytes(weights[19568+m.packed(k,n+j)] for k in range(32) for j in range(8)):n for n in range(0,32,8)}
    keysets=[tuple(x*16+m.dense_from_channel(k) for k in range(16))+
             tuple(y*16+m.dense_from_channel(k) for k in range(16))
             for x,y in itertools.permutations(range(4),2)]
    prob_candidates={}
    for tile in range(4):
        for keys in keysets:
            values=bytes(probs[tile*16+row][k] for row in range(16) for k in keys)
            prob_candidates.setdefault(values,[]).append((tile,keys))
    value_quant=[[m.fp8(qkv[row*96+64+k],decode) for k in range(32)] for row in range(64)]
    value_candidates={}
    for keys in keysets:
        for n in range(0,32,8):
            for perm in ('identity','channel'):
                columns=[n+j if perm=='identity' else m.dense_from_channel(n+j) for j in range(8)]
                b=bytes(value_quant[row][c] for row in keys for c in columns)
                value_candidates.setdefault(b,[]).append((keys,n,perm))
    attended=[[None]*32 for _ in range(64)];operations=[];failures=[]
    expected_prob=[None]*4096;expected_value=[None]*4096;expected_projection=[None]*2048
    mismatches={'probability_bytes':0,'value_bytes':0,'accumulator_bytes':0,'output_bytes':0}
    for index,r in enumerate(records):
        if index<128 or index in qkv_orders or index in score_orders:continue
        if r['b'] in projection_b:continue
        if r['a'] not in prob_candidates:
            best=min(prob_candidates,key=lambda v:sum(x!=y for x,y in zip(v,r['a'])))
            mismatch=sum(x!=y for x,y in zip(best,r['a']));mismatches['probability_bytes']+=mismatch
            choices=prob_candidates[best]
        else:choices=prob_candidates[r['a']];mismatch=0
        bchoices=value_candidates.get(r['b'],[])
        if not bchoices:
            failures.append({'op':index,'error':'value layout not matched'});mismatches['value_bytes']+=256;continue
        matches=[(tile,keys,n,perm) for tile,keys in choices for bkeys,n,perm in bchoices if keys==bkeys]
        if len(matches)!=1:
            failures.append({'op':index,'error':'ambiguous shared key layout','matches':len(matches)});continue
        tile,keys,n,perm=matches[0]
        if perm!='identity':print('value column permutation',perm)
        cb=[];db=[]
        for row in range(16):
            for j in range(8):
                column=n+j;previous=attended[tile*16+row][column]
                if previous is None:previous=0.0
                cb.append(previous)
                total=previous+sum(decode[probs[tile*16+row][key]]*decode[value_quant[key][column]] for key in keys)
                value=m.half(total);db.append(value);attended[tile*16+row][column]=value
        cbytes=struct.pack('<128e',*cb);dbytes=struct.pack('<128e',*db)
        chunk=0 if keys[0]==0 else 1
        actual_d=struct.unpack('<128H',r['d'])
        for row in range(16):
            for k,key in enumerate(keys):
                pos=(tile*16+row)*64+key;v=r['a'][row*32+k]
                if expected_prob[pos] not in (None,v):raise ValueError('probability references disagree')
                expected_prob[pos]=v
            for j in range(8):expected_value[(chunk*64+tile*16+row)*32+n+j]=actual_d[row*8+j]
        cm=sum(x!=y for x,y in zip(cbytes,r['c']));dm=sum(x!=y for x,y in zip(dbytes,r['d']))
        mismatches['accumulator_bytes']+=cm;mismatches['output_bytes']+=dm
        operations.append({'op':index,'query_tile':tile,'key_rows':list(keys),'column':n,
                           'probability_mismatch':mismatch,'c_mismatch':cm,'d_mismatch':dm})
    projection_errors={'a':0,'c':0,'d':0};projection_ops=[]
    projection_inputs=[bytes(m.fp8(attended[tile*16+row][m.dense_from_channel(k)],decode)
                             for row in range(16) for k in range(32)) for tile in range(4)]
    scales=struct.unpack_from('<32e',weights,20592)
    for index,r in enumerate(records):
        if r['b'] not in projection_b:continue
        errors=[sum(x!=y for x,y in zip(v,r['a'])) for v in projection_inputs]
        tile=errors.index(min(errors));projection_errors['a']+=min(errors)
        n=projection_b[r['b']];cv=[];dv=[]
        for row in range(16):
            for j in range(8):
                previous=m.half(partial[6144+(tile*16+row)*32+n+j]*scales[n+j])
                cv.append(previous)
                total=previous+sum(decode[projection_inputs[tile][row*32+k]]*decode[r['b'][k*8+j]] for k in range(32))
                dv.append(m.half(total))
        cm=sum(x!=y for x,y in zip(struct.pack('<128e',*cv),r['c']))
        dm=sum(x!=y for x,y in zip(struct.pack('<128e',*dv),r['d']))
        projection_errors['c']+=cm;projection_errors['d']+=dm
        actual_d=struct.unpack('<128H',r['d'])
        for row in range(16):
            for j in range(8):expected_projection[(tile*16+row)*32+n+j]=actual_d[row*8+j]
        projection_ops.append({'op':index,'tile':tile,'column':n,'a_mismatch':min(errors),'c_mismatch':cm,'d_mismatch':dm})
    result={'schema':'neuralstudio.nr.swin32-values-semantic.v1','cta':cta,'value_operation_count':len(operations),
            'mismatches':mismatches,'failures':failures[:8],'operations':operations,'denominators':denoms,
            'projection_errors':projection_errors,'projection_operations':projection_ops,
            'full_model_inference':False,'result':'pass' if len(operations)==32 and len(projection_ops)==16 and not any(mismatches.values()) and not any(projection_errors.values()) and not failures else 'fail'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    if a.export:
        if a.export.exists():raise ValueError('preserve reference')
        if result['result']!='pass' or any(None in values for values in (expected_prob,expected_value,expected_projection)):
            raise ValueError('cannot export an incomplete semantic model')
        a.export.mkdir(parents=True)
        payloads={'probabilities.e4m3':bytes(expected_prob),'value_partial.f16':struct.pack('<4096H',*expected_value),
                  'projection.f16':struct.pack('<2048H',*expected_projection)}
        for name,data in payloads.items():(a.export/name).write_bytes(data)
        provenance={'cta':cta,'source_observations':pins,'outputs':{name:q.sha(data) for name,data in payloads.items()},
                    'comparison_only':True,'result':'pass'}
        (a.export/'receipt.json').write_text(json.dumps(provenance,indent=2,sort_keys=True)+'\n')
    print({k:v for k,v in result.items() if k not in ('operations','denominators','projection_operations')})
    if operations:print('first_layout',operations[0])


if __name__=='__main__':main()
