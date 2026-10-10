#!/usr/bin/env python3
"""Recover the first Swin MLP as tensors and compare to recorded RTX probes.

Recorded inputs are used only for isolated block diagnosis. This program is
not an end-to-end inference executor and does not generate model weights.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import struct


def half(x):
    return struct.unpack('<e',struct.pack('<e',x))[0]


def fp8(x, decode):
    sign=128 if struct.unpack('<H',struct.pack('<e',x))[0]&0x8000 else 0
    x=abs(x)
    # The finite codebook has monotone magnitude. Ties choose an even code.
    best=min(range(127),key=lambda c:(abs(decode[c]-x),c&1))
    return best|sign


def packed(k,n):
    return (n//16)*512+(n%8)*64+((k%16)//4)*16+((n%16)//8)*8+k%4+(k//16)*4


def channel_from_dense(n):
    return (n&1)+((n&6)<<1)+((n&8)>>2)+(n&16)


def dense_from_channel(k):
    return (k&1)+((k&12)>>1)+((k&2)<<2)+(k&16)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--cta',default='0-0')
    p.add_argument('--ordinal',type=int,default=0)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--export',type=Path)
    a=p.parse_args()
    if a.output.exists(): raise ValueError('preserve previous analysis')
    repo=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('nr_qmma',repo/'scripts/build-dlssnr-block1-qmma-all.py')
    q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
    weights=json.loads((repo/'build/nr_first_inference/weights-manifest.json').read_text())
    t=next(x for x in weights['tensors'] if x['name']=='block1.layer0.layer')
    runtime=(repo/'models/nvidia/nvngx_dlssnr.dll').read_bytes()
    start=weights['resource']['file_offset']+t['data_offset_in_resource']
    w=runtime[start:start+t['data_size']]
    assert q.sha(w)==t['sha256']
    ir=json.loads((a.workspace/'work/subagents/backbone-05-12/block1-prep/block1-normalized-ir.json').read_text())
    instructions=[x for x in ir['instructions'] if x.get('opcode','').startswith('QMMA')]
    trace=a.workspace/'work/block1-qmma-nine-cta-capture/dlssnr-block1-qmma-nine-cta-20260831-bl/raw'
    stem=f'repeat1-cta{a.ordinal}-{a.cta}'
    ins=q.records(trace/(stem+'-input.stdout'))
    outs=q.records(trace/(stem+'-output.stdout'))
    assert len(ins)==len(outs)==len(instructions)==256
    records=[]
    for op,i,o in zip(instructions,ins,outs):
        match=q.OPERANDS.match(op['operands'])
        d,ab,bb=map(int,match.group(1,2,3));cb=None if match.group(4) is None else int(match.group(4))
        assert i['instruction_index']==o['instruction_index']==op['index']
        ri,ro=q.register_columns(i),q.register_columns(o)
        records.append({'a':q.matrix_a(ri,ab),'b':q.matrix_b(ri,bb),
                        'c':bytes(256) if cb is None else q.matrix_half(ri,cb),
                        'd':q.matrix_half(ro,d)})
    decode=[q.decode_e4m3(x) for x in range(256)]
    inputs=[records[i]['a'] for i in (0,2,4,6)]
    # Four 16-token tiles are the logical 8x8 window, in NVIDIA tile order.
    x=[[decode[b] for b in block] for block in inputs]
    expansion=[[None]*128 for _ in range(64)]
    activation=[[None]*128 for _ in range(64)]
    residual=[[None]*32 for _ in range(64)]
    expected_expansion=[[None]*128 for _ in range(64)]
    expected_activation=[[None]*128 for _ in range(64)]
    expected_partial=[[[None]*32 for _ in range(64)] for _ in range(4)]
    candidate_b={}
    for chunk in range(4):
        for n in range(0,32,8):
            for kind,base in [('expand',chunk*1024),('contract',4096+chunk*1024)]:
                values=bytes(w[base+packed(k,n+j)] for k in range(32) for j in range(8))
                assert values not in candidate_b
                candidate_b[values]=(kind,chunk,n)
    scales=list(struct.unpack_from('<32e',w,8208))
    rows=[]
    mismatches={'a':0,'b':0,'c':0,'d':0}
    first=[]
    for index,r in enumerate(records[:128]):
        kind,chunk,n=candidate_b[r['b']]
        if kind=='expand':
            tile=inputs.index(r['a'])
            ca=r['a']
            cc=bytes(256)
        else:
            # Match the logical token tile by the generated activation bytes.
            options=[bytes(activation[tile*16+row][chunk*32+dense_from_channel(k)]
                           for row in range(16) for k in range(32))
                     for tile in range(4)]
            if r['a'] not in options:
                # Diagnose the smallest disagreement without expected ingress.
                tile=min(range(4),key=lambda t:sum(u!=v for u,v in zip(options[t],r['a'])))
            else:
                tile=options.index(r['a'])
            ca=options[tile]
            cvalues=[]
            for row in range(16):
                for j in range(8):
                    value=residual[tile*16+row][n+j]
                    if value is None:
                        value=half(x[tile][row*32+channel_from_dense(n+j)]*scales[n+j])
                    cvalues.append(value)
            cc=struct.pack('<128e',*cvalues)
        result=[]
        cvalues=struct.unpack('<128e',cc)
        for row in range(16):
            for j in range(8):
                total=cvalues[row*8+j]
                for k in range(32):
                    total+=decode[ca[row*32+k]]*decode[r['b'][k*8+j]]
                value=half(total)
                result.append(value)
                if kind=='expand':
                    column=chunk*32+n+j
                    expansion[tile*16+row][column]=value
                    clipped=max(-4.0,min(4.0,value))
                    t0=half(abs(clipped)*-half(struct.unpack('<e',struct.pack('<H',0x2b28))[0])+0.447265625)
                    t1=half(clipped*t0+0.89453125)
                    act=half(value*t1)
                    activation[tile*16+row][column]=fp8(act,decode)
                else:
                    residual[tile*16+row][n+j]=value
        cd=struct.pack('<128e',*result)
        actual_d=struct.unpack('<128H',r['d'])
        for row in range(16):
            for j in range(8):
                if kind=='expand':
                    expected_expansion[tile*16+row][chunk*32+n+j]=actual_d[row*8+j]
                else:
                    expected_partial[chunk][tile*16+row][n+j]=actual_d[row*8+j]
            if kind=='contract':
                for k in range(32):
                    expected_activation[tile*16+row][chunk*32+k]=r['a'][row*32+k]
        local={key:sum(u!=v for u,v in zip(got,r[key])) for key,got in [('a',ca),('b',r['b']),('c',cc),('d',cd)]}
        for key,value in local.items():mismatches[key]+=value
        if any(local.values()) and len(first)<12:first.append({'op':index,'kind':kind,'chunk':chunk,'n':n,'tile':tile,'mismatch':local})
        rows.append({'op':index,'kind':kind,'chunk':chunk,'n':n,'tile':tile,**local})
    result={'schema':'neuralstudio.nr.swin32-mlp-semantic-diagnostic.v1',
            'cta':a.cta,'weight_sha256':t['sha256'],'operation_count':128,
            'mismatched_bytes':mismatches,'first_failures':first,'operations':rows,
            'diagnostic_recorded_input':True,'full_inference':False,
            'result':'pass' if not any(mismatches.values()) else 'fail'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    if a.export:
        if result['result']!='pass':raise ValueError('cannot export a failing semantic interpretation')
        if a.export.exists():raise ValueError('preserve existing fixture')
        (a.export/'inputs').mkdir(parents=True)
        (a.export/'comparison').mkdir()
        (a.export/'inputs/input.e4m3').write_bytes(b''.join(inputs))
        (a.export/'inputs/weight.bin').write_bytes(w)
        def store_half(name,values):
            (a.export/'comparison'/name).write_bytes(b''.join(struct.pack('<H',v) for v in values))
        store_half('expansion.f16', [v for row in expected_expansion for v in row])
        (a.export/'comparison/activation.e4m3').write_bytes(bytes(v for row in expected_activation for v in row))
        store_half('partial.f16',[v for chunk in expected_partial for row in chunk for v in row])
        provenance={'diagnostic_recorded_input':True,'cta':a.cta,'weight_sha256':t['sha256'],
                    'input_trace_sha256':q.sha((trace/(stem+'-input.stdout')).read_bytes()),
                    'output_trace_sha256':q.sha((trace/(stem+'-output.stdout')).read_bytes()),
                    'comparison_bytes_bound_to_runtime':False,
                    'files':{str(f.relative_to(a.export)):q.sha(f.read_bytes()) for f in a.export.rglob('*') if f.is_file()}}
        (a.export/'provenance.json').write_text(json.dumps(provenance,indent=2,sort_keys=True)+'\n')
    print({k:v for k,v in result.items() if k!='operations'})


if __name__=='__main__':main()
