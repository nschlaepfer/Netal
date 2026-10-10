#!/usr/bin/env python3
"""Compare the frozen existing CPU sampler rule with the newly captured RTX bits.
No model/kernel changes, no GPU execution, no expected bytes in candidate input.
"""
import argparse,ast,hashlib,json
from pathlib import Path
import numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--peer',type=Path,required=True);p.add_argument('--fixture',type=Path,required=True);p.add_argument('--rtx',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    source=a.peer/'scripts/ptx_warp_emu.py';source_sha=sha(source);text=source.read_text();tree=ast.parse(text)
    names={'_rtx_accumulate','tex_bilinear_rtx'}
    if any(isinstance(n,ast.FunctionDef) and n.name=='_rtx_nonfinite' for n in tree.body):names.add('_rtx_nonfinite')
    nodes=[]
    for n in tree.body:
        if isinstance(n,ast.FunctionDef) and n.name in names:nodes.append(n)
        elif isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='RTX_TEX_NAN_BITS':
            assert isinstance(n.value,ast.Constant) and type(n.value.value)is int;nodes.append(n)
    assert {n.name for n in nodes if isinstance(n,ast.FunctionDef)}==names
    frozen='\n\n'.join(ast.get_source_segment(text,n) for n in nodes);namespace={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
    f=json.loads((a.fixture/'fixture.json').read_text());assert f['input_sha256']=='98bb7f6defd425955e46f043c722a397453d70cb4812fe71c28cadcfe6624093'
    image=np.zeros((1080,1920,4),dtype='<u2')
    for n,c in enumerate(f['cases']):
        for tap in range(4):
            for ch in range(4):image[8+tap//2,8*n+2+tap%2,ch]=c['taps'][(tap+ch)%4]
    assert hashlib.sha256(image.tobytes()).hexdigest()==f['input_sha256']
    q=np.fromfile(a.fixture/'queries.bin',dtype='<f4').reshape(-1,2)
    with np.errstate(all='ignore'):
        values=namespace['tex_bilinear_rtx'](image.view('<f2').astype(np.float32),1920,1080,q[:,0].astype(np.float64),q[:,1].astype(np.float64))
        got=np.stack(values,axis=1).astype('<f4').view('<u4')
    expected=np.fromfile(a.rtx/'a5-a/samples.bin',dtype='<u4').reshape(-1,4);assert got.shape==expected.shape==(1078,4)
    got.tofile(a.out/'cpu-samples.bin');rows=[];bad=np.argwhere(got!=expected)
    for i,c in enumerate(f['cases']):
        mask=(got!=expected)[i*49:(i+1)*49];finite=all((x&0x7c00)!=0x7c00 for x in c['taps'])
        rows.append({'case':c['name'],'finite_input_pattern':finite,'values':int(mask.size),'mismatching_values':int(mask.sum())})
    assert sha(source)==source_sha
    result={'schema':'neuralstudio.nr.sampler-edge-cpu-comparison.v1','scope':'Existing frozen CPU rule vs repeated RTX primitive; NOT Metal execution or full-model acceptance','emulator_source_sha256':source_sha,'frozen_functions':frozen,'input_sha256':f['input_sha256'],'rtx_samples_sha256':sha(a.rtx/'a5-a/samples.bin'),'cpu_samples_sha256':sha(a.out/'cpu-samples.bin'),'mismatching_values':int(len(bad)),'finite_pattern_mismatches':sum(r['mismatching_values'] for r in rows if r['finite_input_pattern']),'cases':rows,'counterexamples':[{'query':int(i),'channel':int(ch),'case':f['cases'][i//49]['name'],'cpu_bits':f'{int(got[i,ch]):08x}','rtx_bits':f'{int(expected[i,ch]):08x}'} for i,ch in bad],'rtx_mac_parity':False}
    result['receipt_root_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest();(a.out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('frozen_functions','counterexamples')},indent=2))
if __name__=='__main__':main()
