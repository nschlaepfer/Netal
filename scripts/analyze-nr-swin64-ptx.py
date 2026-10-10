#!/usr/bin/env python3
"""Validate fresh observational RTX taps and compare one complete CTA on Metal."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def sha(data):return hashlib.sha256(data).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--stage',choices=['ffn','qkv','score0'],required=True);p.add_argument('--mac',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),'preserve analysis')
    job=json.loads((a.job/'receipt.json').read_text(encoding='utf-8-sig'));inputs=json.loads((a.source/'inputs.json').read_text())
    require(job['result']=='pass' and job['inputs_root_sha256']==inputs['receipt_root_sha256'],'job identity')
    body=dict(inputs);root=body.pop('receipt_root_sha256');require(sha(json.dumps(body,sort_keys=True,separators=(',',':')).encode())==root,'input manifest drift')
    for name,digest in inputs['files'].items():require(sha((a.source/name).read_bytes())==digest,'input file drift')
    for entry in job['artifacts']:
        data=(a.job/'output'/entry['name']).read_bytes();require(len(data)==entry['bytes'] and sha(data)==entry['sha256'],'transfer drift')
    size=24576 if a.stage=='qkv' else 8192;pids=set();taps=set();runs=[]
    target='49a2b1f028e022cadfadefd743c9307ca6bcee5f00530fb8f099c1e32f5ea25b'
    for kind in ['original',a.stage+'-tap']:
        for poison in ['a5','5a']:
            for ordinal in ['a','b']:
                stem=f'{kind}-{poison}-{ordinal}';base=a.job/'output';r=json.loads((base/(stem+'.json')).read_text())
                require(r['pid'] not in pids,'reused process');pids.add(r['pid'])
                require(r['result']=='pass' and r['poison']==int(poison,16) and r['tap_enabled']==(kind!='original'),'run identity')
                require(r['tap_coverage_errors']==r['canary_errors']==r['immutable_errors']==0 and r['expected_input_opened'] is False,'guard/ingress failure')
                data=(base/(stem+'.output.e4m3')).read_bytes();require(len(data)==8847360 and sha(data)==target,'PTX does not reproduce original SASS output')
                counts=(base/(stem+'.coverage.bin')).read_bytes();require(len(counts)==size,'coverage extent')
                require(all(x==(kind!='original') for (x,) in struct.iter_unpack('<I',counts)),'tap write coverage')
                if kind!='original':
                    data=(base/(stem+'.tap.bin')).read_bytes();require(len(data)==size,'tap extent');taps.add(sha(data))
                runs.append({'kind':kind,'pid':r['pid'],'poison':r['poison'],'receipt_sha256':sha((base/(stem+'.json')).read_bytes())})
    require(len(taps)==1,'tap process/poison dependence')
    raw=(a.job/'output'/(a.stage+'-tap-a5-a.tap.bin')).read_bytes();decoded=bytearray(size);offsets=[]
    if a.stage=='ffn':
        for warp in range(2):
            for tx in range(2):
                tile=tx*2+warp
                for row in range(16):
                    for n in range(64):
                        reg=tx*16+(n//8)*2+row//8;lane=(row%8)*4+(n%8)//2
                        source=((warp*32+reg)*32+lane)*4+(n%2)*2;dest=((tile*16+row)*64+n)*2
                        decoded[dest:dest+2]=raw[source:source+2];offsets.append(source)
    elif a.stage=='qkv':
        for head in range(2):
            for tile in range(4):
                source_tile=(tile%2)*2+tile//2
                for row in range(16):
                    for n in range(96):
                        reg=source_tile*24+(n//8)*2+row//8;lane=(row%8)*4+(n%8)//2
                        source=((head*96+reg)*32+lane)*4+(n%2)*2;dest=((tile*16+row)*192+head*96+n)*2
                        decoded[dest:dest+2]=raw[source:source+2];offsets.append(source)
    else:
        for head in range(2):
            for tile in range(2):
                for row in range(16):
                    for key in range(64):
                        native_key=((key//16)%2)*32+(key//32)*16+key%16
                        reg=tile*16+(native_key//8)*2+row//8;lane=(row%8)*4+(native_key%8)//2
                        source=((head*32+reg)*32+lane)*4+(native_key%2)*2;dest=(((tile*16+row)*2+head)*64+key)*2
                        decoded[dest:dest+2]=raw[source:source+2];offsets.append(source)
    require(sorted(offsets)==list(range(0,size,2)),'tap decode not bijective')
    mac_path=a.mac/(('scores' if a.stage=='score0' else a.stage)+'.f16');mac=mac_path.read_bytes()[:size]
    if a.stage=='score0':
        data=mac_path.read_bytes();mac=data[:16*2*64*2]+data[32*2*64*2:48*2*64*2]
    require(len(mac)==size,'Mac sample extent')
    bad=[i for i,(x,y) in enumerate(zip(mac,decoded)) if x!=y]
    result={'schema':'neuralstudio.nr.swin64-ptx-tap-analysis.v1','stage':a.stage,'cta':[0,0,0],'rtx_fresh_processes':len(pids),
            'original_sass_output_sha256':target,'original_and_tapped_ptx_output_exact':True,'tap_poison_invariant':True,
            'tap_raw_sha256':next(iter(taps)),'tap_decoded_sha256':sha(decoded),'mac_sample_sha256':sha(mac),
            'compared_bytes':size,'mismatched_bytes':len(bad),'first_mismatched_byte_offsets':bad[:32],
            'tap_decode_bijective':True,'runs':runs,'gpu':job['gpu'],'inputs_root_sha256':root,
            'mac_run_receipt_sha256':sha((a.mac/'run.json').read_bytes()),
            'claims':{'isolated_recorded_block4_input':True,'complete_block5_metal_parity':False,'full_model_inference':False},
            'result':'pass' if not bad else 'mismatch'}
    result['receipt_root_sha256']=sha(json.dumps(result,sort_keys=True,separators=(',',':')).encode())
    a.output.mkdir(parents=True);(a.output/'decoded.f16').write_bytes(decoded);(a.output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'{a.stage} compared_bytes={size} mismatched_bytes={len(bad)} root='+result['receipt_root_sha256'])


if __name__=='__main__':main()
