#!/usr/bin/env python3
"""Independent fresh-producer audit of the resident full-chain instrument."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):
    r=json.loads(p.read_text())
    if 'receipt_root_sha256' in r:
        b=dict(r);h=b.pop('receipt_root_sha256');assert canonical(b)==h,p
    return r
def option(step,key):
    c=step['command'];assert c.count(key)==1;return Path(c[c.index(key)+1]).resolve()
def main():
    p=argparse.ArgumentParser();p.add_argument('attempt',type=Path);p.add_argument('--reference-archive',required=True,type=Path);p.add_argument('--output-dir',required=True,type=Path);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    root=a.attempt.resolve();repo=Path(__file__).resolve().parents[1];archive=read(a.reference_archive/'archive.json');pinned={str(Path(x['source']).resolve()):x['sha256'] for x in archive['inventory']}
    chain=read(root/'fast-chain.json');reference=Path(chain['reference_attempt']).resolve();assert sha(reference/'full-chain.json')==pinned[str(reference/'full-chain.json')]
    assert chain['result']=='pass' and not chain['failures'];names=('a5-a','a5-b','5a-a','5a-b');assert set(chain['stages'])==set(names)
    steps={s['label']:s for s in chain['steps']};assert len(steps)==len(chain['steps']) and all(s['exit_code']==0 for s in steps.values())
    enc=Path(chain['encoder_run']).resolve();assert enc.is_relative_to(root) and enc!=root
    assert option(steps['encoder'],'--attempt')==enc
    result=subprocess.run([sys.executable,str(repo/'scripts/verify-nr-live-encoder.py'),str(enc),'--output',str(a.output_dir/'encoder-audit.json')],capture_output=True,text=True)
    (a.output_dir/'encoder-audit.log').write_text(result.stdout+result.stderr);assert result.returncode==0,'fresh encoder audit failed'
    ev=read(a.output_dir/'encoder-audit.json');pids={r[k] for r in ev['runs'] for k in ('block0_pid','layout_pid','encoder_pid')};assert len(pids)==12
    middle_results=[]
    for name in names:
        poison=int(name[:2],16);mid=root/f'middle-{name}';mr=read(mid/'resident.json');mp=read(mid/'chain.json');native=read(mid/'run/chain-run.json');entry=enc/name/'encoder/block22-middle.e4m3'
        assert mr['result']=='pass' and not mr['failures'] and mr['encoder_entry_sha256']==sha(entry)
        assert Path(mp['entry_file']).resolve()==entry and option(steps[f'middle-{name}'],'--encoder-entry')==entry
        assert native['pid'] not in pids;pids.add(native['pid']);assert native['poison']==poison
        assert mp['entry_offset']==86649856 and mp['pool_bytes']==212519936 and mp['final_offset']==123145216
        for key in ('split_lib','vit_lib'):
            path=Path(mp[key]).resolve();assert sha(path)==pinned[str(path)]
        refmid=reference/f'middle-{name}'/name;expected=[]
        expected += [(n,None) for n in range(25,58)];expected += [(58,'nr_vit_repack_2d_to_1d')]
        expected += [(n,'vit_block') for n in range(31,39)];expected += [(99,'nr_vit_repack_1d_to_2d')];expected += [(n,None) for n in range(100,133)]
        assert len(native['launches'])==len(expected)==76
        checks=0
        for L,(n,kind) in zip(native['launches'],expected):
            assert L['launch']==n and (kind is None or L['kind']==kind)
            if kind=='vit_block':
                rp=refmid/f'block{n}/run.json';rr=read(rp);assert sha(rp)==pinned[str(rp)]
                assert set(L['outputs'])==set(rr['stages'])
                for stage,h in L['outputs'].items():assert h==rr['stages'][stage]['sha256'] and L['coverage'][stage]==[rr['stages'][stage]['bytes'],0,0];checks+=1
            elif kind:
                rp=refmid/f'repack{n}/run.json';assert sha(rp)==pinned[str(rp)]
                assert L['coverage_dwords']==[163840,0,0] and L['outputs'][kind]==sha(refmid/f'repack{n}'/(kind+'.bin'));checks+=1
            else:
                rp=refmid/f'launch-{n:03d}/run.json';rr=read(rp);assert sha(rp)==pinned[str(rp)]
                assert set(L['outputs'])==set(rr['outputs'])
                for role,h in L['outputs'].items():
                    assert h==rr['outputs'][role]['sha256'];m=rr['outputs'][role];comp=m.get('compared_bytes',m['bytes']);assert L['coverage'][role]==[comp,0,m['bytes']-comp];checks+=1
        assert checks==125 and sha(mid/'run/out_view_132.bin')==mr['final_out_view_sha256']==sha(refmid/'launch-132/out_view.bin')
        body=root/f'body-{name}';bp=read(body/'chain.json');bn=read(body/'run/chain-run.json');assert bn['pid'] not in pids;pids.add(bn['pid'])
        assert option(steps[f'body-{name}'],'--encoder-run')==enc/name/'encoder' and option(steps[f'body-{name}'],'--middle-out')==mid/'run/out_view_132.bin'
        assert option(steps[f'body-{name}'],'--block0-skip')==enc/name/'block0/skip.e4m3'
        initial=np.full(212519936,poison,dtype=np.uint8)
        inputs={90827776:enc/name/'encoder/block22-native.e4m3',93285376:enc/name/'encoder/block14-native.e4m3',97709056:enc/name/'encoder/block8-native.e4m3',141945856:enc/name/'encoder/block4-native.e4m3',123145216:mid/'run/out_view_132.bin',388096:enc/name/'block0/skip.e4m3'}
        for offset,path in inputs.items():data=np.fromfile(path,dtype=np.uint8);initial[offset:offset+data.size]=data
        initial[233984:387616]=255;pool=Path(bp['pool_pre']).resolve();assert pool.is_relative_to(body) and np.array_equal(initial,np.memmap(pool,dtype=np.uint8,mode='r'))
        middle_results.append({'run':name,'middle_pid':native['pid'],'body_pid':bn['pid'],'middle_stage_hashes_matched':checks,'middle_receipt_sha256':sha(mid/'resident.json'),'middle_native_receipt_sha256':sha(mid/'run/chain-run.json')})
    result=subprocess.run([sys.executable,str(repo/'scripts/audit-nr-resident-body.py'),str(root),'--body-prefix','body-','--reference-archive',str(a.reference_archive),'--output',str(a.output_dir/'body-audit.json')],capture_output=True,text=True)
    (a.output_dir/'body-audit.log').write_text(result.stdout+result.stderr);assert result.returncode==0,'resident body audit failed'
    body=read(a.output_dir/'body-audit.json');assert len(pids)==20
    out={'schema':'neuralstudio.nr.independent-fast-chain-audit.v1','attempt':str(root),'chain_receipt_sha256':sha(root/'fast-chain.json'),'strict_reference_archive_root':archive['receipt_root_sha256'],
         'encoder_audit_root':ev['receipt_root_sha256'],'body_audit_root':body['receipt_root_sha256'],'middle':middle_results,'fresh_native_processes':20,
         'result':'pass','scope':'Four fresh pinned-frame original-image chains through resident middle/body. Stage hashes, unchanged libraries, actual fresh entries/pools and coverage summaries checked against strict references. No quiet-timing or broader-input guarantee.',
         'timing_contended':True,'broader_input_modes_validated':False}
    out['receipt_root_sha256']=canonical(out);(a.output_dir/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out['receipt_root_sha256'])
if __name__=='__main__':main()
