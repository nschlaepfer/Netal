#!/usr/bin/env python3
"""Read-only audit of the resident-body benchmark against strict reference cases.

This is a controlled body replay, not proof of a new fresh whole-model run.
"""
import argparse
import hashlib
import json
from pathlib import Path
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):
    r=json.loads(p.read_text())
    if 'receipt_root_sha256' in r:
        b=dict(r);h=b.pop('receipt_root_sha256');assert hashlib.sha256(json.dumps(b,sort_keys=True,separators=(',',':')).encode()).hexdigest()==h,p
    return r
def main():
    p=argparse.ArgumentParser();p.add_argument('attempt',type=Path);p.add_argument('--body-prefix',default='');p.add_argument('--reference-archive',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();assert not a.output.exists()
    archive=load(a.reference_archive/'archive.json');assert archive['result']=='pass'
    pinned={str(Path(x['source']).resolve()):x['sha256'] for x in archive['inventory']};runs=[];pids=set()
    for name in ('a5-a','a5-b','5a-a','5a-b'):
        root=a.attempt/(a.body_prefix+name);r=load(root/'resident.json');native=load(root/'run/chain-run.json');plan=load(root/'chain.json');reference=Path(r['reference_body'])
        assert r['result']=='pass' and not r['failures'] and sha(reference/'body.json')==r['reference_body_sha256']==pinned[str((reference/'body.json').resolve())]
        assert native['pid'] not in pids;pids.add(native['pid']);assert int(r['poison'],16)==int(name[:2],16)
        assert native['pool_pre_sha256']==sha(Path(plan['pool_pre']))==load(reference/'body.json')['pool_pre_133_sha256']
        assert [x['launch'] for x in native['launches']]==[x['launch'] for x in plan['launches']]==list(range(133,157))
        for stage,config in zip(native['launches'],plan['launches']):
            n=stage['launch'];rp=reference/f'launch-{n:03d}/report.json';assert sha(rp)==pinned[str(rp.resolve())];ref=load(rp)
            assert stage['pool_post_sha256']==ref['emulated_post_sha256']
            assert stage['words_written_more_than_once']==0 and stage['words_written_once']==ref['coverage']['oracle_changed_words']
            assert stage['weights_immutable'] and stage['params_immutable']
            assert sha(Path(config['params']))==ref['probe']['params_sha256'];assert sha(Path(config['weights']))==ref['probe']['weights_sha256']==stage['weights_sha256']
            lib=Path(config['lib']).resolve();assert sha(lib)==pinned[str(lib)]
            assert config['grid']==ref['grid'] and config['block']==ref['block'] and config['kernel']==ref['metal_kernel']
            if n>=155:
                tex=stage['texture'];assert tex['pixels_once']==tex['pixels']==2073600 and tex['pixels_multi']==0
                f=root/f'run/tex_out_{n}.bin';assert sha(f)==tex['output_sha256']==ref['texture']['output_sha256']
                prior=native['texture_in_sha256'] if n==155 else native['launches'][-2]['texture']['output_sha256']
                assert tex['input_after_sha256']==prior
        assert r['driver_wall_s']<r['reference_harness_s']
        runs.append({'run':name,'pid':native['pid'],'resident_receipt_sha256':sha(root/'resident.json'),'native_receipt_sha256':sha(root/'run/chain-run.json'),
                     'wall_s':r['driver_wall_s'],'reference_wall_s':r['reference_harness_s'],'wall_ratio':r['reference_harness_s']/r['driver_wall_s'],
                     'gpu_ms':r['gpu_ms_total'],'reference_gpu_ms':r['reference_gpu_ms_total'],'all_24_pool_hashes_match':True})
    r={'schema':'neuralstudio.nr.independent-resident-body-audit.v1','strict_reference_archive_root':archive['receipt_root_sha256'],'runs':runs,'result':'pass',
       'scope':'Four controlled body replays on accepted reference inputs; unchanged libraries, dispatches, parameters, per-launch pools and final images. Initial inputs are reused from the strict reference, so this is not fresh whole-model optimized inference.',
       'per_pixel_count_evidence':'Native GPU-side aggregate summaries, checked against strict references; raw count masks not retained by this optimized runner.',
       'fresh_whole_model_optimized_run':False,'real_time_proven':False}
    r['receipt_root_sha256']=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
