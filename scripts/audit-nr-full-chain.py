#!/usr/bin/env python3
"""Independent read-only audit: separate matching frames from strict provenance.

Never edits peer receipts or executes GPU work. Findings are preserved in a new
local audit. A matching image is not automatically strict acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
FRAME='a07835db4be8469dad1e3d3befcc495be396790f89408be62ddb8a5a710585a2'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
RUNS=('a5-a','a5-b','5a-a','5a-b')
def seal_issues(chain,encoder,root):
    """Pure metadata checks, also exercised by negative controls."""
    issues=[];root=Path(root).resolve()
    if chain.get('result')!='pass':issues.append('full-chain result is not pass')
    steps=chain.get('steps',[])
    if not steps:issues.append('missing executed steps')
    for step in steps:
        if step.get('exit_code')!=0:issues.append('required step failed: '+step.get('label','?'))
    if not any(s.get('label')=='encoder-verify' and s.get('exit_code')==0 for s in steps):issues.append('successful encoder-verify step required')
    labels=[s.get('label') for s in steps]
    for stage in ('middle','body'):
        for run in RUNS:
            if labels.count(stage+'-'+run)!=1:issues.append('missing/duplicate required stage command: '+stage+'-'+run)
    if set(chain.get('stages',{}))!=set(RUNS):issues.append('two fresh full chains per poison required')
    enc=chain.get('encoder_run',{});path=Path(enc.get('path','/')).resolve()
    try:relative=path.relative_to(root)
    except ValueError:relative=None
    if relative is None or not relative.parts:issues.append('encoder resume is not a same-attempt producer')
    if encoder.get('result')!='pass':issues.append('successful independent encoder verification required')
    embedded=enc.get('verification')
    embedded_ok=isinstance(embedded,dict) and bool(embedded.get('receipt_root_sha256')) and embedded.get('receipt_root_sha256')==encoder.get('receipt_root_sha256')
    reference_ok=enc.get('verification_result')=='pass' and bool(enc.get('verification_root')) and enc.get('verification_root')==encoder.get('receipt_root_sha256') and isinstance(enc.get('verification_sha256'),str) and len(enc['verification_sha256'])==64
    if not(embedded_ok or reference_ok):issues.append('full-chain seal does not incorporate successful encoder verification')
    claims=encoder.get('claims',{})
    if claims.get('recorded_block0_entry') is not False or claims.get('native_image_to_blocks0_through22') is not True:issues.append('encoder native-entry provenance missing')
    if encoder.get('fresh_native_processes')!=12:issues.append('encoder fresh-process matrix incomplete')
    return issues
def main():
    p=argparse.ArgumentParser();p.add_argument('attempt',type=Path);p.add_argument('--peer-checkout',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    assert not a.output.exists(),'preserve prior audit';root=a.attempt.resolve();peer=a.peer_checkout.resolve();findings=[];witnesses={};pids=set()
    def check(ok,label):
        if not ok:findings.append(label)
    def read(path):
        path=Path(path)
        if not path.is_file():findings.append('missing receipt: '+str(path));return {}
        witnesses[str(path)]=sha(path);r=json.loads(path.read_text())
        if 'receipt_root_sha256' in r:
            body=dict(r);expected=body.pop('receipt_root_sha256');check(canonical(body)==expected,'invalid canonical root: '+str(path))
        return r
    def inside(path):
        try:Path(path).resolve().relative_to(root);return True
        except ValueError:return False
    def resolve(path):
        f=Path(path);return f.resolve() if f.is_absolute() else (peer/f).resolve()
    def file_hash(path,expected,label):
        path=Path(path);check(path.is_file(),label+' missing')
        if not isinstance(expected,str) or len(expected)!=64:check(False,label+' expected digest not provided')
        elif path.is_file():check(sha(path)==expected,label+' hash mismatch')
    def native_run(r,label,poison):
        if r.get('schema')=='neuralstudio.nr.ptx-probe.v1':
            check(r.get('weights_immutable') is True and r.get('params_immutable') is True,label+' immutable constants')
        else:check(r.get('result')=='pass',label+' result')
        if 'poison' in r:check(r['poison']==poison,label+' poison')
        if 'pid' in r:
            check(r['pid'] not in pids,label+' reused process');pids.add(r['pid'])
    def output_meta(meta,path,label,full=True):
        for k in ('mismatches','coverage_errors','coverage_errors_dwords','canary_errors'):
            if k in meta:check(meta[k]==0,label+' '+k)
        if 'input_immutable' in meta:check(meta['input_immutable'] is True,label+' mutated input')
        if 'sha256' in meta:
            file_hash(path,meta['sha256'],label)
            if 'expected_sha256' in meta and full:check(meta['sha256']==meta['expected_sha256'],label+' RTX mismatch')
    chain=read(root/'full-chain.json')
    check(Path(chain.get('attempt','/')).resolve()==root,'attempt identity')
    steps=chain.get('steps',[])
    enc_path=Path(chain.get('encoder_run',{}).get('path',root/'encoder-run')).resolve();check(inside(enc_path),'encoder resume outside attempt')
    ev=read(enc_path/'verification.json');findings.extend(seal_issues(chain,ev,root))
    if 'verification_sha256' in chain.get('encoder_run',{}):file_hash(enc_path/'verification.json',chain['encoder_run']['verification_sha256'],'sealed encoder verification')
    for run in ev.get('runs',[]):
        for key in ('block0_pid','layout_pid','encoder_pid'):
            pid=run.get(key);check(pid is not None and pid not in pids,'encoder PID matrix missing/duplicate');pids.add(pid)
    index=read(peer/'build/nr_split/fixtures/index.json').get('launches',{})
    frames={};repack_ingress=[];texture_gates={}
    stage_keys=list(RUNS) if set(chain.get('stages',{}))==set(RUNS) else [k for k in ('a5','5a') if k in chain.get('stages',{})]
    for run_id in stage_keys:
        poison=run_id.split('-')[0];ordinal=run_id.split('-')[1] if '-' in run_id else 'a'
        value=int(poison,16);entry=enc_path/f'{poison}-{ordinal}/encoder/block22-middle.e4m3';mid=root/f'middle-{run_id}';base=mid/f'{poison}-{ordinal}'
        prov=read(mid/'provenance.json').get('encoder_entry',{});check(prov.get('live') is True,'recorded middle entry '+poison)
        check(resolve(prov.get('file','/'))==entry.resolve(),'wrong middle producer '+poison);file_hash(entry,prov.get('sha256'),'middle entry '+poison)
        live={86649856:entry}
        def split(n):
            run=base/f'launch-{n:03d}';r=read(run/'run.json');native_run(r,f'{poison} launch{n}',value)
            check(r.get('immutable_input_errors')==0,f'{poison} launch{n} input mutation')
            meta=index.get(str(n),{});check(bool(meta),'missing launch binding metadata '+str(n))
            for b in meta.get('bindings',{}).values():
                if b['direction']=='in':
                    expected=live.get(b['pool_offset']);actual=r.get('input_sources',{}).get(b['role'])
                    check(expected is not None,f'{poison} launch{n} missing producer '+b['role'])
                    check(bool(actual) and expected is not None and resolve(actual)==expected.resolve(),f'{poison} launch{n} nonlive/fallback input '+b['role'])
                    if expected is not None:file_hash(expected,r.get('inputs_sha256',{}).get(b['role']),f'{poison} launch{n} input '+b['role'])
            for role,m in r.get('outputs',{}).items():output_meta(m,run/(role+'.bin'),f'{poison} launch{n} '+role,full=not(n==57 and role=='out_vit'))
            for b in meta.get('bindings',{}).values():
                if b['direction']=='out':live[b['pool_offset']]=run/(b['role']+'.bin')
        for n in range(25,58):split(n)
        # The final head deliberately leaves100 out-of-image tokens poisoned;
        # only the documented shape padding may be zero-initialized for ViT.
        raw=np.fromfile(base/'launch-057/out_vit.bin',dtype=np.uint8)
        expected=np.fromfile(peer/'build/nr_split/fixtures/launch-057/out_vit.bin',dtype=np.uint8)
        mask=np.zeros(655360,dtype=bool)
        for t in range(640):
            tile,slot=divmod(t,16);rr=4*(tile//8)+slot//4;cc=4*(tile%8)+slot%4
            if rr>=18 or cc>=30:continue
            channels=np.arange(1024,dtype=np.uint32);lane=4*(t%8)+((channels%32)%16)//4;reg=((t%16)//8)+2*((channels%32)//16)
            offsets=(t//16)*(16*1024)+(channels//32)*512+lane*16+reg*4+channels%4;mask[offsets]=True
        check(raw.size==expected.size==655360 and int(mask.sum())==552960,poison+' final-head mask geometry')
        check(np.array_equal(raw[mask],expected[mask]),poison+' final-head active values differ')
        check(bool(np.all(raw[~mask]==value)),poison+' undocumented final-head padding writes')
        padded=np.fromfile(base/'vit_input_padded.bin',dtype=np.uint8);wanted=raw.copy();wanted[~mask]=0
        check(padded.size==720896 and np.array_equal(padded[:655360],wanted) and bool(np.all(padded[655360:]==0)),poison+' shape padding changed live values')
        for direction in (58,99):
            r=read(base/f'repack{direction}/run.json');native_run(r,f'{poison} repack{direction}',value)
            for kernel,meta in r.get('kernels',{}).items():
                key='repack58' if kernel=='nr_vit_repack_2d_to_1d' else 'repack99';source=r.get('input_sources',{}).get(key)
                ok=bool(source) and source!='fixture' and inside(resolve(source))
                check(ok,f'{poison} repack{direction} executed {key} with fixture/external input')
                repack_ingress.append({'run':run_id,'poison':poison,'invocation':direction,'executed':key,'source':source,'inside_attempt':ok})
                output_meta(meta,base/f'repack{direction}'/(kernel+'.bin'),f'{poison} repack{direction} '+kernel)
        prev=base/'repack58/nr_vit_repack_2d_to_1d.bin'
        for b in range(31,39):
            run=base/f'block{b}';r=read(run/'run.json');native_run(r,f'{poison} ViT{b}',value)
            check(r.get('isolated') is False and resolve(r.get('tokens_source','/'))==prev.resolve(),f'{poison} ViT{b} recorded/nonlive entry')
            file_hash(prev,r.get('inputs_sha256',{}).get('tokens'),f'{poison} ViT{b} input')
            check(r.get('immutable_input_errors')==0,f'{poison} ViT{b} input mutation')
            for stage,m in r.get('stages',{}).items():output_meta(m,run/(stage+'.bin'),f'{poison} ViT{b} '+stage)
            prev=run/'projection.bin'
        live[211864576]=base/'repack99/nr_vit_repack_1d_to_2d.bin'
        for n in range(100,133):split(n)
        body_dir=root/f'body-{run_id}';body=read(body_dir/'body.json');check(body.get('result')=='pass',run_id+' body result')
        launches=[x['launch'] for x in body.get('launches',[])]
        check(launches==list(range(133,157)),run_id+' complete decoder/output/copy inventory')
        expected_sources={90827776:enc_path/f'{poison}-{ordinal}/encoder/block22-native.e4m3',93285376:enc_path/f'{poison}-{ordinal}/encoder/block14-native.e4m3',
            97709056:enc_path/f'{poison}-{ordinal}/encoder/block8-native.e4m3',141945856:enc_path/f'{poison}-{ordinal}/encoder/block4-native.e4m3',
            123145216:base/'launch-132/out_view.bin',388096:enc_path/f'{poison}-{ordinal}/block0/skip.e4m3'}
        check({s['pool_offset'] for s in body.get('inputs',[])}==set(expected_sources),run_id+' body producer inventory')
        initial=np.full(212519936,value,dtype=np.uint8)
        for source in body.get('inputs',[]):
            check(source.get('live',True) is True,poison+' body recorded input')
            f=Path(source['file']);f=f if f.is_absolute() else enc_path/f'{poison}-{ordinal}/encoder'/f
            check(inside(f),poison+' body external producer');file_hash(f,source['sha256'],poison+' body input')
            off=source['pool_offset'];check(off in expected_sources and f.resolve()==expected_sources[off].resolve(),run_id+' body input bound to wrong producer')
            if f.is_file():
                data=np.fromfile(f,dtype=np.uint8);check(data.size==source['bytes'],run_id+' body input extent')
                if 0<=off and off+data.size<=initial.size:initial[off:off+data.size]=data
                else:check(False,run_id+' body input outside pool')
        control=body.get('control_words',{});check(control=={'range':[233984,387616],'initial':'0xff'},run_id+' undeclared counter initialization')
        initial[233984:387616]=255;prepool=body_dir/'pool_pre_133.bin'
        file_hash(prepool,body.get('pool_pre_133_sha256'),run_id+' initial pool')
        if prepool.is_file():check(np.array_equal(np.memmap(prepool,dtype=np.uint8,mode='r'),initial),run_id+' initial pool contains undeclared/archived bytes')
        previous_hash=body.get('pool_pre_133_sha256');previous_path=prepool
        for stage in body.get('launches',[]):
            check(stage['result']=='pass' and stage.get('mismatching_bytes')==0,poison+' decoder launch '+str(stage['launch']))
            report=read(body_dir/f'launch-{stage["launch"]:03d}/report.json');probe=report.get('probe',{})
            native_run(probe,poison+' decoder probe '+str(stage['launch']),value)
            check(report.get('pre_sha256')==previous_hash and resolve(report.get('pool_pre_source','/'))==previous_path.resolve(),run_id+' decoder pool handoff '+str(stage['launch']))
            previous_hash=report.get('emulated_post_sha256');previous_path=body_dir/f'launch-{stage["launch"]:03d}/run/pool_post.bin'
        file_hash(previous_path,previous_hash,run_id+' final pool producer')
        final_launch=156 if 156 in launches else 155
        texture=body_dir/f'launch-{final_launch}/run/tex_out.bin';frames[run_id]=sha(texture) if texture.is_file() else None
        check(frames[run_id]==FRAME,run_id+' final image bytes differ from RTX')
        texture_gates[run_id]={}
        for n in (155,156):
            if n not in launches:continue
            report=read(body_dir/f'launch-{n}/report.json');tm=report.get('texture',{});probe=report.get('probe',{});counts=body_dir/f'launch-{n}/run/tex_coverage.bin'
            pb=tm.get('poison_byte');observed=int(pb,16) if isinstance(pb,str) and pb.lower() in ('a5','5a','0xa5','0x5a') else pb
            check(observed==value and probe.get('texture_poison')==value,run_id+f' texture{n} poison not evidenced')
            check(counts.is_file(),run_id+f' texture{n} per-pixel write counts not retained');covered=False
            if counts.is_file():
                c=np.fromfile(counts,dtype='<u4');covered=c.size==1920*1080 and bool(np.all(c==1));check(covered,run_id+f' texture{n} missing/multiple writes')
            check(tm.get('input_texture_immutable') is True and bool(probe.get('texture_in_sha256')) and probe.get('texture_in_sha256')==probe.get('texture_in_after_sha256'),run_id+f' texture{n} input mutation check')
            if n==155:check(probe.get('texture_in_sha256')==chain.get('original_image_sha256'),run_id+' post stage changed original-image input')
            else:
                source=body_dir/'launch-155/run/tex_out.bin';check(resolve(tm.get('input_texture','/'))==source.resolve(),run_id+' copy not fed by live post texture');file_hash(source,probe.get('texture_in_sha256'),run_id+' copy input hash')
            file_hash(body_dir/f'launch-{n}/run/tex_out.bin',tm.get('output_sha256'),run_id+f' texture{n} output hash')
            texture_gates[run_id][str(n)]={'poison_byte':observed,'coverage_file':str(counts),'exactly_once':covered}
    result={'schema':'neuralstudio.nr.independent-full-chain-audit.v1','attempt':str(root),'chain_receipt_sha256':sha(root/'full-chain.json'),
            'final_frame_sha256':frames,'both_frames_match_rtx':len(frames)>=2 and all(h==FRAME for h in frames.values()),'four_frames_match_rtx':set(frames)==set(RUNS) and all(h==FRAME for h in frames.values()),'findings':findings,
            'repack_ingress':repack_ingress,'texture_gates':texture_gates,'observed_receipt_hashes':witnesses,
            'strict_acceptance':not findings,'result':'pass' if not findings else 'fail',
            'scope':'Read-only receipt, live-producer path, artifact and final-frame audit. Not full arithmetic proof or performance measurement.'}
    result['receipt_root_sha256']=canonical(result);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'frames_match':result['both_frames_match_rtx'],'strict_acceptance':result['strict_acceptance'],'findings':findings,'root':result['receipt_root_sha256']},indent=2));return bool(findings)
if __name__=='__main__':raise SystemExit(main())
