#!/usr/bin/env python3
"""Independent static review of the pinned peer v2 candidate; no GPU execution."""
import argparse,hashlib,json,re
from pathlib import Path

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def need(v,m):
    if not v:raise ValueError(m)
def audit(peer):
    pins={};checks=[];findings=[]
    def observe(p):
        h=sha(p);need(str(p) not in pins or pins[str(p)]==h,'review-time drift');pins[str(p)]=h;return h
    for candidate,accepted,pin in [('singlewarp-cand-v2','recovery-v23t','ff302c0ac482a4c5a6db6309af390ab998cbb167051c293ff5884b0b5d9fee62'),('singlewarp-cand-reset-v2','recovery-v17','7689c89163026d3994aac432c7050b3141de992210edc0960decab5c0f11f55f')]:
        c=peer/'build/nr_ptx'/candidate;a=peer/'build/nr_ptx'/accepted
        need(observe(c/'post.metallib')==pin,'candidate library pin')
        manifest=json.loads((c/'build-receipt.json').read_text());observe(c/'build-receipt.json')
        need(manifest['identity_drift_during_build']==[] and all(x['exit']==0 for x in manifest['commands']),'build commands and identity drift')
        for n,h in manifest['files'].items():need(observe(c/n)==h,'compiled output identity')
        for n,h in manifest['gen'].items():need(observe(c/'gen'/n)==h,'generated source identity')
        flagged=[x['argv'] for x in manifest['commands'] if '--single-warp' in x['argv']]
        need(len(flagged)==1 and flagged[0][flagged[0].index('--entry')+1]=='cc_tinlayout_fused_post_block_swin_1h_32_fp8','only selected post kernel specialized')
        for name in ('nr_ptx_rt.metal','nr_tex_sample.metal'):
            need(observe(c/'gen'/name)==observe(a/'gen'/name),'unchanged runtime/sampler include')
        name='ptx_post_block_swin_1h_32.metal';observe(a/'gen'/name)
        old=(a/'gen'/name).read_text();new=(c/'gen'/name).read_text()
        oldbody=old[old.index('    pc_next = pc + 1u;'):]
        normalized=oldbody.replace('atomic_fetch_add_explicit(&exit_count, 1u, memory_order_relaxed); done = true; break;','done = true; break;').replace('default: done = true; atomic_fetch_add_explicit(&exit_count, 1u, memory_order_relaxed); break;','default: done = true; break;')
        need(normalized==new[new.index('    pc_next = pc + 1u;'):],'complete case/state body preservation')
        code=re.sub(r'//[^\n]*','',new)
        need(code.count('simdgroup_barrier(')==3 and code.count('threadgroup_barrier(')==1,'three loop fences plus prologue')
        need('if (tlin < 1u) nr_tp_scratch_arr[tlin].enabled = (nthreads <= 32u * 1u);' in code,'scratch initialization executable')
        checks.append({'candidate':candidate,'case_state_body_equal':True,'runtime_sampler_equal':True,'generated_pins_match':True,'scratch_init_present':True})
    for name,exe in [('singlewarp-cand-v2','c015bfef6d25bdbab6b54ded972658c1883fc18a1cb9a81b9b20eb2569ee0647'),('singlewarp-cand-halffma-v2','7b1976fd00cf6d4b23952f7399d8659cd01060345a53fed4a5f76ad031578c21')]:
        b=peer/'build/nr_full'/name;record=(b/'build-sha256.txt').read_text();observe(b/'build-sha256.txt')
        for line in record.splitlines():
            h,p=line.split(None,1);p=Path(p);p=p if p.is_absolute() else peer/p
            need(observe(p)==h,'declared harness source/object/executable pin')
        need(observe(b/'nr_full_frame')==exe,'reviewed harness executable')
        header=b/'source/src/nr_temporal/nr_single_warp_guard.h'
        need(observe(header)=='cb281a804196e77da6cfe1c69bb208fc3bf1f4d38a49b0ebe9f0c76705d3d522','guard header bytes')
        if 'nr_single_warp_guard.h' not in record:findings.append({'build':name,'finding':'Guard header snapshot exists but is absent from build-sha256.txt.'})
        body=b/'source/src/nr_body/nr_body_runtime_v17.mm';source=body.read_text()
        normalized=re.sub(r'#ifdef NR_BODY_SINGLE_WARP\n.*?#endif\n','',source,flags=re.S)
        need(hashlib.sha256(normalized.encode()).hexdigest()=='64bbe143ffc11373ddc0d796a86d6adf45104113be52022bdc50e9e647dccc1a','default body unchanged after removing opt-in blocks')
        guard=source.index('require(swok,')
        begin=source.rfind('id<MTLComputeCommandEncoder> enc = compute();',0,guard)
        end=source.index('[enc endEncoding]',guard)
        need(begin>=0 and begin<guard<end,'guard ordering recognized')
        findings.append({'build':name,'finding':'Guard refusal throws while a compute encoder is open, before endEncoding; actual failure cleanup remains unverified. Move the guard before opening that encoder or provide exception-safe closure.'})
    test=peer/'scripts/test-nr-single-warp.py';observe(test)
    if 'shutil.rmtree(tmp' in test.read_text():findings.append({'finding':'Test rerun deletes its fixed /tmp directory, losing previous counterexample files; use a fresh retained directory per run.'})
    for path,h in pins.items():need(sha(path)==h,'review-end drift')
    return {'schema':'neuralstudio.nr.single-warp-v2-independent-review.v1','result':'source_checks_pass_live_gate_needs_repairs','checks':checks,'findings':findings,'pins':pins,'GPU_executed':False,'live_parity_accepted':False,'speedup_accepted':False,'default_promoted':False,'limits':['Static source/build review only; the guard error path has not been executed and no crash is claimed.','These findings do not establish a numerical mismatch in a valid candidate execution.']}
def main():
    p=argparse.ArgumentParser();p.add_argument('--peer',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();need(not a.out.exists(),'preserve prior review')
    r=audit(a.peer.resolve());r['reviewer_sha256']=sha(__file__);r['receipt_root_sha256']=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='pins'},indent=2))
if __name__=='__main__':main()
