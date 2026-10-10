#!/usr/bin/env python3
"""Strict live checked-pair comparison with explicit frame inventory and poison scope."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(p.read_text())
def need(v,msg):
    if not v:raise ValueError(msg)
def expected_keys(replicas,frames):
    need(replicas>0 and frames>0,'nonempty declared inventory required')
    return {(f'checked-{r}',f) for r in range(1,replicas+1) for f in range(frames)}

def paired_plan(baseline,candidate,frames,replicas,baseline_complement=None,candidate_complement=None):
    """Explicit reviewed patterns, never discovered from whatever files exist."""
    need(frames>0 and replicas==2,'reviewed base inventory requires exactly two replicas')
    need((baseline_complement is None)==(candidate_complement is None),'both complement sides required')
    alternating=[165 if i%2==0 else 90 for i in range(frames)]
    plans=[('checked-1',baseline,candidate,alternating),
           ('checked-2',baseline,candidate,[165]*frames)]
    if baseline_complement is not None:
        need(baseline_complement!=baseline and candidate_complement!=candidate,'complement must preserve existing base directories')
        plans.append(('checked-3-complement',baseline_complement,candidate_complement,[90 if x==165 else 165 for x in alternating]))
    return plans

def preflight_plan(plans,frames):
    """Fail missing/extra receipts before expensive binary hashing."""
    expected={}
    for tag,base,candidate,poisons in plans:
        need(len(poisons)==frames and set(poisons)<={165,90},'declared poison inventory')
        for folder in (base,candidate):
            expected.setdefault(folder,set()).add(tag)
            p=folder/tag/'receipt.json'
            need(p.is_file(),f'missing declared receipt: {p}')
            r=read(p)
            need(len(r.get('frames',[]))==r.get('frames_in_process')==frames,'complete frame inventory')
            for i,f in enumerate(r['frames']):
                need(f.get('frame')==i and f.get('poison')==poisons[i],'declared frame index/poison')
    for folder,tags in expected.items():
        actual={p.name for p in folder.glob('checked-*') if p.is_dir()}
        need(actual==tags,f'unexpected/missing replica directories: {folder}')


def audit(peer,baseline,candidate,name,replicas,baseline_complement=None,candidate_complement=None):
    repo=Path(__file__).resolve().parents[1];helper=repo/'scripts/audit-nr-half-fma-regression.py'
    need(sha(helper)=='1b002008ef0d6993bebd6a2ada9d763bcdcf7869497b1d3375e8d4605da251ff','reference provider pin')
    spec=importlib.util.spec_from_file_location('refs',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    rtx,n,resets,images,motion,fractional=m.configuration(repo,name)
    need(name in ('black3','uniform3'),'bounded reviewed pair configurations')
    plans=paired_plan(baseline,candidate,n,replicas,baseline_complement,candidate_complement)
    preflight_plan(plans,n)
    enc=repo/('build/nr_encoder_temporal/half-fma-spatial-v1' if fractional else 'build/nr_encoder_temporal/half-fma-v1')
    er=m.sealed(enc/'build.json');pins={};records={};pids=set()
    def observe(p):
        h=sha(p);need(str(p) not in pins or pins[str(p)]==h,'artifact changed during review');pins[str(p)]=h;return h
    def resolve(p):return Path(p).resolve() if Path(p).is_absolute() else (peer/p).resolve()
    for name_,digest in er['assets'].items():need(observe(enc/'assets'/name_)==digest,'encoder asset pin')
    mappings={'entry':'block22-middle','skip_c256':'block22-native','skip_c128':'block14-native','skip_c64':'block8-native','skip_c32':'block4-native','block0_skip':'block0-skip'}
    for side,folder,bname,version,body_sha in [('baseline',baseline,'v29t-v11-sp' if fractional else 'v29t-v11',11,'4a55d1d818f064e5d597ad9b1e9f69e3cc48321608e0f910e8504a4317020ec9'),('resident',candidate,'v29t-sp' if fractional else 'v29t',14,'51abea035b5add43744aefe793f051cadd1aa0181c79d657d495c1538fdb6b1f')]:
        build=peer/'build/nr_full'/bname
        for line in (build/'build-sha256.txt').read_text().splitlines():
            h,p=line.split(None,1);need(observe(resolve(p.strip()))==h,'build identity')
        need(observe(build/f'source/src/nr_body/nr_body_runtime_v{version}.mm')==body_sha,'reviewed body source')
        need(observe(build/'source/src/nr_body/nr_full_frame.mm')=='645552aab97f0c69ab60ab053f11f34fad904a83a64d09228bd71329e8c08c0c','reviewed checked-capable harness')
        need(observe(build/'source/encoder/nr_encoder_runtime.o')==er['object_sha256'],'accepted encoder object')
        need((build/'build-defines.txt').read_text().strip()=='defines: none','no test hooks')
        for tag,base_dir,candidate_dir,declared_poisons in plans:
            d=(base_dir if side=='baseline' else candidate_dir)/tag;p=d/'receipt.json';r=read(p);observe(p)
            need(r['schema']=='neuralstudio.nr.full-frame-inmemory.v11' and r['result']=='pass' and len(r['frames'])==r['frames_in_process']==n,'complete successful declared frame inventory')
            need(r['pid'] not in pids and r['context_preparations']==1 and r['contexts_prepared_once'],'fresh one-context process');pids.add(r['pid'])
            need(r['executable_sha256']==observe(build/'nr_full_frame') and resolve(r['command'][0])==build/'nr_full_frame','executed build')
            descriptor=resolve(r['command'][1]);need(descriptor==peer/('build/nr_body/desc-v28t/chain.json' if fractional else 'build/nr_body/desc-v27t/chain.json') and observe(descriptor)==r['body_descriptor_sha256'],'descriptor identity')
            desc=read(descriptor);old_desc=read(peer/('build/nr_body/desc-v24t/chain.json' if fractional else 'build/nr_body/desc-v23t/chain.json'))
            need({k:v for k,v in desc.items() if k not in ('resident_lib','resident_lib_sha256','note_v13')}==old_desc,'unchanged model descriptor')
            need(observe(resolve(desc['resident_lib']))==desc['resident_lib_sha256']=='f8dc47fc056254505ccf4a6783ae0bccca41c9a75ec078e02be1840649961351','validator library pin')
            need('--body-checked' in r['command'] and resolve(r['encoder_asset_root'])==enc/'assets','checked flag and encoder assets')
            need(r['command'][r['command'].index('--reset-sequence')+1]==','.join(map(str,resets)),'reset mask')
            need(observe(resolve(r['command'][r['command'].index('--motion')+1]))==motion,'motion identity')
            for i,f in enumerate(r['frames']):
                need(f['poison']==declared_poisons[i],'reviewed declared poison pattern')
                need(f['frame']==i and f['body_mode']=='checked' and f['status']=='comparison_passed' and f['result']=='pass','actual checked mode and success')
                e,b=f['encoder'],f['body'];need(e['code']==b['code']==0 and e['gpu_complete'] and e['input_immutable'] and not b['check_failures'],'runtime and guards')
                need(e['guarded_outputs_checked']==(1039 if resets[i] else 1040) and e['coverage_values_checked']==(5900107776 if resets[i] else 5935497216),'encoder coverage')
                need(f['input_sha256']==e['input_sha256']==observe(resolve(f['input_file']))==images[i],'live original input')
                need(f['temporal_state_advanced'] and f['frames_since_reset_after']==i+1 and f['history_copy_value_differences']==0,'history state and bit comparison')
                need(f['ex2_half_faults']=={'argument':0,'domain':0} and f['test_fault']=='','no hidden faults')
                need(b['final_pixels_not_written']==b['final_pixels_written_more_than_once']==0 and b['final_pixels']==2073600,'full output coverage')
                need(b['input_texture_immutable'] and f['producer_regions_unchanged_by_body'] and b['constants_unchanged_after_frame'],'immutability')
                need(b['checks_skipped']==[],'strong checked run has no omitted checks')
                for resource,digest in b['loaded'].items():
                    if resource.startswith('lib:'):path=resolve(resource[4:])
                    elif resource.startswith('file:'):path=resolve(resource[5:].rsplit(':',1)[0])
                    elif resource.startswith('ex2_half_table:'):path=resolve(resource.split(':',1)[1]);need(digest=='f9ae941e60868e72489db582fe725bdb401e033fce8694ecd35dd0c3329ba3cf','EX2 table pin')
                    else:raise ValueError('unknown loaded resource')
                    need(path.is_relative_to(peer/'build') and observe(path)==digest,'actual loaded model resource')
                for k,target in mappings.items():need(f['regions'][k]['sha256']==rtx['boundary_ports'][2*i+1][target],'RTX producer boundary')
                wanted=rtx['outputs'][i]
                need(b['history_sha256_after']==f['surface_sha256_caller_readback']==wanted,'final/history identity')
                if i:need(b['history_sha256_before']==rtx['outputs'][i-1],'this process prior output history')
                for p in [d/f'final-{i}.rgba16le',d/f'diag/frame-{i}/tex_out_155.bin',d/f'diag/frame-{i}/tex_out_156.bin',resolve(f['reference']['file'])]:need(p.stat().st_size==16588800 and observe(p)==wanted,'actual final/post/history/reference bytes')
                cp=d/f'diag/frame-{i}/chain-run.json';c=read(cp);observe(cp);need(c['pid']==r['pid'] and c['poison']==f['poison'],'same-process checkpoint receipt')
                records[side,tag,i]=(f,c)
            for p,h in r['post_run_file_identity'].items():need(h['unchanged'] and observe(resolve(p))==h['post']==h.get('pre',h.get('load')),'post-run file identity')
    rows=[];poisons={i:set() for i in range(n)}
    for tag,i in sorted((tag,i) for tag,_,_,_ in plans for i in range(n)):
        f0,c0=records['baseline',tag,i];f1,c1=records['resident',tag,i]
        need(f0['poison']==f1['poison'],'paired poison');poisons[i].add(f1['poison'])
        old={(x['launch'],k):v for x in c0['launches'] if x['kind']!='ptx' for k,v in x['outputs'].items()};need(len(old)==125,'all old middle identities')
        for x in c0['launches']:
            if x['kind']=='ptx':old[x['launch'],'pool']=x['pool_post_sha256']
        new={(x['launch'],k[:-7]):v for x in c1['launches'] for k,v in x['digests'].items() if k.endswith('_sha256')}
        need(len(old)==149 and old.keys()<=new.keys() and all(old[k]==new[k] for k in old),'all middle and decoder pool checkpoints, none missing')
        need(c1['decoder_pre_image_source']=='gpu-snapshot' and c1['decoder_pre_image_sha256']==c1['decoder_pre_image_expected_sha256']==c0['decoder']['pool_pre_sha256'],'actual GPU preimage identity')
        need(c1['pool_final_sha256']==c0['decoder']['pool_final_sha256'],'final pool identity')
        need(c1['tex155_output_sha256']==next(x for x in c0['launches'] if x['launch']==155)['texture']['output_sha256'],'155 texture identity')
        need(c1['command_buffers']==1 and c1['validators']=={'expected':150,'run':150,'faults':0},'single buffer and full resident validators')
        rows.append({'replica':tag,'frame':i,'poison':f1['poison'],'strong_identities':152,'final_sha256':rtx['outputs'][i]})
    need(len(rows)==len(plans)*n,'no vacuous or partial success')
    for p,h in pins.items():need(sha(Path(p))==h,'review-time identity')
    return {'schema':'neuralstudio.nr.live-checked-pair-independent.v2','result':'pass_declared_live_checked_subset','configuration':name,'baseline':str(baseline),'resident':str(candidate),'declared_runs':[{'tag':tag,'baseline':str(b),'resident':str(c),'poisons':ps} for tag,b,c,ps in plans],'reference_root':rtx['receipt_root_sha256'],'rows':rows,'strong_identities_compared':len(rows)*152,'poisons_by_frame':{str(i):sorted(v) for i,v in poisons.items()},'both_poisons_every_frame':all(v=={165,90} for v in poisons.values()),'resident_promoted':False,'speedup_accepted':False,'full_goal_complete':False,'pins':pins,'scope':'Only the exact declared paired replicas/frames. Opposite-poison coverage is reported separately; no missing-frame or missing-checkpoint pass.','verifier_sha256':sha(Path(__file__))}


def main():
    p=argparse.ArgumentParser();p.add_argument('--peer',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--resident',type=Path,required=True);p.add_argument('--configuration',choices=('black3','uniform3'),required=True);p.add_argument('--replicas',type=int,default=2);p.add_argument('--baseline-complement',type=Path);p.add_argument('--resident-complement',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();need(not a.output.exists(),'preserve audit')
    try:r=audit(a.peer.resolve(),a.baseline.resolve(),a.resident.resolve(),a.configuration,a.replicas,a.baseline_complement.resolve() if a.baseline_complement else None,a.resident_complement.resolve() if a.resident_complement else None)
    except Exception as e:r={'schema':'neuralstudio.nr.live-checked-pair-independent.v2','result':'fail','error':str(e),'verifier_sha256':sha(Path(__file__))}
    r['receipt_root_sha256']=canonical(r);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('rows','pins')},indent=2))
    if r['result']=='fail':raise SystemExit(3)


if __name__=='__main__':main()
