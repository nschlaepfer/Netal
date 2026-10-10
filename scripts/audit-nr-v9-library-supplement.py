#!/usr/bin/env python3
"""Independent closure of the v9 library-binding supplement, not a reseal of v9."""
import argparse
import hashlib
import json
from pathlib import Path

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canon(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def check(ok,why):
    if not ok:raise ValueError(why)
def read(p):
    r=json.loads(p.read_text())
    if 'receipt_root_sha256' in r:
        b=dict(r);h=b.pop('receipt_root_sha256');check(canon(b)==h,'seal '+str(p))
    return r
def audit(peer,parent,supplement):
    repo=Path(__file__).resolve().parents[1];peer=peer.resolve();parent=parent.resolve();supplement=supplement.resolve();s=read(supplement/'summary.json')
    base=read(repo/'build/nr_full_chain_audit/attempt-v9-base-audit-v1.json');comparison=read(repo/'build/nr_full_chain_audit/attempt-v9-fused-audit-v2.json')
    check(base['strict_acceptance'] and comparison['matched_stage_hashes_total']==500 and comparison['strict_base_audit_root']==base['receipt_root_sha256'],'audited v9 parent comparison')
    check(sha(parent/'full-chain.json')==base['chain_receipt_sha256'],'parent immutable')
    check(s['result']=='pass' and len(s['runs'])==2 and {r['poison'] for r in s['runs']}=={'a5','5a'},'two fresh poison runs')
    build=peer/'build/nr_full/v4b'
    for line in (build/'build-sha256.txt').read_text().splitlines():
        h,path=line.split(None,1);check(sha(peer/path.strip())==h,'frozen supplemental build')
    rows=[];pids=set();repo_assets=repo/'build/nr_encoder_runtime/build-v1/assets';constant_cache={}
    for sr in s['runs']:
        poison=sr['poison'];key=poison+'-a';d=supplement/('original-'+poison);r=read(d/'receipt.json');f=r['frames'][0];b=f['body'];raw=d/'diag/frame-0/chain-run.json';native=read(raw);claim=read(d/'supplement.json')
        check(len(r['frames'])==1 and r['result']==f['result']=='pass' and r['pid']==native['pid']==sr['pid'] and r['pid'] not in pids,'fresh complete in-memory frame');pids.add(r['pid'])
        check(r['executable_sha256']==sr['executable_sha256']==sha(build/'nr_full_frame'),'actual executable')
        check(native['poison']==f['poison']==int(poison,16) and f['input_sha256']=='81ec1de5d1e76e6b55c4fd987b3732180c9720919fbd760dd5b4e8aa8cdd0342','original input/poison')
        check(sha(raw)==claim['chain_run_sha256']==sr['supplement']['chain_run_sha256'] and claim['receipt_root_sha256']==sr['supplement']['root'],'native receipt binding')
        expected_row=next(x for x in comparison['runs'] if x['key']==key);old_path=parent/f'middle-fused-{key}/run/chain-run.json';check(sha(old_path)==expected_row['native_receipt_sha256'],'previous independently audited stage data')
        old=read(old_path)['launches'];middle=[x for x in native['launches'] if x['kind']!='ptx'];check(len(old)==len(middle)==76,'all middle stages')
        count=0
        for got,want in zip(middle,old):
            check(got['launch']==want['launch'] and got['kind']==want['kind'] and got['outputs']==want['outputs'],'actual 125-hash equality');count+=len(got['outputs'])
            for field in ('coverage','coverage_dwords'):
                if field in want:check(got[field]==want[field],'actual middle coverage counters')
        check(count==125 and b['middle_view_sha256']==expected_row['final_view_sha256'],'full middle/view equality')
        check(b['code']==0 and b['checks_enforced']==241 and not b['check_failures'] and b['constants_rehashed_after_frame']==153 and b['constants_unchanged_after_frame'] is True,'enforced body and constant checks')
        check(f['encoder']['code']==0 and f['encoder']['guarded_outputs_checked']==1039 and f['encoder']['coverage_values_checked']==5900107776 and f['producer_regions_unchanged_by_body'] is True,'fresh producer checks')
        loaded=b['loaded'];libs={k[4:]:v for k,v in loaded.items() if k.startswith('lib:')};check(len(libs)==7 and libs==sr['libraries_loaded_full_path'],'seven uniquely named libraries')
        for role,expected in comparison['current_library_hashes_observed_not_historical_seals'].items():check(libs[expected['path']]==expected['sha256'],'separate library role '+role)
        check(r['files_unchanged_after_run'] is True,'post-run file gate')
        for name,meta in r['post_run_file_identity'].items():
            path=Path(name).resolve();check(path.is_relative_to(peer/'build') or path.is_relative_to(repo_assets),'file scope')
            check(meta['unchanged'] is True and meta.get('pre',meta.get('load'))==meta['post']==sha(path),'pre/load/post file identity')
        for path,h in libs.items():check(r['post_run_file_identity'][path]['post']==h,'all library identities post-bound')
        for name,h in loaded.items():
            if name.startswith('file:'):
                path=Path(name[5:].rsplit(':',1)[0]).resolve();check(path.is_relative_to(peer/'build'),'constant path scope')
                if path not in constant_cache:constant_cache[path]=sha(path)
                check(constant_cache[path]==h,'loaded model constant')
        final=d/'final-0.rgba16le';check(final.stat().st_size==16588800 and sha(final)==b['final_surface_sha256']==f['surface_sha256_caller_readback']=='a07835db4be8469dad1e3d3befcc495be396790f89408be62ddb8a5a710585a2','actual final frame')
        check(b['final_pixels']==2073600 and b['final_pixels_not_written']==b['final_pixels_written_more_than_once']==0 and b['input_texture_immutable'] is True,'final frame coverage/input checks')
        rows.append({'poison':poison,'pid':r['pid'],'matched_middle_hashes':count,'native_receipt_sha256':sha(raw),'frame_receipt_sha256':sha(d/'receipt.json'),'libraries':libs,'final_sha256':sha(final)})
    return {'schema':'neuralstudio.nr.v9-library-supplement-independent-audit.v1','result':'pass','base_strict_audit_root':base['receipt_root_sha256'],'fused_equality_audit_root':comparison['receipt_root_sha256'],'supplement_root':s['receipt_root_sha256'],'runs':rows,'supplemental_library_binding_closed':True,'original_v9_receipt_rewritten':False,'full_goal_complete':False,'scope':'Audited base scalar full chain plus four same-attempt fused-equivalence runs plus two fresh in-memory frames with 125 matching middle hashes each and unique load/post library identities. These are distinct compositional evidence; no temporal/general-input/performance claim. Separate middle output payloads were trimmed by the producer; retained execution-time hashes and actual final-frame files were checked.','verifier_sha256':sha(Path(__file__))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--peer-root',required=True,type=Path);p.add_argument('--parent',required=True,type=Path);p.add_argument('--supplement',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();check(not a.output.exists(),'preserve audit')
    try:r=audit(a.peer_root,a.parent,a.supplement)
    except Exception as e:r={'schema':'neuralstudio.nr.v9-library-supplement-independent-audit.v1','result':'fail','error':str(e),'supplemental_library_binding_closed':False}
    r['receipt_root_sha256']=canon(r);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'result':r['result'],'error':r.get('error'),'root':r['receipt_root_sha256']},indent=2))
    if r['result']!='pass':raise SystemExit(3)
if __name__=='__main__':main()
