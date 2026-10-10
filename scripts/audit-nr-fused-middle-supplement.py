#!/usr/bin/env python3
"""Check v9's actual fused middle against its same-attempt strict scalar chain.

This is a supplemental equivalence audit, not a claim that the strict decoder
consumed the fused output. Keep missing historical library bindings explicit.
"""
import argparse
import hashlib
import json
from pathlib import Path

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def require(ok,label):
    if not ok:raise ValueError(label)
def read(p):
    r=json.loads(p.read_text())
    if 'receipt_root_sha256' in r:
        b=dict(r);h=b.pop('receipt_root_sha256');require(canonical(b)==h,'seal '+str(p))
    return r
def option(command,name):
    require(command.count(name)==1,'unique argument '+name);return command[command.index(name)+1]
def audit(root,peer,base_path):
    root=root.resolve();peer=peer.resolve();chain=read(root/'full-chain.json');base=read(base_path)
    require(base['result']=='pass' and base['strict_acceptance'] and not base['findings'] and Path(base['attempt']).resolve()==root and base['chain_receipt_sha256']==sha(root/'full-chain.json'),'matching independent base strict audit')
    names=('a5-a','a5-b','5a-a','5a-b');steps={s['label']:s for s in chain['steps']};require(len(steps)==len(chain['steps']),'unique steps')
    enc=Path(chain['encoder_run']['path']).resolve();ev=read(enc/'verification.json');pids={r[k] for r in ev['runs'] for k in ('block0_pid','layout_pid','encoder_pid')};require(len(pids)==12,'producer process matrix')
    observed_libs={};rows=[];gaps=[]
    def resolve(p):
        path=Path(p);return path.resolve() if path.is_absolute() else (peer/path).resolve()
    for key in names:
        strict=root/f'middle-{key}'/key;fused=root/f'middle-fused-{key}';meta=chain['stages'][key]['middle_fused'];step=steps['middle-fused-'+key];r=read(fused/'resident.json');plan=read(fused/'chain.json');native=read(fused/'run/chain-run.json');entry=enc/key/'encoder/block22-middle.e4m3'
        require(step['exit_code']==0 and r['result']==meta['result']=='pass' and not r['failures'] and not meta['failures'],'fused execution result')
        require(sha(fused/'resident.json')==meta['resident_json_sha256'],'sealed fused receipt')
        require(resolve(option(step['command'],'--attempt'))==fused and resolve(option(step['command'],'--reference-middle'))==strict and resolve(option(step['command'],'--encoder-entry'))==entry,'same-attempt command paths')
        require(resolve(plan['entry_file'])==entry and sha(entry)==r['encoder_entry_sha256'],'live producer entry')
        require(plan['entry_offset']==86649856 and plan['final_offset']==123145216 and plan['pool_bytes']==212519936,'pool layout')
        require(native['pid'] not in pids and native['poison']==int(key[:2],16),'fresh fused process and poison');pids.add(native['pid'])
        split=resolve(meta['libraries']['split_attempt']);require(sha(split/'middle_chain_probe')==meta['probe_sha256'],'sealed fused executable')
        expected_libs={'split_lib':split/'split.metallib','vit_lib':resolve(option(step['command'],'--vit-attempt'))/'vit.metallib','vit_tp_lib':resolve(meta['libraries']['vit_tp_lib']),'split_tp_lib':resolve(meta['libraries']['split_tp_lib'])}
        for role,path in expected_libs.items():
            require(resolve(plan[role])==path,'selected library path '+role);h=sha(path)
            require(role not in observed_libs or observed_libs[role]['sha256']==h,'library drift across audit');observed_libs[role]={'path':str(path),'sha256':h}
        if 'selected_libraries' in meta:
            selected=meta['selected_libraries'];strict_split=resolve(option(steps['middle-'+key]['command'],'--split-attempt'))
            expected_roles={'fused_split_lib':expected_libs['split_lib'],'vit_lib':expected_libs['vit_lib'],'vit_tp_lib':expected_libs['vit_tp_lib'],'split_tp_lib':expected_libs['split_tp_lib'],'strict_split_lib':strict_split/'split.metallib'}
            require(set(selected)==set(expected_roles),'unique complete library-role seal')
            for role,path in expected_roles.items():require(resolve(selected[role]['path'])==path and selected[role]['sha256']==sha(path),'sealed selected library '+role)
            probes=meta['selected_probes'];vit=expected_libs['vit_lib'].parent
            expected_probes={'fused_middle_chain_probe':split/'middle_chain_probe','strict_split_probe':strict_split/'split_probe','vit_probe':vit/'vit_probe','repack_probe':vit/'repack_probe'}
            require(set(probes)==set(expected_probes),'complete selected probe seal')
            for role,path in expected_probes.items():require(resolve(probes[role]['path'])==path and probes[role]['sha256']==sha(path),'sealed selected probe '+role)
        else:
            recorded=meta['lib_sha256'];require(recorded.get('tp.metallib')==observed_libs['split_tp_lib']['sha256'],'persisted split TP digest')
            gaps.append({'key':key,'issue':'basename collision and incomplete library inventory','missing_unique_roles':['split_lib','vit_lib','vit_tp_lib']})
        expected=[(n,None) for n in range(25,58)]+[(58,'nr_vit_repack_2d_to_1d')]+[(n,'vit_block') for n in range(31,39)]+[(99,'nr_vit_repack_1d_to_2d')]+[(n,None) for n in range(100,133)]
        require(len(native['launches'])==len(expected)==76,'complete middle launch inventory');checks=0
        for L,(n,kind) in zip(native['launches'],expected):
            require(L['launch']==n and (kind is None or L['kind']==kind),'middle launch order')
            if kind=='vit_block':folder=strict/f'block{n}';rp=folder/'run.json';rr=read(rp);outputs=rr['stages']
            elif kind:folder=strict/f'repack{n}';rp=folder/'run.json';rr=read(rp);outputs={kind:rr['kernels'][kind]}
            else:folder=strict/f'launch-{n:03d}';rp=folder/'run.json';rr=read(rp);outputs=rr['outputs']
            require(sha(rp)==base['observed_receipt_hashes'][str(rp)],'base audited scalar receipt')
            require(set(L['outputs'])==set(outputs),'stage output inventory')
            for role,m in outputs.items():
                target=folder/(role+'.bin')
                if kind and kind!='vit_block':
                    # Repack receipts have no output digest. Bind their actual
                    # output to the independently sealed next consumer instead.
                    consumer=strict/('block31/run.json' if n==58 else 'launch-100/run.json');cr=read(consumer)
                    require(sha(consumer)==base['observed_receipt_hashes'][str(consumer)],'audited repack consumer')
                    if n==58:expected_hash=cr['inputs_sha256']['tokens']
                    else:
                        matches=[cr['inputs_sha256'][r] for r,p in cr['input_sources'].items() if resolve(p)==target]
                        require(len(matches)==1,'repack99 live consumer binding');expected_hash=matches[0]
                else:expected_hash=m['sha256']
                require(L['outputs'][role]==expected_hash==sha(target),'actual same-attempt stage equality');checks+=1
                if kind=='vit_block':require(L['coverage'][role]==[m['bytes'],0,0],'ViT write coverage')
                elif kind:require(L['coverage_dwords']==[163840,0,0],'repack write coverage')
                else:
                    valid=m.get('compared_bytes',m['bytes']);require(L['coverage'][role]==[valid,0,m['bytes']-valid],'split write coverage')
        require(checks==125==r['checks']==meta['checks'],'all 125 stage hashes')
        final=sha(fused/'run/out_view_132.bin');require(final==r['final_out_view_sha256']==sha(strict/'launch-132/out_view.bin')==meta['final_out_view_sha256'],'actual final middle view')
        body_inputs=read(root/f'body-{key}/body.json')['inputs'];source=next(s['file'] for s in body_inputs if s['pool_offset']==123145216)
        require(resolve(source)==strict/'launch-132/out_view.bin','identify which middle actually feeds the strict decoder')
        rows.append({'key':key,'fused_pid':native['pid'],'matched_stage_hashes':checks,'final_view_sha256':final,'native_receipt_sha256':sha(fused/'run/chain-run.json'),'resident_receipt_sha256':sha(fused/'resident.json'),'strict_decoder_ingress':'same-attempt scalar middle, equal to the separately executed fused view'})
    return {'schema':'neuralstudio.nr.fused-middle-supplement-audit.v1','result':'stage_equality_pass_provenance_incomplete' if gaps else 'pass','strict_base_audit_root':base['receipt_root_sha256'],'chain_root':chain['receipt_root_sha256'],'runs':rows,'matched_stage_hashes_total':500,'current_library_hashes_observed_not_historical_seals':observed_libs,'provenance_gaps':gaps,'full_production_strict_promotion':not gaps,'full_goal_complete':False,'scope':'Four fresh fused middle executions use the same-attempt encoder entries; all125 recorded stage hashes and actual final views match independently audited strict scalar outputs per key. The strict decoder consumes scalar output. Direct full in-memory execution is separately evidenced.','verifier_sha256':sha(Path(__file__))}
def main():
    p=argparse.ArgumentParser();p.add_argument('attempt',type=Path);p.add_argument('--peer-root',required=True,type=Path);p.add_argument('--base-audit',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();require(not a.output.exists(),'preserve audit')
    try:r=audit(a.attempt,a.peer_root,a.base_audit)
    except Exception as e:r={'schema':'neuralstudio.nr.fused-middle-supplement-audit.v1','result':'fail','error':str(e),'full_production_strict_promotion':False}
    r['receipt_root_sha256']=canonical(r);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
    if r['result']=='fail':raise SystemExit(3)
if __name__=='__main__':main()
