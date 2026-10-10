#!/usr/bin/env python3
"""Independently bind four history observations to sealed identities and controls."""
import argparse
import hashlib
import json
from pathlib import Path
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sealed(p):
    r=read(p);b=dict(r);h=b.pop('receipt_root_sha256');assert canonical(b)==h,'seal '+str(p);return r
def audit_original_contract(directory):
    repo=Path(__file__).resolve().parents[1]
    labels=['continuous-a5-a','continuous-a5-b','continuous-5a-a','continuous-5a-b']
    prior=sealed(repo/'build/nr_encoder_fscale/live-v1/verification.json');native=prior['runs'][0]
    first_ports={Path(n).stem:m['sha256'] for n,m in native['native_exports'].items()};first_ports['block0-skip']=native['block0_skip_sha256']
    first_pre={'block0-skip':first_ports['block0-skip'],'block0-encoder':sha(repo/'build/nr_encoder_fscale/live-v1/a5-a/block0/encoder.e4m3')}
    unobserved=sealed(repo/'build/nr_reset_sequence/pilot-reset10-a5-parts-v3/capture/verification.json');targets=unobserved['outputs']
    for n,h in zip(('output.rgba16le','frame-1-output.rgba16le'),targets):assert sha(repo/'build/nr_reset_sequence/pilot-reset10-a5-parts-v3/capture'/n)==h
    targets=sealed(directory/labels[0]/(labels[0]+'-reference.json'))['outputs'];base=sealed(repo/'build/nr_three_frame_history/matrix-audit-v1.json');assert targets[:3]==base['outputs'] and len(targets)==16
    control=sealed(directory/'control-all-reset-a5-reference.json');assert control['resets']==[1]*16 and control['outputs']==[targets[0]]*16
    pids=set();rows=[];ports=None;samplers=None
    for label,poison in zip(labels,('a5','a5','5a','5a')):
        d=directory/label;r=sealed(d/(label+'-reference.json'));cap=read(d/'capture-receipt.json');inv=read(d/'artifact-inventory.json');files={m['path']:m for m in inv['files']}
        assert r['result']=='pass_bounded_history_observation' and r['label']==label and r['resets']==[1]+[0]*15 and r['poison']==poison
        assert r['source_root']=='d1b151dc5a8a0101bbb9f9963fdf8d7bf1a3650ef4b21729acf67367cfcd723a'
        assert r['auditor_sha256']==sha(repo/'build/nr_sixteen/recovered-auditor-v1/build-nr-sixteen-reference.py') and r['parser_sha256']==sha(repo/'scripts/build-rtx-oracle-manifest.py')
        assert r['all_inventory_payloads_rehashed_on_rig'] and r['gpu_executed_by_this_auditor'] is False
        assert sha(d/'capture-receipt.json')==r['capture_receipt_sha256'] and sha(d/'artifact-inventory.json')==r['inventory_sha256']==cap['artifact_inventory_sha256']
        assert sha(d/'host-output.txt')==files['host-output.txt']['sha256'];host=(d/'host-output.txt').read_text(encoding='utf-8-sig').splitlines()
        def values(n):return [s.split('=',1)[1] for s in host if s.startswith('runtime.'+n+'=')]
        assert values('process_id')==[str(r['pid'])] and r['pid']>0 and r['pid'] not in pids;pids.add(r['pid'])
        assert values('output_poison')==[poison] and values('color_case')==['0'] and values('sequence_frames')==['16']
        assert values('sequence_frame')==[str(i)+' reset='+str(int(i==0)) for i in range(16)] and values('evaluate_result')==['0x1']*16
        assert files['input-color.rgba16le']['sha256']=='81ec1de5d1e76e6b55c4fd987b3732180c9720919fbd760dd5b4e8aa8cdd0342'
        for prefix in ['']+[f'frame-{i}-' for i in range(1,16)]:
            for before,after in [('input-color.rgba16le','input-color-after.rgba16le'),('input-depth.r32le','input-depth-after.r32le'),('input-motion.rg16le','input-motion-after.rg16le')]:assert files[before]['sha256']==files[prefix+after]['sha256']
        assert cap['runtime_file_sha256']=='e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e' and cap['analysis_runtime'] is None
        assert cap['counts']['launches']==2496 and cap['counts']['snapshots']==32 and cap['snapshot_launch_ids']==[156*i+j for i in range(16) for j in (2,24)]
        assert r['graph_roots']==['0e7a4f6ba72bde69ec42eea4968ebee4d359f551d2961dd03945f7a7c25a2d61']*16
        assert r['outputs']==targets and [files[n]['sha256'] for n in ['output.rgba16le']+[f'frame-{i}-output.rgba16le' for i in range(1,16)]]==targets
        history=r['history_sha256']
        for i,p in enumerate(['']+[f'frame-{i}-' for i in range(1,16)]):
            assert history[p+'history-after.rgba16le']==targets[i]
            if i:assert history[p+'history-before.rgba16le']==targets[i-1]
        for n,h in history.items():assert files[n]['bytes']==16588800 and files[n]['sha256']==h
        assert [b['launch'] for b in r['boundaries']]==[156*i+j for i in range(16) for j in (2,24)] and [b['frame'] for b in r['boundaries']]==[i for i in range(16) for _ in range(2)]
        current=[]
        for b in r['boundaries']:
            assert b['snapshot']['delta_sha256']==files[f'pool-snapshots/snapshot-{b["launch"]:04d}.delta']['sha256'];current.append(b['ports'])
        assert current[0]==first_pre and current[1]==first_ports,'first-call boundaries vs accepted encoder'
        assert current[2]['block0-skip']==current[3]['block0-skip'] and current[4]['block0-skip']==current[5]['block0-skip'] and current[2]!=current[0] and current[3]!=current[1]
        assert ports is None or current==ports,'four-process boundary invariance';ports=current
        assert samplers is None or r['samplers']==samplers;samplers=r['samplers']
        rows.append({'label':label,'pid':r['pid'],'poison':poison,'reference_root':r['receipt_root_sha256'],'capture_sha256':r['capture_receipt_sha256']})
    for i in range(16):
        assert control['boundaries'][2*i]['ports']==first_pre and control['boundaries'][2*i+1]['ports']==first_ports
    return {'schema':'neuralstudio.nr.sixteen-matrix-independent.v1','result':'pass_bounded_temporal_reference','runs':rows,'fresh_processes':4,'evaluations':64,'reset_mask':[1]+[0]*15,'outputs':targets,'boundary_ports':ports,'samplers':samplers,'control_root':control['receipt_root_sha256'],'unobserved_pilot_root':unobserved['receipt_root_sha256'],'first_call_native_parent_root':prior['receipt_root_sha256'],'history_continuity_proven':True,'poison_invariance':True,'runtime_input_use_forbidden':True,'mac_temporal_parity':False,'full_goal_complete':False,'limits':['One original image, zero motion and fixed descriptors, sixteen uninterrupted calls per context, reset only at frame0.','Raw pool/history payloads were rehashed and reconstructed on the rig; local audit binds the retained receipts/inventories.','No claim of new-input intermediate parity on Mac, moving-image history, disocclusion or arbitrary sequence length.'],'verifier_sha256':sha(Path(__file__))}
def audit(directory):
    repo=Path(__file__).resolve().parents[1]
    exporter=repo/'build/nr_sixteen/recovered-auditor-v1/build-nr-sixteen-reference.py'
    assert sha(exporter)=='d87a6d1f7e9cc64e85442243770d9597ff95e776e00f0f45a572ffcd2f5809ab','exact historical exporter source'
    r=audit_original_contract(directory)
    previous=sealed(repo/'build/nr_sixteen/matrix-audit-v1.json')
    assert previous['receipt_root_sha256']=='4feea5197385aecc8a9784f3ac955cd9362ec9f1f3b7ebfe2c6d7aaf55998576','historical matrix identity'
    ignored={'verifier_sha256','receipt_root_sha256'}
    assert {k:v for k,v in r.items() if k not in ignored}=={k:v for k,v in previous.items() if k not in ignored},'all historical matrix facts reproduced'
    r['recovery']={'historical_matrix_root':previous['receipt_root_sha256'],'exact_exporter_sha256':sha(exporter),
        'current_extended_exporter_sha256':sha(repo/'scripts/build-nr-sixteen-reference.py'),
        'previous_failed_revalidation_root':sealed(repo/'build/nr_sixteen/matrix-recovery-audit-v1.json')['receipt_root_sha256'],
        'new_RTX_execution':False,'remote_payloads_rehashed_now':False,
        'scope':'Revalidated retained local receipts, inventories, host logs, source pins and controls; prior remote payload rehash remains historical.'}
    return r

def main():
    p=argparse.ArgumentParser();p.add_argument('--receipts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    try:r=audit(a.receipts.resolve())
    except Exception as e:r={'schema':'neuralstudio.nr.sixteen-matrix-independent.v1','result':'fail','error':type(e).__name__+': '+str(e)}
    r['receipt_root_sha256']=canonical(r);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('samplers','boundary_ports')},indent=2))
    if r['result']=='fail':raise SystemExit(3)
if __name__=='__main__':main()
