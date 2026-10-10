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
def audit(directory):
    repo=Path(__file__).resolve().parents[1]
    labels=['motion1-reset100-a5-a','motion1-reset100-a5-b','motion1-reset100-5a-a','motion1-reset100-5a-b']
    prior=sealed(repo/'build/nr_encoder_fscale/live-v1/verification.json');native=prior['runs'][0]
    first_ports={Path(n).stem:m['sha256'] for n,m in native['native_exports'].items()};first_ports['block0-skip']=native['block0_skip_sha256']
    first_pre={'block0-skip':first_ports['block0-skip'],'block0-encoder':sha(repo/'build/nr_encoder_fscale/live-v1/a5-a/block0/encoder.e4m3')}
    unobserved=sealed(repo/'build/nr_reset_sequence/pilot-reset10-a5-parts-v3/capture/verification.json');targets=unobserved['outputs']
    for n,h in zip(('output.rgba16le','frame-1-output.rgba16le'),targets):assert sha(repo/'build/nr_reset_sequence/pilot-reset10-a5-parts-v3/capture'/n)==h
    targets=sealed(directory/labels[0]/(labels[0]+'-reference.json'))['outputs'];assert targets[0]==unobserved['outputs'][0]
    zero=sealed(directory/'zero-control100-a5-reference.json');base=sealed(repo/'build/nr_three_frame_history/matrix-audit-v1.json')
    assert zero['motion_case']==0 and zero['outputs']==base['outputs'] and [b['ports'] for b in zero['boundaries']]==base['boundary_ports']
    control=sealed(directory/'motion1-control111-a5-reference.json');assert control['resets']==[1,1,1] and control['outputs']==[targets[0]]*3
    pids=set();rows=[];ports=None;samplers=None
    for label,poison in zip(labels,('a5','a5','5a','5a')):
        d=directory/label;r=sealed(d/(label+'-reference.json'));cap=read(d/'capture-receipt.json');inv=read(d/'artifact-inventory.json');files={m['path']:m for m in inv['files']}
        assert r['result']=='pass_bounded_history_observation' and r['label']==label and r['resets']==[1,0,0] and r['poison']==poison
        assert r['motion_case']==1 and r['motion_sha256']=='44150d0bee7989b32461d371c352a325ef7f615e43fe4bbff8ecc5bbc629adce'
        assert r['source_root']=='c7afbe92e4905160a6ae015deae7f64dd8b43f4000c36c2b2bed4532c3627026'
        assert r['auditor_sha256']==sha(repo/'scripts/build-nr-motion-reference.py') and r['parser_sha256']==sha(repo/'scripts/build-rtx-oracle-manifest.py')
        assert r['all_inventory_payloads_rehashed_on_rig'] and r['gpu_executed_by_this_auditor'] is False
        assert sha(d/'capture-receipt.json')==r['capture_receipt_sha256'] and sha(d/'artifact-inventory.json')==r['inventory_sha256']==cap['artifact_inventory_sha256']
        assert sha(d/'host-output.txt')==files['host-output.txt']['sha256'];host=(d/'host-output.txt').read_text(encoding='utf-8-sig').splitlines()
        def values(n):return [s.split('=',1)[1] for s in host if s.startswith('runtime.'+n+'=')]
        assert values('process_id')==[str(r['pid'])] and r['pid']>0 and r['pid'] not in pids;pids.add(r['pid'])
        assert values('output_poison')==[poison] and values('color_case')==['0'] and values('motion_case')==['1'] and values('sequence_frames')==['3']
        assert values('sequence_frame')==['0 reset=1','1 reset=0','2 reset=0'] and values('evaluate_result')==['0x1','0x1','0x1']
        assert files['input-color.rgba16le']['sha256']=='81ec1de5d1e76e6b55c4fd987b3732180c9720919fbd760dd5b4e8aa8cdd0342'
        assert files['input-motion.rg16le']['sha256']==r['motion_sha256']
        for prefix in ('','frame-1-','frame-2-'):
            for before,after in [('input-color.rgba16le','input-color-after.rgba16le'),('input-depth.r32le','input-depth-after.r32le'),('input-motion.rg16le','input-motion-after.rg16le')]:assert files[before]['sha256']==files[prefix+after]['sha256']
        assert cap['runtime_file_sha256']=='e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e' and cap['analysis_runtime'] is None
        assert cap['counts']['launches']==468 and cap['counts']['snapshots']==6 and cap['snapshot_launch_ids']==[2,24,158,180,314,336]
        assert r['graph_roots']==['0e7a4f6ba72bde69ec42eea4968ebee4d359f551d2961dd03945f7a7c25a2d61']*3
        assert r['outputs']==targets and [files[n]['sha256'] for n in ('output.rgba16le','frame-1-output.rgba16le','frame-2-output.rgba16le')]==targets
        history=r['history_sha256'];assert history['history-after.rgba16le']==history['frame-1-history-before.rgba16le']==targets[0] and history['frame-1-history-after.rgba16le']==history['frame-2-history-before.rgba16le']==targets[1] and history['frame-2-history-after.rgba16le']==targets[2]
        for n,h in history.items():assert files[n]['bytes']==16588800 and files[n]['sha256']==h
        assert [b['launch'] for b in r['boundaries']]==[2,24,158,180,314,336] and [b['frame'] for b in r['boundaries']]==[0,0,1,1,2,2]
        current=[]
        for b in r['boundaries']:
            assert b['snapshot']['delta_sha256']==files[f'pool-snapshots/snapshot-{b["launch"]:04d}.delta']['sha256'];current.append(b['ports'])
        assert current[0]==first_pre and current[1]==first_ports,'first-call boundaries vs accepted encoder'
        assert current[2]['block0-skip']==current[3]['block0-skip'] and current[4]['block0-skip']==current[5]['block0-skip'] and current[2]!=current[0] and current[3]!=current[1]
        assert ports is None or current==ports,'four-process boundary invariance';ports=current
        assert samplers is None or r['samplers']==samplers;samplers=r['samplers']
        rows.append({'label':label,'pid':r['pid'],'poison':poison,'reference_root':r['receipt_root_sha256'],'capture_sha256':r['capture_receipt_sha256']})
    assert control['boundaries'][0]['ports']==control['boundaries'][2]['ports']==control['boundaries'][4]['ports']==first_pre
    assert control['boundaries'][1]['ports']==control['boundaries'][3]['ports']==control['boundaries'][5]['ports']==first_ports
    return {'schema':'neuralstudio.nr.motion-matrix-independent.v1','result':'pass_bounded_temporal_reference','runs':rows,'fresh_processes':4,'evaluations':12,'motion_case':1,'motion_sha256':'44150d0bee7989b32461d371c352a325ef7f615e43fe4bbff8ecc5bbc629adce','outputs':targets,'boundary_ports':ports,'samplers':samplers,'control_root':control['receipt_root_sha256'],'unobserved_pilot_root':unobserved['receipt_root_sha256'],'first_call_native_parent_root':prior['receipt_root_sha256'],'history_continuity_proven':True,'poison_invariance':True,'runtime_input_use_forbidden':True,'mac_temporal_parity':False,'full_goal_complete':False,'limits':['One original image, uniform motion(0.25,-0.5), fixed descriptors, three calls per context.','Raw pool/history payloads were rehashed and reconstructed on the rig; local audit binds the retained receipts/inventories.','No claim of new-input intermediate parity on Mac, moving-image history, disocclusion or arbitrary sequence length.'],'verifier_sha256':sha(Path(__file__))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--receipts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    try:r=audit(a.receipts.resolve())
    except Exception as e:r={'schema':'neuralstudio.nr.motion-matrix-independent.v1','result':'fail','error':str(e)}
    r['receipt_root_sha256']=canonical(r);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('samplers','boundary_ports')},indent=2))
    if r['result']=='fail':raise SystemExit(3)
if __name__=='__main__':main()
