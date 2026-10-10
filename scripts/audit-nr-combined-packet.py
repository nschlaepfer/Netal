#!/usr/bin/env python3
"""Recheck transported RTX receipts and actual combined-case comparison payloads."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sealed(p,root=None):
    r=read(p);b=dict(r);h=b.pop('receipt_root_sha256');assert canonical(b)==h and(root is None or h==root),'seal '+str(p);return r


def audit(packet):
    repo=Path(__file__).resolve().parents[1];d=packet/'capture'
    transport=sealed(packet/'transport-manifest.json','66da5deaef605376be08b0e4e290635322833fcb6b1f7285f60b9fd5e462c457')
    for n,m in transport['files'].items():assert (d/n).stat().st_size==m['bytes'] and sha(d/n)==m['sha256'],'transported payload '+n
    matrix=sealed(d/'matrix.json','45c6aedceb8a48eefd79961d37d657c3dfdeb973ca8fa9d91b2401ce6a3c5553')
    assert matrix['result']=='pass_bounded_combined_RTX_reference' and not matrix['mac_parity']
    source=sealed(d/'source-receipt.json','b1149eebbfd7971c7f7912b378a3a1f1f5da3450338f0d69c8bd1dc2dce3c1b1')
    for n,h in source['files'].items():assert sha(repo/'build/nr_combined_sequence/source-v1'/n)==h
    compiled=read(d/'build-receipt.json');assert compiled['exit_code']==0 and compiled['source_root']==source['receipt_root_sha256']
    assert compiled['files']['nvngx_dlssnr.dll']=='e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e'
    assert compiled['source_receipt_sha256']==sha(d/'source-receipt.json')
    assert matrix['auditor_sha256']==sha(repo/'scripts/build-nr-combined-reference.py')
    assert matrix['controller_sha256']==sha(repo/'scripts/run-nr-combined-matrix.py')
    expected_inputs=['81ec1de5d1e76e6b55c4fd987b3732180c9720919fbd760dd5b4e8aa8cdd0342','d2db0ead99a6307a6504afed2d49867f0421e3665bb397bed8277ae4f2046d06','af31801008df2fbb537f365265271e716b26339b4c5a2fa2069c01c1a949e8b8']
    assert matrix['input_hashes']==expected_inputs and matrix['motion_sha256']=='776bec76c8d7fe489da918b80f9178d135acecc88b977212e43c370bbb0cd3aa'
    assert [r['label'] for r in matrix['runs']]==['combined-a5-a','combined-a5-b','combined-5a-a','combined-5a-b']
    controls={
      'control-static-spatial-a5':repo/'build/nr_spatial_motion/receipts-v1/motion2-reset100-a5-a/motion2-reset100-a5-a-reference.json',
      'control-mixed-zero-a5':repo/'build/nr_image_sequence/receipts-v1/mixed-reset100-a5-a/mixed-reset100-a5-a-reference.json',
    }
    assert [r['label'] for r in matrix['controls']]==list(controls)
    pids=set();rows=[]
    for row in matrix['controls']+matrix['runs']:
        label=row['label'];ref=sealed(d/(label+'-reference.json'),row['receipt_root_sha256'])
        cap=read(d/(label+'-capture.json'));inv=read(d/(label+'-inventory.json'));run=read(d/(label+'-job.json'))
        assert run['exit_code']==0 and run['build_receipt_sha256']==sha(d/'build-receipt.json')
        assert sha(d/(label+'-capture.json'))==ref['capture_receipt_sha256']==run['capture_receipt_sha256']
        assert sha(d/(label+'-inventory.json'))==ref['inventory_sha256']==cap['artifact_inventory_sha256']
        assert cap['counts']['launches']==468 and cap['counts']['snapshots']==6 and cap['input_arrays_immutable'] and cap['rtx_inference_proven']
        assert cap['harness_sha256']==compiled['files']['color-cases.exe'] and cap['proxy_sha256']==compiled['files']['nvcuda.dll']
        assert ref['source_root']==compiled['source_root'] and ref['resets']==run['resets']==[1,0,0]
        assert ref['all_inventory_payloads_rehashed_on_rig'] and ref['runtime_input_use_forbidden'] and not ref['mac_temporal_parity']
        assert ref['graph_roots']==['0e7a4f6ba72bde69ec42eea4968ebee4d359f551d2961dd03945f7a7c25a2d61']*3
        assert ref['pid']>0 and ref['pid'] not in pids;pids.add(ref['pid'])
        files={x['path']:x for x in inv['files']};assert len(files)==len(inv['files'])
        weights=[(n,m['bytes'],m['sha256']) for n,m in sorted(files.items()) if n.startswith('cuda-dumps/transfer-')]
        assert len(weights)==153 and canonical(weights)=='ae6617321f5dba37b2ba46209e928c7d10ffaecfa8338a225ba450c6a6a4da9f'
        for i,prefix in enumerate(('','frame-1-','frame-2-')):
            assert files[prefix+'output.rgba16le']['sha256']==ref['outputs'][i]
            assert files[prefix+'history-after.rgba16le']['sha256']==ref['outputs'][i]
            if i:assert files[prefix+'history-before.rgba16le']['sha256']==ref['outputs'][i-1]
        if label in controls:
            parent=sealed(controls[label],row['control_parent_root'])
            assert ref['outputs']==parent['outputs'] and ref['history_sha256']==parent['history_sha256']
            assert [b['ports'] for b in ref['boundaries']]==[b['ports'] for b in parent['boundaries']]
        else:
            assert ref['pid']==row['pid'] and ref['poison']==row['poison']==label.split('-')[1]
            assert ref['image_sequence']=='mixed' and ref['color_cases']==[0,2,1] and ref['motion_case']==2
            for key in ('outputs','input_hashes','motion_sha256','history_sha256'):assert ref[key]==matrix[key]
            assert [b['ports'] for b in ref['boundaries']]==matrix['boundary_ports']
        rows.append({'label':label,'reference_root':ref['receipt_root_sha256'],'pid':ref['pid']})
    for i,prefix in enumerate(('','frame-1-','frame-2-')):
        for name,wanted in [('input-color.rgba16le',matrix['input_hashes'][i]),('output.rgba16le',matrix['outputs'][i])]:
            assert (d/(prefix+name)).stat().st_size==16588800 and sha(d/(prefix+name))==wanted
    assert (d/'input-motion.rg16le').stat().st_size==8294400 and sha(d/'input-motion.rg16le')==matrix['motion_sha256']
    return {'schema':'neuralstudio.nr.combined-packet-audit.v1','result':'pass_reference_packet_only','RTX_matrix_root':matrix['receipt_root_sha256'],
            'source_root':source['receipt_root_sha256'],'runs':rows,'outputs':matrix['outputs'],'input_hashes':matrix['input_hashes'],
            'motion_sha256':matrix['motion_sha256'],'boundary_ports':matrix['boundary_ports'],'payload_directory':str(d),
            'role':'Original inputs enter live native inference; expected outputs are CPU comparison-only after execution.',
            'full_model_Mac_parity':False,'production_promotion':False,'raw_RTX_pool_history_payloads_audited_on_rig_not_locally':True,
            'files':transport['files'],'auditor_sha256':sha(Path(__file__))}


def main():
    p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();r=audit(a.packet.resolve());r['receipt_root_sha256']=canonical(r);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k not in ('files','boundary_ports')},indent=2))


if __name__=='__main__':main()
