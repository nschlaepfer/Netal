#!/usr/bin/env python3
"""Verify the complete compact Block-1 implementation and rounded RCP domain."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def sha(data):return hashlib.sha256(data).hexdigest()
def file_sha(path):return sha(path.read_bytes())
def require(test,message):
    if not test:raise ValueError(message)


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),'preserve sealed verification')
    repo=Path(__file__).resolve().parents[1]
    reference=repo/'build/nr_swin_block/reference-v2'
    ref=json.loads((reference/'receipt.json').read_text())
    require(ref['cell_layout_bijective'] is True and ref['window_count']==8640,'reference layout')
    require(ref['original_block1_output_sha256']=='176b47ca5b7312c28717abf71a5d6b71c2333b382c09946cafc48794a52c76fe','RTX output provenance')
    require(file_sha(reference/'input.e4m3')==ref['flat_input_sha256'] and file_sha(reference/'output.e4m3')==ref['flat_output_sha256'],'full reference drift')
    target_full=(reference/'output.e4m3').read_bytes()
    rtx=repo/'build/nr_rcp_half/rtx-v1';mac=repo/'build/nr_rcp_half/macos-v1'
    rcp_job=json.loads((rtx/'receipt.json').read_text(encoding='utf-8-sig'))
    require(rcp_job['source_sha256']==file_sha(repo/'build/nr_rcp_half/source-v1/rsq_half_oracle.cu'),'RCP source drift')
    for entry in rcp_job['artifacts']:require(file_sha(rtx/'output'/entry['name'])==entry['sha256'],'RCP transfer drift')
    mathematical=bytearray()
    for bits in range(1,0x7c00):
        x=struct.unpack('<e',struct.pack('<H',bits))[0]
        try:mathematical.extend(struct.pack('<e',1/x))
        except OverflowError:mathematical.extend(struct.pack('<H',0x7c00))
    rtx_pids=set();rcp_mac_pids=set();rcp_hashes=set()
    for poison in ('a5','5a'):
        for ordinal in ('a','b'):
            stem=f'{poison}-{ordinal}'
            rr=json.loads((rtx/'output'/(stem+'.json')).read_text());mr=json.loads((mac/stem/'run.json').read_text())
            require(rr['schema']=='neuralstudio.nr.rcp-half-rtx.v1' and mr['schema']=='neuralstudio.nr.rcp-half-metal.v1','RCP arithmetic identity')
            require(rr['instruction']=='rcp.approx.ftz.f32 + cvt.rn.f16.f32','wrong CUDA instruction')
            require(rr['pid'] not in rtx_pids and mr['pid'] not in rcp_mac_pids,'RCP process reuse')
            rtx_pids.add(rr['pid']);rcp_mac_pids.add(mr['pid'])
            for r in (rr,mr):require(r['result']=='pass' and r['errors']==0 and r['count']==31743 and r['poison']==int(poison,16) and r['expected_input_opened'] is False,'RCP run contract')
            rb=(rtx/'output'/(stem+'.bin')).read_bytes();mb=(mac/stem/'output.bin').read_bytes()
            require(rb==mb==mathematical,'RCP output mismatch');rcp_hashes.add(sha(rb))
    require(len(rcp_hashes)==1,'RCP poison dependence')
    refs={};names=['expansion.f16','activation.e4m3','partial.f16','qkv.f16','normalized.e4m3','scores.f16','probabilities.e4m3','value_partial.f16','projection.f16','output.e4m3']
    for cta in range(9):
        paths={name:repo/f'build/nr_swin_mlp/fixtures/cta{cta}/comparison'/name for name in names[:3]}
        paths['qkv.f16']=repo/f'build/nr_swin_qkv/reference-v1/cta{cta}/qkv.f16'
        for name in names[4:6]:paths[name]=repo/f'build/nr_swin_attention/reference-v1/cta{cta}'/name
        for name in names[6:9]:paths[name]=repo/f'build/nr_swin_values/reference-v1/cta{cta}'/name
        paths['output.e4m3']=reference/f'cta{cta}/output.e4m3'
        refs[cta]={name:path.read_bytes() for name,path in paths.items()}
    pids=set();anchor_compared=0;full_compared=0;full_anchor_compared=0;runs=[]
    def check(run,rows,poison,input_sha):
        r=json.loads((run/'run.json').read_text())
        require(r['pid'] not in pids,'native process reuse');pids.add(r['pid'])
        require(r['result']=='pass' and r['rows']==rows and r['poison']==int(poison,16),'native shape/poison')
        require(r['dispatch_count']==10 and r['expected_file_opened'] is False,'native dispatch/ingress')
        require(r['coverage_errors']==r['canary_errors']==r['immutable_errors']==0,'native guards')
        require(r['input_sha256']==input_sha and r['weight_sha256']=='0270fb8f406d3d0a916953ddaf6ec5d29328fe95b9d6cb408ca0c25c91108e3a','native inputs')
        return r
    for cta in range(9):
        for poison in ('a5','5a'):
            for ordinal in ('a','b'):
                run=a.attempt/f'cta{cta}-{poison}-{ordinal}'
                r=check(run,64,poison,ref['ctas'][cta]['input_sha256'])
                for name,data in refs[cta].items():
                    require((run/name).read_bytes()==data and r['outputs'][name]==sha(data),'anchor mismatch: '+name)
                    anchor_compared+=len(data)
                runs.append({'kind':'anchor','cta':cta,'pid':r['pid'],'poison':r['poison'],'receipt_sha256':file_sha(run/'run.json')})
    for poison in ('a5','5a'):
        for ordinal in ('a','b'):
            run=a.attempt/f'full-{poison}-{ordinal}';r=check(run,552960,poison,ref['flat_input_sha256'])
            require((run/'output.e4m3').read_bytes()==target_full and r['outputs']['output.e4m3']==ref['flat_output_sha256'],'full-grid mismatch')
            full_compared+=len(target_full)
            for name in names:
                path=run/name;require(file_sha(path)==r['outputs'][name],'full artifact drift: '+name)
                with path.open('rb') as f:
                    for cta,rc in enumerate(ref['ctas']):
                        x,y,_=rc['cta'];base_row=(y*120+x)*64
                        chunks=4 if name=='partial.f16' else 2 if name=='value_partial.f16' else 1
                        row_bytes=len(refs[cta][name])//(chunks*64)
                        gathered=bytearray()
                        for chunk in range(chunks):
                            f.seek((chunk*552960+base_row)*row_bytes);gathered.extend(f.read(64*row_bytes))
                        require(gathered==refs[cta][name],'full-grid sampled intermediate mismatch: '+name)
                        full_anchor_compared+=len(gathered)
            runs.append({'kind':'full-grid','pid':r['pid'],'poison':r['poison'],'receipt_sha256':file_sha(run/'run.json'),'output_sha256':r['outputs']['output.e4m3']})
    sources=['src/nr_swin/qmma_fixed.metalh','src/nr_swin/swin32_mlp.metal','src/nr_swin/swin32_qkv.metal','src/nr_swin/swin32_attention.metal','src/nr_swin/swin32_block.metal',
             'src/nr_swin/mlp_probe.mm','src/nr_swin/rcp_half.metal','scripts/generate-nr-swin-block-probe.py',
             'scripts/build-nr-block1-buffer-reference.py','scripts/analyze-nr-swin-values.py','scripts/generate-nr-rcp-half-oracle.py','scripts/verify-nr-swin-block.py']
    result={'schema':'neuralstudio.nr.swin32-complete-block-verification.v1',
            'reference_root_sha256':ref['receipt_root_sha256'],'fresh_metal_processes':len(pids),
            'anchor_processes':36,'full_grid_processes':4,'full_grid_windows':8640,'full_grid_rows':552960,
            'anchor_compared_bytes':anchor_compared,'full_grid_compared_bytes':full_compared,
            'full_grid_sampled_intermediate_compared_bytes':full_anchor_compared,'mismatched_bytes':0,
            'original_rtx_output_sha256':ref['original_block1_output_sha256'],'logical_output_sha256':ref['flat_output_sha256'],
            'write_coverage_exactly_once':True,'canaries_intact':True,'inputs_immutable':True,'runs':runs,
            'rcp_domain':{'inputs':31743,'rtx_processes':4,'metal_processes':4,'gpu_driver':rcp_job['gpu'],
                          'output_sha256':next(iter(rcp_hashes)),'mathematically_rounded_half_equal':True,'mismatched_bytes':0},
            'source_sha256':{name:file_sha(repo/name) for name in sources},
            'generated_host_sha256':file_sha(a.attempt/'probe.mm'),'host_sha256':file_sha(a.attempt/'probe'),
            'metallib_sha256':file_sha(a.attempt/'block.metallib'),
            'claims':{'complete_block1_on_full_grid':True,'isolated_recorded_block0_input':True,
                      'native_block0_to_block1_live_handoff':False,'other_weights_and_swin_variants':False,
                      'full_model_inference':False},'result':'pass'}
    result['receipt_root_sha256']=sha(json.dumps(result,sort_keys=True,separators=(',',':')).encode())
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('receipt_root_sha256='+result['receipt_root_sha256'])
    print('full_grid_compared_bytes='+str(full_compared)+' mismatched_bytes=0')


if __name__=='__main__':main()
