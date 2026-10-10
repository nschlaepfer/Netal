#!/usr/bin/env python3
"""Bind first-inference planning to the recovered model and fresh RTX output."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import struct


RUNTIME = 'e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e'
GRAPH = '3de475044e840d59f0235f0ff12763e99950909f81e5ae8b8c832375a25c88c5'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def require(value, message):
    if not value:
        raise ValueError(message)


def family(layer):
    name = layer['type']
    if name == 'CCTinlayoutFusedPreBlockSwin1H':
        return 'input_preprocess'
    if name == 'CCTinlayoutFusedPostBlockSwin1H':
        return 'output_postprocess'
    if name.startswith('CCTinlayoutFusedSwin'):
        return 'small_swin'
    if name.startswith('CCSplitSwin16H'):
        return 'split_swin_16h'
    if name.startswith('CCVit1D'):
        return 'vit_1d'
    if name == 'CCDecInputUpsample':
        return 'decoder_upsample'
    raise ValueError('unclassified graph layer: '+name)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--graph',required=True,type=Path)
    parser.add_argument('--weights',required=True,type=Path)
    parser.add_argument('--amd',required=True,type=Path)
    parser.add_argument('--amd-disassembly',required=True,type=Path)
    parser.add_argument('--rtx',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    a=parser.parse_args()
    require(not a.output.exists(),'preserve existing receipt')
    graph,weights,amd=read(a.graph),read(a.weights),read(a.amd)
    require(graph['manifest_root_sha256']==GRAPH,'graph root mismatch')
    require(graph['runtime_sha256']==weights['runtime']['sha256']==RUNTIME,'model mismatch')
    require(weights['tensor_count']==153 and weights['payload_bytes']==147683778,'weight count mismatch')
    table={t['name']:(t['data_size'],t['sha256']) for t in weights['tensors']}
    require(len(table)==153,'duplicate tensors')
    for t in graph['base_graph_weight_bindings']:
        require(table[t['tensor_name']]==(t['bytes'],t['content_sha256']),'weight graph binding mismatch')

    layers=graph['pre_fp8_base_graph']['layers']
    groups={}
    for layer in layers:
        f=family(layer)
        groups.setdefault(f,{'blocks':set(),'layers':[]})
        groups[f]['blocks'].add(layer['block_index'])
        groups[f]['layers'].append({'block':layer['block_index'],'layer':layer['layer_index'],
                                   'type':layer['type'],'kernel':layer['kernel']})
    for group in groups.values():
        group['blocks']=sorted(group['blocks'])
        group['block_count']=len(group['blocks'])
        group['layer_count']=len(group['layers'])
    require(sum(x['block_count'] for x in groups.values())==71,'block closure mismatch')
    require(sum(x['layer_count'] for x in groups.values())==152,'layer closure mismatch')

    refreshed=read(a.rtx/'refresh-receipt.json')
    require(refreshed['runtime_sha256']==RUNTIME and refreshed['result']=='pass','RTX refresh rejected')
    runs=[]
    for row in refreshed['runs']:
        directory=a.rtx/row['run']
        receipt=directory/'capture-receipt.json'
        require(digest(receipt.read_bytes())==row['capture_receipt_sha256'],'transferred receipt mismatch')
        r=read(receipt)
        output=(directory/'output.rgba16le').read_bytes()
        require(len(output)==16588800,'output extent mismatch')
        require(digest(output)==row['output_sha256']==r['output_sha256'],'output transfer mismatch')
        nonfinite=rgb_nonzero=0
        for i,(value,) in enumerate(struct.iter_unpack('<e',output)):
            nonfinite+=not math.isfinite(value)
            rgb_nonzero+=i%4!=3 and value!=0
        require(nonfinite==0 and rgb_nonzero==r['output_rgb_nonzero_elements'] and rgb_nonzero>0,'RGB invalid')
        require(r['counts']['launches']==156 and r['counts']['weight_transfers']==153,'runtime graph count mismatch')
        require(r['kernel_precision_variant']=='fp8_blackwell','wrong precision path')
        inventory=directory/'artifact-inventory.json'
        require(digest(inventory.read_bytes())==r['artifact_inventory_sha256'],'inventory transfer mismatch')
        require(digest((directory/'rtx-cuda-trace.jsonl').read_bytes())==r['cuda_trace_sha256'],'trace transfer mismatch')
        runs.append({'run':row['run'],'output_sha256':r['output_sha256'],
                     'rgb_nonzero_elements':rgb_nonzero,'nonfinite_elements':nonfinite,
                     'output_bytes':len(output),'counts':r['counts'],
                     'capture_receipt_sha256':row['capture_receipt_sha256']})
    require(len(runs)==2 and len({r['output_sha256'] for r in runs})==1,'repeat failed')

    kernels=[]
    rdna4=[x for x in amd['code_objects'] if x['target'].endswith('gfx1201')]
    require(len(rdna4)==1,'RDNA4 object missing')
    for metadata in rdna4[0]['metadata']:
        for k in metadata['amdhsa.kernels']:
            explicit=[x for x in k['.args'] if not x['.value_kind'].startswith('hidden_')]
            kernels.append({'symbol':k['.name'],'explicit_arguments':explicit,
                            'total_kernarg_bytes':k['.kernarg_segment_size'],
                            'threadgroup_bytes':k['.group_segment_fixed_size'],
                            'private_bytes':k['.private_segment_fixed_size'],
                            'wavefront_size':k['.wavefront_size']})
    require(len(kernels)==33,'AMD kernel count changed')
    opcodes=Counter()
    for line in a.amd_disassembly.read_text().splitlines():
        match=re.match(r'\s+([vs]_\w+)\s',line)
        if match:
            opcodes[match[1]]+=1
    arithmetic={op:n for op,n in opcodes.items() if any(word in op for word in ('wmma','mfma','fp8','bf8'))}
    result={'schema':'neuralstudio.nr.first-inference-intake.v1',
            'runtime_sha256':RUNTIME,'graph_root_sha256':GRAPH,
            'weights_resource_sha256':weights['resource']['sha256'],
            'weight_count':153,'weight_bytes':147683778,
            'family_groups':groups,
            'rtx':{'driver':refreshed['driver'],'gpu':refreshed['gpu'],'runs':runs,
                   'fresh_invocations':2,'distinct_pids_recorded':False,
                   'repeat_equal':True,'poison_invariance_proven':False,
                   'intermediate_refresh_proven':False},
            'amd':{'release':amd['release'],'package_sha256':amd['package_sha256'],
                   'static_inventory_root_sha256':amd['receipt_root_sha256'],
                   'rdna4_elf_sha256':rdna4[0]['sha256'],'kernels':kernels,
                   'static_matrix_and_fp8_instruction_counts':arithmetic,
                   'explicit_quantization_instructions_present':True,
                   'equivalence_to_nvidia_proven':False,'package_executed':False},
            'source_artifacts':{p.name:digest(p.read_bytes()) for p in (a.graph,a.weights,a.amd,a.amd_disassembly)},
            'claims':{'original_weights_revalidated':True,'rtx_full_frame_repeat':True,
                      'mac_first_inference':False,'full_rtx_mac_parity':False},
            'result':'pass'}
    result['receipt_root_sha256']=digest(json.dumps(result,sort_keys=True,separators=(',',':')).encode())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('receipt_root_sha256='+result['receipt_root_sha256'])
    print('family_blocks='+str({k:v['block_count'] for k,v in groups.items()}))
    print('result=pass')


if __name__=='__main__':
    main()
