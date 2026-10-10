#!/usr/bin/env python3
"""Extract compressed PTX from our pinned NVIDIA DLL without executing it.

The container header and entry sizes are read directly from the local file.
Python 3.14's standard Zstandard decoder is used; no external extractor source
or third-party model payload is imported.
"""
import argparse
from compression import zstd
import hashlib
import json
from pathlib import Path
import re
import struct


RUNTIME='e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e'
def sha(data):return hashlib.sha256(data).hexdigest()
def require(test,message):
    if not test:raise ValueError(message)


def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',required=True,type=Path)
    p.add_argument('--graph',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    require(not a.output.exists(),'preserve prior extraction')
    data=a.runtime.read_bytes();require(sha(data)==RUNTIME,'wrong runtime')
    graph=json.loads(a.graph.read_text());require(graph['runtime_sha256']==RUNTIME,'wrong graph identity')
    selected={m['sha256']:m for m in graph['modules']}
    offsets=[];position=0;magic=bytes.fromhex('50ed55ba')
    while (position:=data.find(magic,position))>=0:
        offsets.append(position);position+=4
    require(len(offsets)==15,'fatbin count mismatch')
    prepared=[];matched_modules=set();matched_functions=set()
    for index,offset in enumerate(offsets):
        _,version,header_size,payload_size=struct.unpack_from('<IHHQ',data,offset)
        end=offset+header_size+payload_size
        require(version==1 and header_size==16 and end<=len(data),'invalid fatbin bounds')
        container_sha=sha(data[offset:end]);module=selected.get(container_sha)
        if module:matched_modules.add(module['module_id'])
        cursor=offset+header_size;entries=[]
        while cursor<end:
            require(cursor+64<=end,'truncated entry header')
            kind,entry_version,entry_header,size=struct.unpack_from('<HHIQ',data,cursor)
            require(kind in (1,2) and entry_header>=64 and cursor+entry_header+size<=end,'invalid entry bounds')
            payload=data[cursor+entry_header:cursor+entry_header+size]
            entry={'kind':kind,'entry_version':entry_version,'stored_bytes':size,'stored_sha256':sha(payload)}
            if kind==1:
                require(payload[:4]==bytes.fromhex('28b52ffd'),'PTX entry is not a Zstandard frame')
                expected_size=struct.unpack_from('<Q',data,cursor+56)[0]
                require(0<expected_size<128*1024*1024,'invalid declared decompressed size')
                decoder=zstd.ZstdDecompressor();ptx=decoder.decompress(payload,max_length=expected_size+1)
                require(decoder.eof and len(ptx)==expected_size,'PTX frame size mismatch')
                require(len(decoder.unused_data)<8 and not any(decoder.unused_data),'unexpected data after compressed PTX')
                text=ptx.decode('utf-8')
                version_match=re.search(r'\.version\s+([0-9.]+)',text)
                target_match=re.search(r'\.target\s+(sm_\d+)',text)
                require(version_match and target_match,'missing PTX version/target')
                names=re.findall(r'\.visible\s+\.entry\s+([\w$]+)\s*\(',text)
                require(names and len(names)==len(set(names)),'invalid PTX entries')
                selected_names=[k['name'] for k in graph['selected_kernels'] if module and k['module_id']==module['module_id']]
                require(set(selected_names)<=set(names),'selected function missing from PTX')
                for name in selected_names:matched_functions.add((module['module_id'],name))
                entry.update({'decompressed_bytes':len(ptx),'ptx_sha256':sha(ptx),
                              'ptx_version':version_match.group(1),'target':target_match.group(1),
                              'entry_count':len(names),'entry_names':names,'selected_names':selected_names,
                              'filename':f'module-{index:02d}.ptx'})
                prepared.append((entry['filename'],ptx))
            else:
                require(payload[:4]==b'\x7fELF','compiled entry is not ELF')
            entries.append(entry);cursor+=entry_header+size
        require(cursor==end,'fatbin end mismatch')
        prepared.append((f'module-{index:02d}.json',{'container_sha256':container_sha,'container_bytes':end-offset,
                         'selected_module_id':module['module_id'] if module else None,'entries':entries}))
    require(len(matched_modules)==9 and len(matched_functions)==96,'selected runtime coverage incomplete')
    a.output.mkdir(parents=True)
    modules=[]
    for name,value in prepared:
        if isinstance(value,bytes):(a.output/name).write_bytes(value)
        else:
            modules.append(value);(a.output/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    receipt={'schema':'neuralstudio.nr.compressed-ptx-recovery.v1','runtime_sha256':RUNTIME,
             'graph_manifest_root_sha256':graph['manifest_root_sha256'],
             'fatbin_count':len(modules),'ptx_module_count':sum(e['kind']==1 for m in modules for e in m['entries']),
             'ptx_entry_count':sum(e.get('entry_count',0) for m in modules for e in m['entries']),
             'selected_modules_matched':len(matched_modules),'selected_functions_matched':len(matched_functions),
             'modules':modules,'runtime_executed':False,'extracted_ptx_runtime_input':False,
             'claims':{'pinned_runtime_is_sass_only':False,'selected_ptx_source_recovered':True,
                       'metal_execution_of_ptx_verified':False,'full_model_inference':False},'result':'pass'}
    receipt['receipt_root_sha256']=sha(json.dumps(receipt,sort_keys=True,separators=(',',':')).encode())
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print('ptx_modules='+str(receipt['ptx_module_count'])+' entries='+str(receipt['ptx_entry_count']))
    print('selected_modules=9 selected_functions=96')
    print('receipt_root_sha256='+receipt['receipt_root_sha256'])


if __name__=='__main__':main()
