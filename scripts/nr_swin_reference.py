"""Read pinned Block-1 tensor observations for offline comparison only."""
import importlib.util
import json
from pathlib import Path


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def load_cta(repo,workspace,index):
    q=load_module('nr_qmma_reference',repo/'scripts/build-dlssnr-block1-qmma-all.py')
    ctas=json.loads((workspace/'work/block1-qmma-nine-cta-analysis/block1-qmma-nine-cta-anchors.json').read_text())['ctas']
    cta=ctas[index]['cta']
    raw=workspace/'work/block1-qmma-nine-cta-capture/dlssnr-block1-qmma-nine-cta-20260831-bl/raw'
    files=[raw/f'repeat{repeat}-cta{index}-{cta[0]}-{cta[1]}-{side}.stdout'
           for repeat in (1,2) for side in ('input','output')]
    ins,outs,ins2,outs2=[q.records(path) for path in files]
    if q.normalized(ins)!=q.normalized(ins2) or q.normalized(outs)!=q.normalized(outs2):
        raise ValueError('RTX register observations differ across repeats')
    ir=workspace/'work/subagents/backbone-05-12/block1-prep/block1-normalized-ir.json'
    instructions=[op for op in json.loads(ir.read_text())['instructions'] if op.get('opcode','').startswith('QMMA')]
    if len(instructions)!=len(ins) or len(ins)!=len(outs) or len(ins)!=256:
        raise ValueError('tensor observation count mismatch')
    records=[]
    for op,i,o in zip(instructions,ins,outs):
        match=q.OPERANDS.fullmatch(op['operands'])
        if match is None:raise ValueError('unexpected tensor operands')
        d,ab,bb=map(int,match.group(1,2,3));cb=None if match.group(4) is None else int(match.group(4))
        if i['instruction_index']!=o['instruction_index'] or i['instruction_index']!=op['index']:
            raise ValueError('instruction boundary mismatch')
        if i['active_mask']!=0xffffffff or o['active_mask']!=0xffffffff:
            raise ValueError('partial warp')
        ri,ro=q.register_columns(i),q.register_columns(o)
        records.append({'a':q.matrix_a(ri,ab),'b':q.matrix_b(ri,bb),
                        'c':bytes(256) if cb is None else q.matrix_half(ri,cb),
                        'd':q.matrix_half(ro,d)})
    return q,cta,records,{path.name:q.sha(path.read_bytes()) for path in files+[ir]}
