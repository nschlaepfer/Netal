#!/usr/bin/env python3
"""Source-only boundary ABI sanitation, shape and failure tests; no vendor inputs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('abi',ROOT/'tools/netal_sr_abi.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
checks=0

def check(value,name):
    global checks
    assert value,name
    checks+=1
    print('ok '+name)

def fail(call,name):
    try:call()
    except ValueError:check(True,name)
    else:check(False,name)

base={name:bytes(c['bytes']) for name,c in a.CONTRACTS.items()}
raw=bytearray(b'\xff'*328);struct.pack_into('<e',raw,192,-24)
canonical=a.canonicalize('hiluma-output-reset-template.bin',raw)
check(all(canonical[o:o+8]==bytes(8) for o in a.CONTRACTS['hiluma-output-reset-template.bin']['pointers']),'all output pointers removed')
check(canonical[194:200]==bytes(6),'unstable tail padding removed')
check(struct.unpack_from('<e',canonical,192)[0]==-24,'live half coefficient at 192 is not mistaken for a pointer')
fail(lambda:a.canonicalize('hiluma-output-reset-template.bin',bytes(327)),'truncated ABI rejected')
profile=a.for_geometry(base,(1280,800),(854,534))
output=profile['hiluma-output-reset-template.bin']
check(struct.unpack_from('<2e',output,190)==(-23.984375,-23.96875),'horizontal and vertical limits derived independently')
check(struct.unpack_from('<2I',output,136)==(1280,800),'output dimensions set')
check(struct.unpack_from('<2I',output,104)==(853,533),'render bounds set')
fail(lambda:a.for_geometry(base,(16384,16384),(1,1)),'unsupported half coefficient range rejected')
fail(lambda:a.for_geometry(base,(0,1080),(1280,720)),'zero dimension rejected')
fail(lambda:a.for_geometry(base,(1280,720),(1920,1080)),'render larger than output rejected')
with tempfile.TemporaryDirectory() as temporary:
    root=Path(temporary);dumps=root/'dumps';dumps.mkdir();trace=root/'trace.jsonl'
    records=[]
    for i,(name,contract) in enumerate(a.CONTRACTS.items(),1):
        raw=bytes([i])*contract['bytes'];filename=f'arguments-{i:04d}.bin';(dumps/filename).write_bytes(raw)
        # Synthetic fixture pins in this module instance only; production hashes remain unchanged.
        contract['sha256']=hashlib.sha256(a.canonicalize(name,raw)).hexdigest()
        records.append(dict(call='cuLaunchKernel',name=contract['function'],argument_bytes=contract['bytes'],argument_dump=filename))
    def write(rows):trace.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
    write(records);files,manifest=a.extract(trace,dumps)
    check(len(files)==3,'three canonical templates extracted')
    package=root/'abi';package.mkdir()
    for name,data in files.items():(package/name).write_bytes(data)
    (package/'manifest.json').write_text(json.dumps(manifest))
    check(a.validate_package(package)[0]==files,'package inventory validates')
    write(records+[records[0]])
    fail(lambda:a.extract(trace,dumps),'ambiguous duplicate launch rejected')
    write(records[:-1]);fail(lambda:a.extract(trace,dumps),'missing boundary launch rejected')
    bad=[dict(r) for r in records];bad[0]['argument_dump']='../outside.bin';write(bad)
    fail(lambda:a.extract(trace,dumps),'path traversal rejected')
    write(records);(dumps/records[0]['argument_dump']).write_bytes(b'short')
    fail(lambda:a.extract(trace,dumps),'truncated launch file rejected before reading')
    (package/next(iter(files))).write_bytes(b'modified')
    fail(lambda:a.validate_package(package),'changed package data rejected')
print(f'{checks} boundary ABI checks passed')
