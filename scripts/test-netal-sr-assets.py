#!/usr/bin/env python3
"""Source-only SR layout/repack checks. No private weights, captures, NumPy or GPU."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from netal_sr_layouts import profile,STAGES

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('repack',ROOT/'tools/pwin_pipeline/repack.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
checks=0

def check(value,label):
    global checks
    assert value,label
    checks+=1
    print('ok '+label)

def fail(function,label):
    try:function()
    except ValueError:check(True,label)
    else:check(False,label)

for size in ((0,720),(-1,1),(1,16385)):
    fail(lambda:profile(*size),'reject invalid output geometry '+str(size))
files,m=profile(1920,1080)
check(len(files)==22 and len(m['stages'])==11,'all eleven stage ABIs generated')
check(m['arenas']=={'6':11059200,'7':13271040,'8':8890560},'1080p arena geometry')
check(hashlib.sha256(files['pwin/enc0/arguments-template.bin']).hexdigest()=='bc0f17dcc28e338510455a753f5ec2bbf5461bc8442b557b0948f00ff8de00e2','pinned baseline encoder ABI fingerprint')
check(hashlib.sha256(files['pwin/pwin-06/replay-config.bin']).hexdigest()=='6676bdcd339ae632f908f284096f16cabe3200505dfc92d4d209f151ee204c48','pinned baseline decoder dispatch fingerprint')
for size in ((1,1),(1280,800),(1728,972),(3840,2160),(16384,16384)):
    generated,metadata=profile(*size)
    valid=True
    for stage in metadata['stages']:
        args=generated[stage['directory']+'/arguments-template.bin']
        for relocation in stage['relocations']:
            offset=relocation['argument_offset']
            valid &= args[offset:offset+8]==bytes(8)
    check(valid,'no captured addresses in '+str(size))
# Check every finite IEEE half encoding against Python's independent IEEE conversion.
finite_match=True
for half in range(65536):
    if half & 0x7c00==0x7c00:continue
    expected=struct.unpack('<I',struct.pack('<f',struct.unpack('<e',struct.pack('<H',half))[0]))[0]
    finite_match &= r.half_to_float_bits(half)==expected
check(finite_match,'all 63488 finite half encodings widen exactly')
check(r.half_to_float_bits(0x7c01)==0x7fc02000 and r.half_to_float_bits(0xfc01)==0xffc02000,'signaling NaN payload and sign preserved with quieting')
# Encode a logical tile using the declared fragment ABI, then require row-major output.
for role in ('A','B'):
    words=[]
    for lane in range(32):
        for word in range(4):
            if role=='A':
                row,col=(lane>>2)+8*(word&1),2*(lane&3)+8*(word>>1)
                lo,hi=row*16+col,row*16+col+1
            else:
                row,col=2*(lane&3)+8*(word&1),(lane>>2)+8*(word>>1)
                lo,hi=row*16+col,(row+1)*16+col
            words.append(lo|(hi<<16))
    blob=struct.pack('<128I',*words)
    data,layout=r.repack(blob,{'matrices':[{'name':'W','k':16,'n':16,'fragment':role,'tiles':[[0]]}]})
    check(data==struct.pack('<256H',*range(256)),role+' fragment maps into logical tile')
fail(lambda:r.repack(bytes(511),{'matrices':[{'name':'W','k':16,'n':16,'tiles':[[0]]}]}),'short model read refused')
fail(lambda:r.repack(bytes(512),{'matrices':[{'name':'W','k':32,'n':16,'tiles':[[0]]}]}),'incomplete matrix topology refused')
# A small grouped vector has padding after every group, not only at the end.
blob=struct.pack('<16H',*range(16))
data,layout=r.repack(blob,{'vectors':[{'name':'v','n':16,'offsets':[0]}],
                          'derived':[{'name':'compact','source':'v','operation':'compact_groups','group_width':4,'keep':3}]})
start=layout['compact']['offset']
check(list(struct.unpack('<12H',data[start:start+24]))==[0,1,2,4,5,6,8,9,10,12,13,14],'group padding removed without losing later groups')
with tempfile.TemporaryDirectory() as temporary:
    path=Path(temporary)/'enc4.json'
    subprocess.run([sys.executable,str(ROOT/'tools/pwin_pipeline/make-spec-enc4.py'),str(path)],check=True,capture_output=True)
    check(json.loads(path.read_text())==json.loads((ROOT/'tools/pwin_pipeline/spec-enc4.json').read_text()),'enc4 spec generator preserves the slim projection recipe')
print(f'{checks} SR source-asset checks passed')
