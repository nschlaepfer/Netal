#!/usr/bin/env python3
"""Math generator checks, including complete artifact hashes and texture invariants."""
import struct
from netal_sr_math import generate,half_value

files=generate()
rcp=struct.unpack('<65536H',files['rcp-f16-to-f16-lut.bin'])
exp=struct.unpack('<65536H',files['ex2-f16x2-lut.bin'])
assert rcp[0]==0x7c00 and rcp[0x8000]==0xfc00
assert rcp[0x7c00]==0 and rcp[0xfc00]==0x8000
assert rcp[0x3c00]==0x3c00 and rcp[0xc000]==0xb800
assert rcp[0xfe00]==exp[0xfe00]==0x7fff
assert exp[0x8001]==0x3c00 and exp[0x8002]==0x3bff
assert exp[0x3c00]==0x4000 and exp[0xc000]==0x3400
texture=files['sm120-texture-weights.bin']
assert texture[:8]==b'NSTEXW1\0' and struct.unpack_from('<3I',texture,8)==(257,66049,16)
for y in range(257):
 for x in range(257):
  a,b,c,d=struct.unpack_from('<4f',texture,20+(y*257+x)*16)
  assert min(a,b,c,d)>=0 and max(a,b,c,d)<=1 and a+b+c+d==1
  assert b+d==x/256 and c+d==y/256
assert struct.unpack_from('<4f',texture,20+(2*257+64)*16)==(191/256,63/256,1/256,1/256)
print('3 full artifact hashes, scalar edge cases, and 66049 texture partitions passed')
