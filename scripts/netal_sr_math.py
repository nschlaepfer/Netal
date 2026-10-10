"""Formula-generated SR arithmetic assets, without model data or captured binary inputs.

Contracts are pinned to the existing verified runtime assets. Exp2 uses truncation toward
zero and one explicitly recorded SM120 calibration point; do not silently replace it with
nearest-even half conversion. RCP uses nearest-even half conversion and canonical NaNs.
"""
import hashlib
import math
import struct

EXPECTED = {
 'sm120-texture-weights.bin':'0fb8513774e3d71d75073001f79572c30643da6a9b0bec4a7e6808952d5b095b',
 'rcp-f16-to-f16-lut.bin':'0ad848a40f4d577ed93975fa8185da7267a3b2c7972172868625d5505a5b4f96',
 'ex2-f16x2-lut.bin':'0b1bab5ab5e9f0b651a93cbb9f3d9ba6dfd06213bbde208c85ea6a1715fa5359',
}
EX2_CALIBRATION = {0x8001:0x3c00}  # exp2(-minimum half subnormal) observed as 1 on SM120.


def half_value(bits):return struct.unpack('<e',struct.pack('<H',bits))[0]
def half_bits(value):return struct.unpack('<H',struct.pack('<e',value))[0]


def reciprocal_table():
 out=bytearray()
 for bits in range(65536):
  value=half_value(bits)
  if math.isnan(value):result=0x7fff
  elif value==0:result=0xfc00 if bits&0x8000 else 0x7c00
  else:
   try:result=half_bits(1.0/value)
   except OverflowError:result=0xfc00 if value<0 else 0x7c00
  out.extend(struct.pack('<H',result))
 return bytes(out)


def exp2_table():
 out=bytearray()
 for bits in range(65536):
  value=half_value(bits)
  if bits in EX2_CALIBRATION:result=EX2_CALIBRATION[bits]
  elif math.isnan(value):result=0x7fff
  elif value>=16:result=0x7c00
  elif value < -25:result=0
  else:
   exact=2.0**value
   try:
    result=half_bits(exact)
    if half_value(result)>exact:result-=1
   except OverflowError:result=0x7bff
  out.extend(struct.pack('<H',result))
 return bytes(out)


def texture_table():
 out=bytearray(b'NSTEXW1\0'+struct.pack('<3I',257,257*257,16))
 for y in range(257):
  for x in range(257):
   # Round the joint coefficient half-up, then derive the other three so that both
   # marginal weights and their total are exact. Rounding four products independently
   # with ties-to-even gives the wrong result at 2,048 coefficient entries.
   joint=(x*y+128)//256
   out.extend(struct.pack('<4f',(256-x-y+joint)/256,(x-joint)/256,(y-joint)/256,joint/256))
 return bytes(out)


def generate():
 files={'sm120-texture-weights.bin':texture_table(),'rcp-f16-to-f16-lut.bin':reciprocal_table(),'ex2-f16x2-lut.bin':exp2_table()}
 for name,data in files.items():
  if hashlib.sha256(data).hexdigest()!=EXPECTED[name]:
   raise ValueError('Arithmetic contract drift while generating '+name)
 return files
