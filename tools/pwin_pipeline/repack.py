#!/usr/bin/env python3
"""Repack supported model weight fragments using only Python's standard library.

Input tensors remain local user-supplied data. Layout specs describe offsets/shapes, not weights.
B fragments use (lane, word), or the explicit legacy --word-major layout. A fragments always
use (lane, word). Bias half values widen exactly to IEEE binary32, including NaN payloads.
"""
import argparse
import json
import struct
from pathlib import Path


def half_to_float_bits(value):
    sign, exponent, fraction = (value & 0x8000) << 16, (value >> 10) & 31, value & 1023
    if exponent == 31:
        return sign | 0x7f800000 | ((fraction << 13) | (0x400000 if fraction else 0))
    if exponent:
        return sign | ((exponent + 112) << 23) | (fraction << 13)
    if not fraction:
        return sign
    exponent = -14
    while not fraction & 1024:
        fraction <<= 1; exponent -= 1
    return sign | ((exponent + 127) << 23) | ((fraction & 1023) << 13)


def repack(blob, spec, word_major=False):
    out, layout = bytearray(), {}
    def align(n):
        out.extend(b'\0' * (-len(out) % n))
    def take(offset, length):
        if not isinstance(offset, int) or offset < 0 or offset + length > len(blob):
            raise ValueError('Model layout reads outside the supplied weights')
        return blob[offset:offset+length]
    def add_name(name, entry):
        if name in layout:
            raise ValueError('Duplicate layout name: ' + name)
        layout[name] = entry
    for matrix in spec.get('matrices', []):
        k, n = matrix['k'], matrix['n']
        if k <= 0 or n <= 0 or k % 16 or n % 16:
            raise ValueError('Matrix dimensions must be positive multiples of 16')
        tiles = matrix['tiles']
        if len(tiles) != k//16 or any(len(row) != n//16 for row in tiles):
            raise ValueError('Matrix tile grid does not cover its declared shape')
        values = bytearray(k*n*2)
        role_a = matrix.get('fragment', 'B') == 'A'
        for ki, row in enumerate(tiles):
            for ni, offset in enumerate(row):
                words = struct.unpack('<128I', take(offset,512))
                for lane in range(32):
                    for word in range(4):
                        packed = words[word*32+lane if word_major and not role_a else lane*4+word]
                        if role_a:
                            r, c = (lane>>2)+8*(word&1), 2*(lane&3)+8*(word>>1)
                            positions = ((r,c),(r,c+1))
                        else:
                            r, c = 2*(lane&3)+8*(word&1), (lane>>2)+8*(word>>1)
                            positions = ((r,c),(r+1,c))
                        for element,(r,c) in enumerate(positions):
                            struct.pack_into('<H',values,2*((ki*16+r)*n+ni*16+c),(packed>>(16*element))&65535)
        align(256)
        add_name(matrix['name'],dict(offset=len(out),k=k,n=n,dtype='half'))
        out.extend(values)
    for kind in ('biases','vectors'):
        for entry in spec.get(kind, []):
            n, offsets = entry['n'], entry['offsets']
            if n <= 0 or n%16 or len(offsets) not in (0,n//16):
                raise ValueError('Vector tile list does not cover its declared shape')
            values = b''.join(take(offset,32) for offset in offsets) if offsets else bytes(n*2)
            if kind == 'biases':
                align(256)
                add_name(entry['name'],dict(offset=len(out),n=n,rows=16,dtype='float'))
                row = b''.join(struct.pack('<I',half_to_float_bits(value)) for (value,) in struct.iter_unpack('<H',values))
                out.extend(row*16)
            else:
                align(64)
                add_name(entry['name'],dict(offset=len(out),n=n,dtype='half'))
                out.extend(values)
    for derived in spec.get('derived', []):
        if derived['operation'] != 'compact_groups':
            raise ValueError('Unknown derived weight operation')
        source = layout[derived['source']]
        group, keep = derived['group_width'], derived['keep']
        if group <= 0 or not 0 < keep <= group or source['n'] % group:
            raise ValueError('Invalid channel group compaction')
        columns = source['n'] // group * keep
        rows = source.get('k', source.get('rows', 1))
        size = 2 if source['dtype'] == 'half' else 4
        data = b''.join(out[source['offset']+(row*source['n']+start)*size:
                            source['offset']+(row*source['n']+start+keep)*size]
                        for row in range(rows) for start in range(0,source['n'],group))
        align(256)
        result = dict(source, offset=len(out), n=columns)
        add_name(derived['name'], result)
        out.extend(data)
    return bytes(out), layout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',required=True,type=Path)
    parser.add_argument('--spec',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--layout',required=True,type=Path)
    parser.add_argument('--word-major',type=int,choices=(0,1),default=0)
    args=parser.parse_args()
    data,layout=repack(args.weights.read_bytes(),json.loads(args.spec.read_text()),bool(args.word_major))
    args.out.write_bytes(data)
    args.layout.write_text(json.dumps(layout,indent=1)+'\n')
    print(f'wrote {len(data)} bytes')


if __name__=='__main__':
    main()
