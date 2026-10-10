#!/usr/bin/env python3
"""Inspect an AMD NR release as data, without loading or executing its code."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def sha(data):
    return hashlib.sha256(data).hexdigest()


class MessagePack:
    def __init__(self, data):
        self.data, self.pos = data, 0

    def take(self, count):
        end = self.pos + count
        if end > len(self.data):
            raise ValueError('truncated metadata')
        value = self.data[self.pos:end]
        self.pos = end
        return value

    def number(self, fmt):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def read(self):
        tag = self.number('B')
        if tag <= 0x7f:
            return tag
        if tag >= 0xe0:
            return tag - 256
        if 0xa0 <= tag <= 0xbf:
            return self.take(tag & 31).decode()
        if 0x90 <= tag <= 0x9f:
            return [self.read() for _ in range(tag & 15)]
        if 0x80 <= tag <= 0x8f:
            return {self.read(): self.read() for _ in range(tag & 15)}
        if tag in (0xc0, 0xc2, 0xc3):
            return {0xc0: None, 0xc2: False, 0xc3: True}[tag]
        formats = {0xcc:'>B',0xcd:'>H',0xce:'>I',0xcf:'>Q',
                   0xd0:'>b',0xd1:'>h',0xd2:'>i',0xd3:'>q',0xca:'>f',0xcb:'>d'}
        if tag in formats:
            return self.number(formats[tag])
        if tag in (0xd9,0xda,0xdb):
            return self.take(self.number({0xd9:'>B',0xda:'>H',0xdb:'>I'}[tag])).decode()
        if tag in (0xdc,0xdd):
            return [self.read() for _ in range(self.number('>H' if tag == 0xdc else '>I'))]
        if tag in (0xde,0xdf):
            return {self.read():self.read() for _ in range(self.number('>H' if tag == 0xde else '>I'))}
        raise ValueError(f'unsupported metadata type {tag:#x}')


def elf_inventory(data):
    if data[:6] != b'\x7fELF\x02\x01':
        raise ValueError('ELF64 little endian required')
    machine = struct.unpack_from('<H', data, 18)[0]
    if machine != 224:
        raise ValueError('AMDGPU object required')
    section_offset = struct.unpack_from('<Q', data, 40)[0]
    section_size, section_count, names_index = struct.unpack_from('<HHH', data, 58)
    if section_size != 64 or section_offset + section_count*64 > len(data):
        raise ValueError('invalid ELF section table')
    headers = [struct.unpack_from('<IIQQQQIIQQ',data,section_offset+i*64) for i in range(section_count)]
    def content(header):
        offset, size = header[4:6]
        if offset+size > len(data):
            raise ValueError('section exceeds object')
        return data[offset:offset+size]
    names = content(headers[names_index])
    def string(table, offset):
        end = table.find(b'\0',offset)
        if end < 0:
            raise ValueError('unterminated ELF string')
        return table[offset:end].decode()
    sections, symbols, metadata = [], [], []
    for header in headers:
        name = string(names,header[0])
        sections.append({'name':name,'type':header[1],'bytes':header[5]})
        if header[1] == 2:
            symbol_data = content(header)
            symbol_names = content(headers[header[6]])
            if header[9] != 24 or len(symbol_data)%24:
                raise ValueError('invalid symbol table')
            for offset in range(0,len(symbol_data),24):
                st_name, info, other, index, value, size = struct.unpack_from('<IBBHQQ',symbol_data,offset)
                if st_name and info & 15 == 2:
                    symbols.append({'name':string(symbol_names,st_name),'bytes':size})
        if header[1] == 7:
            note = content(header)
            offset = 0
            while offset < len(note):
                name_size, desc_size, kind = struct.unpack_from('<III',note,offset)
                offset += 12
                owner = note[offset:offset+name_size].rstrip(b'\0').decode()
                offset += (name_size+3)&~3
                descriptor = note[offset:offset+desc_size]
                offset += (desc_size+3)&~3
                if owner == 'AMDGPU' and kind == 32:
                    parser = MessagePack(descriptor)
                    decoded = parser.read()
                    if parser.pos != len(descriptor):
                        raise ValueError('trailing MessagePack bytes')
                    metadata.append(decoded)
    return {'machine':machine,'flags':struct.unpack_from('<I',data,48)[0],
            'sections':sections,'function_symbols':symbols,'metadata':metadata}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('package',type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    data=a.package.read_bytes()
    if sha(data) != '5a28d907bba471504c6c7be6901a4eb6610f46b4d86ecb23b0430ba8bf3fc5d5':
        raise ValueError('unrecognized v0.2.10 installer hash')
    if a.output.exists():
        raise ValueError('preserve existing output')
    marker=b'__CLANG_OFFLOAD_BUNDLE__'
    base=data.find(marker)
    if base<0 or data.find(marker,base+1)>=0:
        raise ValueError('expected one offload bundle')
    pos=base+len(marker)
    count=struct.unpack_from('<Q',data,pos)[0]
    pos+=8
    if count != 5:
        raise ValueError('expected host entry plus four architecture objects')
    entries=[]
    for _ in range(count):
        offset,size,name_size=struct.unpack_from('<QQQ',data,pos)
        pos+=24
        target=data[pos:pos+name_size].decode();pos+=name_size
        if size == 0 and target.startswith('host-'):
            continue
        payload=data[base+offset:base+offset+size]
        if len(payload)!=size:
            raise ValueError('truncated code object')
        item={'target':target,'bytes':size,'sha256':sha(payload),**elf_inventory(payload)}
        entries.append((item,payload))
    a.output.mkdir(parents=True)
    for item,payload in entries:
        arch=item['target'].split('--')[-1]
        (a.output/f'{arch}.elf').write_bytes(payload)
    receipt={'schema':'neuralstudio.nr.amd-static-intake.v1',
             'package_sha256':sha(data),'release':'v0.2.10',
             'source_url':'https://github.com/danielblnc/DLSS-NR-on-AMD/releases/tag/v0.2.10',
             'executed_package':False,'nvidia_runtime_loaded':False,
             'code_objects':[item for item,_ in entries],
             'claims':{'native_mac_inference':False,'amd_correctness_verified':False}}
    receipt['receipt_root_sha256']=sha(json.dumps(receipt,sort_keys=True,separators=(',',':')).encode())
    (a.output/'inventory.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    for item,_ in entries:
        kernels=[k for m in item['metadata'] for k in m.get('amdhsa.kernels',[])]
        print(item['target'],item['bytes'],'kernels='+str(len(kernels)))
    print('receipt_root_sha256='+receipt['receipt_root_sha256'])


if __name__=='__main__':
    main()
