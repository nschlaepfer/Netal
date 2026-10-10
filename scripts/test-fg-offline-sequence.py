#!/usr/bin/env python3
"""Check the shared FG input decoder against independently encoded bytes."""
from pathlib import Path
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TEXTURE = 640 * 360 * 4
RECORD = 16 + 3 * TEXTURE
HEADER = struct.pack('<8sIIII', b'NFGSEQ01', 640, 360, 3, RECORD)
frames = []
for index, reset in enumerate((True, False, True)):
    motion = struct.pack('<ee', 0 if reset else -16, 0 if reset else 4) * (640 * 360)
    frames.append(struct.pack('<QII', index, reset, 0) + bytes((3, 5, 7, 255)) * (640 * 360)
                  + motion + struct.pack('<f', 0.5) * (640 * 360))
valid = HEADER + b''.join(frames)

with tempfile.TemporaryDirectory(prefix='netal-fg-sequence-') as temp:
    folder = Path(temp)
    source = folder / 'check.cpp'
    source.write_text(r'''
#include "offline_sequence.h"
#include <fstream>
#include <iostream>
#include <iterator>
int main(int argc, char** argv) {
  try {
    if(argc!=2) return 3;
    std::ifstream file(argv[1],std::ios::binary);
    neuralstudio::fg::offline::Bytes bytes{std::istreambuf_iterator<char>(file),{}};
    auto frames=neuralstudio::fg::offline::Decode(bytes);
    if(frames.size()!=3 || frames[0].color[0]!=3 || frames[1].motion[1]!=0xcc ||
       frames[1].motion[3]!=0x44 || !frames[2].reset || frames[2].depth[3]!=0x3f) return 4;
    auto encoded=neuralstudio::fg::offline::Header(3);
    for(const auto& frame:frames) neuralstudio::fg::offline::Append(encoded,frame);
    if(encoded!=bytes) return 5;
    return 0;
  } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 2; }
}
''')
    binary = folder / 'check'
    subprocess.run(['xcrun', 'clang++', '-std=c++20', '-Wall', '-Wextra', '-Werror',
                    '-I', str(ROOT / 'src/fg_host'), str(source), '-o', str(binary)], check=True)
    fixture = folder / 'sequence.nfg'
    fixture.write_bytes(valid)
    subprocess.run([str(binary), str(fixture)], check=True)
    cases = [('truncated', valid[:-1]), ('trailing', valid+b'\0')]
    def mutate(name, offset, replacement):
        b = bytearray(valid); b[offset:offset+len(replacement)] = replacement
        cases.append((name, b))
    mutate('magic', 0, b'BADINPUT')
    mutate('extent', 8, struct.pack('<I', 1920))
    mutate('count', 16, struct.pack('<I', 0xffffffff))
    mutate('record-size', 20, struct.pack('<I', RECORD+4))
    mutate('first-not-reset', 32, struct.pack('<I', 0))
    mutate('frame-gap', 24+RECORD, struct.pack('<Q', 7))
    mutate('invalid-reset-flag', 24+RECORD+8, struct.pack('<I', 2))
    mutate('reserved-field', 36, struct.pack('<I', 1))
    mutate('reset-motion', 40+TEXTURE, struct.pack('<ee', -1, 0))
    mutate('motion-nan', 40+RECORD+TEXTURE, struct.pack('<HH', 0x7e00, 0))
    mutate('depth-nan', 40+2*TEXTURE, struct.pack('<I', 0x7fc00000))
    mutate('depth-out-of-range', 40+2*TEXTURE, struct.pack('<f', 1.1))
    for name, data in cases:
        fixture.write_bytes(data)
        run = subprocess.run([str(binary), str(fixture)], capture_output=True, text=True)
        assert run.returncode == 2, (name, run.returncode, run.stderr)
    print(f'FG offline sequence: independent roundtrip and {len(cases)} malformed-input rejections passed')
