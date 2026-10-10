"""Generate the SR PWin ABI from model topology and resolution, never captured pointers.

The formulas match SynthesizeGeometry in dlss_sr_session.mm. Weight offsets are model-layout
metadata for the supported quality network; this module contains no model tensor bytes.
"""
from dataclasses import dataclass
import struct


@dataclass(frozen=True)
class Stage:
    name: str
    level: int
    simdgroups: int
    weights: int
    bindings: tuple
    shift: tuple = (4, 4)


# Binding: (argument byte offset, arena id, area level, area multiplier).
# Model-weight bindings are added separately; nonzero offsets are derived from level areas.
STAGES = (
    Stage('enc0', 0, 1, 0, ((8,6,0,0),(24,7,0,0),(32,8,3,20),(48,8,3,20))),
    Stage('enc1', 1, 2, 67264, ((8,7,0,0),(24,6,0,0),(32,6,0,8))),
    Stage('enc2', 2, 2, 215872, ((8,6,0,0),(24,7,0,0),(32,7,1,12))),
    Stage('enc3', 3, 4, 380928, ((8,7,0,0),(24,6,0,0),(32,6,2,16))),
    Stage('enc4', 4, 4, 759552, ((8,6,0,0),(24,7,0,0),(32,7,3,20)), (4,0)),
    Stage('dec5', 5, 8, 1384576, ((8,7,0,0),(24,8,0,0)), (0,2)),
    Stage('dec4', 4, 4, 2189952, ((8,8,0,0),(16,7,3,20),(24,6,0,0),(40,6,0,0)), (4,0)),
    Stage('dec3', 3, 4, 2782848, ((8,6,0,0),(16,6,2,16),(24,7,0,0),(40,7,0,0))),
    Stage('dec2', 2, 2, 3161984, ((8,7,0,0),(16,7,1,12),(24,6,0,0),(40,6,0,0))),
    Stage('dec1', 1, 2, 3327360, ((8,6,0,0),(16,6,0,8),(24,7,0,0),(40,7,0,0))),
    Stage('dec0', 0, 1, 3476352, ((8,7,0,0),(16,8,3,20),(24,6,0,0),(40,7,0,32))),
)


def round_up(n, multiple):
    return ((n + multiple - 1) // multiple) * multiple


def profile(width, height):
    if not (isinstance(width, int) and isinstance(height, int) and 1 <= width <= 16384 and 1 <= height <= 16384):
        raise ValueError('Output dimensions must be integers in 1..16384')
    levels = [(max(256, round_up((width + 3) // 4, 32)), max(256, round_up((height + 3) // 4, 32)))]
    for _ in range(5):
        w, h = levels[-1]; levels.append(((w + 1) // 2, (h + 1) // 2))
    areas = [w*h for w,h in levels]
    files, stages = {}, []
    for index, stage in enumerate(STAGES):
        w, h = levels[stage.level]
        directory = 'pwin/enc0' if index == 0 else f'pwin/pwin-{index+1:02d}'
        entry = f'dltss_pwin_{stage.name}_layer'
        grid = [(w + stage.shift[0] + 7)//8, (h + stage.shift[1] + 7)//8, 1]
        block = [32, 1, stage.simdgroups]
        relocations = [(arg, arena, areas[level]*factor) for arg,arena,level,factor in stage.bindings]
        relocations.append((64, 9, stage.weights))
        config = bytearray(296)
        config[:8] = b'NSPWIN1\0'
        struct.pack_into('<I', config, 8, 1)
        config[12:12+len(entry)] = entry.encode('ascii')
        if index:
            debug_entry = f'pwin_{index+1:02d}_wmma_debug'.encode('ascii')
            config[76:76+len(debug_entry)] = debug_entry
        struct.pack_into('<3I3II', config, 140, *grid, *block, len(relocations))
        for row, values in enumerate(relocations):
            struct.pack_into('<IIQ', config, 168+16*row, *values)
        arguments = bytearray(176)
        struct.pack_into('<II', arguments, 0, w, h)
        struct.pack_into('<II', arguments, 56, *stage.shift)
        struct.pack_into('<f', arguments, 164, 1.0)
        files[directory+'/replay-config.bin'] = bytes(config)
        files[directory+'/arguments-template.bin'] = bytes(arguments)
        stages.append({'entry':entry, 'directory':directory, 'extent':[w,h], 'grid':grid, 'block':block,
                       'relocations':[dict(argument_offset=a, allocation_id=b, allocation_offset=c) for a,b,c in relocations]})
    metadata = {'schema':'netal.sr.pwin-layout.v1','output':[width,height],
                'arenas':{'6':areas[0]*80,'7':areas[0]*96,'8':areas[0]*64+areas[3]*20},
                'stages':stages, 'contains_model_weights':False,
                'boundary_abi_included':False, 'complete_runtime':False}
    return files, metadata
