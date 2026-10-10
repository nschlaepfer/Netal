#!/usr/bin/env python3
"""Generate compact-EX2/live-zero FG overlays from source; no model or GPU execution."""
import pathlib, sys
repository, overlay = map(pathlib.Path, sys.argv[1:3])
hmma_zero = sys.argv[3] == '1'
for name in ('l52', 'l54'):
    source = repository / 'src/fg_production_contract_l3_l87' / f'{name}_production_exact.metal'
    text = source.read_text()
    include = '#include "../fg_math_ex2/sm120_ex2.metalh"'
    call = 'neuralstudio_fg_math::sm120_ex2_nonpositive('
    if text.count(include) != 1:
        raise SystemExit(f'compact runtime include anchor drift: {source}')
    if text.count(call) not in (1, 2):
        raise SystemExit(f'compact runtime call anchor drift: {source}')
    text = text.replace(include, '#include "sm120_ex2_compact.metalh"')
    text = text.replace(call,
                        'neuralstudio_fg_math::sm120_ex2_nonpositive_compact(')
    expected_types = 1 if name == 'l52' else 2
    for old, new in (
        ('device const uint* ex2_unit_table',
         'device const uchar* ex2_unit_table'),
        ('device const uint* ex2_fraction_table',
         'device const uchar* ex2_fraction_table')):
        if text.count(old) != expected_types:
            raise SystemExit(f'compact runtime table-type drift: {source}: {old}')
        text = text.replace(old, new)
    (overlay / f'{name}_compact.metal').write_text(text)
if hmma_zero:
    conv = (repository/'src/fg_conv/conv_3x3.metal').read_text()
    anchor = '      accumulator = fg_hmma_f16_bits_16(a_values, b_values, accumulator);'
    if conv.count(anchor) != 1:
        raise SystemExit('live HMMA conv3 anchor drift')
    conv = conv.replace(anchor, anchor + '\n'
        '      if (as_type<ushort>(accumulator) == 0x8000u)\n'
        '        accumulator = as_type<half>(ushort(0u));')
    (overlay/'conv_3x3_live_zero.metal').write_text(conv)
    bot0 = (repository/'src/fg_connected_custom_l56_l83/connected_block0_bot0.metal').read_text()
    anchor = 'a=fg_hmma_f16_bits_16(av,bv,a);'
    if bot0.count(anchor) != 1:
        raise SystemExit('live HMMA bot0 anchor drift')
    bot0 = bot0.replace(anchor, anchor +
        'if(as_type<ushort>(a)==0x8000u)a=as_type<half>(ushort(0u));')
    (overlay/'block0_bot0_live_zero.metal').write_text(bot0)
