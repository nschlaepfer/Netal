#!/usr/bin/env python3
"""Independently verify value-keyed FP16 color-conversion repairs.

No frame, model, GPU output or network weights are read. Decimal arithmetic at
two precisions checks each target against its rounding midpoints. This is a
numerical cross-check, not a formal interval-arithmetic proof.
"""
from decimal import Decimal, localcontext
import json

FORWARD = [(0x0de8,0x1cc5),(0x302f,0x35fd),(0x3e97,0x3a7d),
           (0x44e1,0x3b5f),(0x4975,0x3bb3),(0x532c,0x3bf1),(0x60af,0x3bff)]
INVERSE = [(0x3ad5,0x40a7),(0x3bfd,0x5cae)]


def half_value(bits):
    exponent, mantissa = (bits >> 10) & 31, bits & 1023
    if bits & 0x8000 or exponent == 31:
        raise ValueError('positive finite half required')
    return Decimal(mantissa if exponent == 0 else 1024+mantissa) * Decimal(2)**(exponent-25 if exponent else -24)


def evaluate(bits, inverse):
    x = half_value(bits)
    if inverse:
        y = x/Decimal('12.92') if x <= Decimal('0.04045') else ((x+Decimal('0.055'))/Decimal('1.055'))**(Decimal(12)/5)
        return y/(1-y)
    y = x/(1+x)
    return Decimal('12.92')*y if y <= Decimal('0.0031308') else Decimal('1.055')*y**(Decimal(5)/12)-Decimal('0.055')


def main():
    records = []
    for inverse, pairs in ((False, FORWARD), (True, INVERSE)):
        for src, dst in pairs:
            values = []
            for precision in (80, 160):
                with localcontext() as ctx:
                    ctx.prec = precision
                    value = evaluate(src, inverse)
                    low = (half_value(dst-1)+half_value(dst))/2
                    high = (half_value(dst)+half_value(dst+1))/2
                    assert low < value < high, 'target is outside its rounding cell'
                    margin = min(value-low, high-value)/(high-low)
                    assert margin > Decimal('1e-20'), 'too close to midpoint for this cross-check'
                    values.append(value)
            assert abs(values[0]-values[1]) < Decimal('1e-70')
            records.append({'direction':'inverse' if inverse else 'forward',
                'input':f'{src:04x}','rounded_output':f'{dst:04x}',
                'relative_midpoint_margin':str(margin)})
    print(json.dumps({'precisions':[80,160],'records':records,
        'model_or_frame_data_read':False,'formal_proof_claim':False,'result':'pass'},indent=2))


if __name__ == '__main__':
    main()
