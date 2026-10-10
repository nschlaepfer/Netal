#!/usr/bin/env python3
"""Post-execution full-model numerical comparison, never inference input or acceptance.

Inputs are already-computed1920x1080 RGBA16F model outputs. Equal bytes do not
establish RTX provenance or input identity. Numerical differences do not by
themselves establish visual/temporal acceptability. No clipping or repair.
"""
import argparse
import hashlib
import json
import pathlib
import numpy as np

WIDTH, HEIGHT, CHANNELS = 1920, 1080, 4


def metrics(reference, candidate):
    if reference.dtype != np.dtype('<f2') or candidate.dtype != np.dtype('<f2'):
        raise ValueError('raw little-endian half arrays required')
    if reference.shape != candidate.shape or reference.ndim != 3 or reference.shape[-1] != 4:
        raise ValueError('matching HxWx4 arrays required')
    rb, cb = reference.view('<u2'), candidate.view('<u2')
    different = rb != cb
    rf, cf = reference.astype(np.float64), candidate.astype(np.float64)
    finite = np.isfinite(rf) & np.isfinite(cf)
    numeric_equal = finite & (rf == cf)
    def special(x):
        return {'nan':int(np.isnan(x).sum()),'positive_inf':int(np.isposinf(x).sum()),
                'negative_inf':int(np.isneginf(x).sum()),'negative_zero':int(((x==0)&np.signbit(x)).sum())}
    def scope(mask):
        d = np.abs(cf[mask]-rf[mask])
        if not d.size:
            return {'finite_pairs':0,'mean_abs':None,'rmse':None,'max_abs':None,'p50_abs':None,'p95_abs':None,'p99_abs':None}
        return {'finite_pairs':int(d.size),'mean_abs':float(d.mean()),'rmse':float(np.sqrt(np.mean(d*d))),
                'max_abs':float(d.max()),'p50_abs':float(np.quantile(d,.5)),
                'p95_abs':float(np.quantile(d,.95)),'p99_abs':float(np.quantile(d,.99))}
    rgb_mask=finite.copy();rgb_mask[...,3]=False
    alpha_mask=finite.copy();alpha_mask[...,:3]=False
    flat=np.flatnonzero(different)
    # Ordered half encodings give representation-step distance, NOT an error
    # bound. Signed zero has one representation step, despite numeric equality.
    def ordered(bits):
        return np.where(bits & 0x8000, (~bits)&0xffff, bits|0x8000).astype(np.int32)
    steps=np.abs(ordered(rb)[finite]-ordered(cb)[finite])
    return {
        'shape':list(reference.shape),'half_components':int(reference.size),
        'bit_identical_components':int((~different).sum()),'differing_half_components':int(different.sum()),
        'bit_identical_fraction':float((~different).mean()),
        'differing_rgba_pixels':int(different.any(axis=-1).sum()),
        'differing_rgb_pixels':int(different[...,:3].any(axis=-1).sum()),
        'first_different_component_y_x_channel':list(map(int,np.unravel_index(flat[0],reference.shape))) if flat.size else None,
        'finite_numeric_equal_components':int(numeric_equal.sum()),
        'nonfinite_pair_components_excluded_from_error':int((~finite).sum()),
        'reference_special_values':special(rf),'candidate_special_values':special(cf),
        'all_finite_components':scope(finite),'rgb_finite_components':scope(rgb_mask),
        'alpha_finite_components':scope(alpha_mask),
        'finite_half_representation_steps_max':int(steps.max()) if steps.size else None,
        'finite_half_representation_steps_p99':float(np.quantile(steps,.99)) if steps.size else None,
        'reference_rgb_outside_0_1_finite':int(((rf[...,:3]<0)|(rf[...,:3]>1))[np.isfinite(rf[...,:3])].sum()),
        'candidate_rgb_outside_0_1_finite':int(((cf[...,:3]<0)|(cf[...,:3]>1))[np.isfinite(cf[...,:3])].sum()),
        'note':'Unclipped model-space differences. Nonfinite pairs explicitly excluded from finite-error metrics, never repaired. Representation steps are not quality thresholds.'}


def compare(reference, candidate):
    raw=[];paths=[pathlib.Path(reference),pathlib.Path(candidate)]
    for p in paths:
        if p.stat().st_size != WIDTH*HEIGHT*CHANNELS*2:
            raise ValueError('expected full1920x1080 RGBA16F output: '+str(p))
        raw.append(p.read_bytes())
    arrays=[np.frombuffer(b,dtype='<f2').reshape(HEIGHT,WIDTH,CHANNELS) for b in raw]
    return {'schema':'netal.nr.full-model-numerical-comparison.v1',
        'reference_sha256':hashlib.sha256(raw[0]).hexdigest(),'candidate_sha256':hashlib.sha256(raw[1]).hexdigest(),
        'all_bytes_equal':raw[0]==raw[1],'metrics':metrics(*arrays),
        'input_identity_verified':False,'model_provenance_verified':False,
        'rtx_parity_accepted':False,'visual_quality_accepted':False,'temporal_quality_accepted':False,
        'timing_claim':False,'execution_ingress':False,
        'note':'Output-only audit. Pair this with retained execution/input/weight receipts; equal outputs alone do not prove a correct model or identical inputs.'}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference');p.add_argument('candidate');p.add_argument('--out',required=True)
    a=p.parse_args();result=compare(a.reference,a.candidate)
    with open(a.out,'x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2,allow_nan=False))
