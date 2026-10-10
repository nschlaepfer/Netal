#!/usr/bin/env python3
"""Read-only reprojection diagnostics. No inference, history import or auto-admission.

Compare all hypotheses on the SAME valid pixels. A winning residual alone does
not establish motion convention: lighting, occlusion, jitter and scene content
can confound it. Input coordinates are pixels of the1152x720 color texture, not
model/output pixels. All calculations run on CPU after raw capture completion.
"""
import argparse
import hashlib
import json
import pathlib
import numpy as np


def read_frame(path):
    path = pathlib.Path(path)
    raw = path.read_bytes()
    r = json.loads(raw)
    if r.get('schema') != 'netal.nr.motion-pair-frame.v1':
        raise ValueError('wrong capture schema')
    if r.get('gpu_completed') is not True or r.get('encoded') is not True:
        raise ValueError('GPU completion/copy not established')
    if r.get('actual_nr') is not False or r.get('temporal_admitted') is not False:
        raise ValueError('requires capture-only evidence')
    expected = {'color.rgba16f': (115, 8, 4), 'motion.rg16f': (65, 4, 2)}
    planes = {}
    for e in r['files']:
        name = e['file']
        if name not in expected or name in planes:
            raise ValueError('unknown/duplicate plane')
        fmt, bpp, channels = expected[name]
        if (e['width'], e['height'], e['metal_pixel_format'], e['bytes_per_row'], e['bytes']) != (
                1152, 720, fmt, 1152*bpp, 1152*720*bpp):
            raise ValueError('plane geometry/format mismatch')
        plane = path.parent / name
        if plane.stat().st_size != e['bytes']:
            raise ValueError('plane length mismatch')
        data = plane.read_bytes()
        if hashlib.sha256(data).hexdigest() != e['sha256']:
            raise ValueError('plane hash mismatch')
        planes[name] = np.frombuffer(data, dtype='<f2').reshape(720,1152,channels).astype(np.float32)
    if planes.keys() != expected.keys():
        raise ValueError('incomplete planes')
    return r, planes, hashlib.sha256(raw).hexdigest()


def sample(image, x, y):
    """Bilinear texel-index coordinates, returning validity separately."""
    h, w = image.shape[:2]
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= w-1) & (y >= 0) & (y <= h-1)
    xx = np.clip(np.nan_to_num(x, nan=0, posinf=0, neginf=0), 0, w-1)
    yy = np.clip(np.nan_to_num(y, nan=0, posinf=0, neginf=0), 0, h-1)
    x0, y0 = xx.astype(np.int64), yy.astype(np.int64)
    x1, y1 = np.minimum(x0+1,w-1), np.minimum(y0+1,h-1)
    dx, dy = (xx-x0)[...,None], (yy-y0)[...,None]
    out = ((1-dy)*((1-dx)*image[y0,x0]+dx*image[y0,x1]) +
           dy*((1-dx)*image[y1,x0]+dx*image[y1,x1]))
    return out, valid & np.isfinite(out).all(axis=-1)


def census(previous, current, motion, scales, jitter_delta=(0.,0.)):
    previous, current = previous[...,:3], current[...,:3]
    h, w = current.shape[:2]
    yy, xx = np.mgrid[:h,:w]
    mx, my = motion[...,0]*scales[0], motion[...,1]*scales[1]
    hypotheses = {
        'zero_motion': (np.zeros_like(mx),np.zeros_like(my)),
        'current_to_previous_input_pixels': (mx,my),
        'wrong_sign': (-mx,-my),
        'wrong_y_sign': (mx,-my),
        'wrong_game_output_scale_5_over_3': (mx*(5/3),my*(5/3)),
        'wrong_model_scale_1p5_in_input_space': (mx*1.5,my*1.5),
        'plus_half_texel_offset': (mx+.5,my+.5),
        'plus_recorded_jitter_delta': (mx+jitter_delta[0],my+jitter_delta[1]),
        'minus_recorded_jitter_delta': (mx-jitter_delta[0],my-jitter_delta[1]),
        'wrong_sign_plus_jitter_delta': (-mx+jitter_delta[0],-my+jitter_delta[1]),
        'wrong_y_sign_plus_jitter_delta': (mx+jitter_delta[0],-my+jitter_delta[1]),
        'wrong_scale_5_over_3_plus_jitter_delta': (mx*(5/3)+jitter_delta[0],my*(5/3)+jitter_delta[1]),
        'wrong_scale_1p5_plus_jitter_delta': (mx*1.5+jitter_delta[0],my*1.5+jitter_delta[1]),
    }
    valid = np.isfinite(current).all(axis=-1) & np.isfinite(motion).all(axis=-1)
    sampled = {}
    for name,(dx,dy) in hypotheses.items():
        values, ok = sample(previous,xx+dx,yy+dy)
        sampled[name] = values
        valid &= ok
    # Fixed, explicit contrast compression for diagnostic residuals ONLY.
    target = np.log1p(np.maximum(current,0))
    gx = np.abs(np.diff(target,axis=1,prepend=target[:,:1])).mean(axis=-1)
    gy = np.abs(np.diff(target,axis=0,prepend=target[:1])).mean(axis=-1)
    textured = valid & ((gx+gy) > .01)
    moving = valid & (np.hypot(mx,my) > .25)
    masks = {'common_valid':valid, 'textured_common':textured, 'moving_common':moving,
             'moving_textured_common':moving & textured}
    scores = {}
    for name, values in sampled.items():
        error = np.abs(np.log1p(np.maximum(values,0))-target).mean(axis=-1)
        scores[name] = {}
        for label,mask in masks.items():
            e = error[mask]
            scores[name][label] = {'pixels':int(e.size), 'mean_abs_log1p_rgb':float(e.mean()) if e.size else None,
                'p50':float(np.quantile(e,.5)) if e.size else None,'p95':float(np.quantile(e,.95)) if e.size else None}
    finite = np.isfinite(motion).all(axis=-1)
    magnitude = np.hypot(mx[finite],my[finite])
    return {'scores':scores, 'common_valid_fraction':float(valid.mean()),
        'motion_nonfinite_pixels':int((~finite).sum()),
        'motion_magnitude_p50_p95_max':np.quantile(magnitude,[.5,.95,1]).tolist() if magnitude.size else None,
        'sampling':'bilinear previous input color, integer index = texel center',
        'metric':'mean RGB absolute difference after fixed log1p(max(rgb,0)); no exposure fitting',
        'game_convention_verified':False,'temporal_admission':False,
        'warning':'Residual ranking is diagnostic only; same mask across hypotheses, no automatic policy selection.'}


def analyze(left, right):
    a, ap, ah = read_frame(left)
    b, bp, bh = read_frame(right)
    if (a.get('pair_index'),b.get('pair_index')) != (0,1):
        raise ValueError('ordered pair indices required')
    if a['pid'] != b['pid'] or a['stream'] != b['stream'] or a['source_frame']+1 != b['source_frame']:
        raise ValueError('pair must have same process/stream and consecutive scaler-call ordinals')
    if not a.get('command_id') or not b.get('command_id') or a['command_id']==b['command_id']:
        raise ValueError('distinct command identities required')
    am, bm = a['metadata'], b['metadata']
    for k in ('input_width','input_height','output_width','output_height','color_offset_x','color_offset_y',
              'motion_offset_x','motion_offset_y','output_offset_x','output_offset_y','motion_at_output_resolution',
              'motion_includes_jitter','motion_scale_x','motion_scale_y','pre_exposure'):
        if k not in am or k not in bm or am[k] != bm[k]:
            raise ValueError('changed or absent metadata: '+k)
    for k in ('color_offset_x','color_offset_y','motion_offset_x','motion_offset_y','output_offset_x',
              'output_offset_y','motion_at_output_resolution','motion_includes_jitter'):
        if bm[k] != 0: raise ValueError('unsupported/unknown metadata: '+k)
    if bm.get('reset') is not False:
        raise ValueError('second input is reset/unknown; no temporal comparison')
    scales = [bm['motion_scale_x'],bm['motion_scale_y']]
    jitter = [am['jitter_x']-bm['jitter_x'],am['jitter_y']-bm['jitter_y']]
    if not np.isfinite(scales+jitter+[bm['pre_exposure']]).all():
        raise ValueError('nonfinite scalar metadata')
    result = census(ap['color.rgba16f'],bp['color.rgba16f'],bp['motion.rg16f'],scales,jitter)
    result.update(schema='netal.nr.motion-reprojection-census.v1', manifest_sha256=[ah,bh],
                  source_frames=[a['source_frame'],b['source_frame']],pid=a['pid'],stream=a['stream'],
                  source_frame_semantics=b.get('source_frame_semantics','unknown'), jitter_delta_previous_minus_current=jitter)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('previous_manifest');p.add_argument('current_manifest');p.add_argument('--out',required=True)
    args=p.parse_args(); result=analyze(args.previous_manifest,args.current_manifest)
    with open(args.out,'x') as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2,allow_nan=False))
