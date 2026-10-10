#!/usr/bin/env python3
"""Verify raw frame captures and render labelled HDR previews, never inference inputs.

The fixed Reinhard+sRGB display transform is for side-by-side inspection only;
it is NOT the game's tone mapper or proof of on-screen presentation.
"""
import argparse, hashlib, json, pathlib
import numpy as np
from PIL import Image, ImageDraw

def decode(data, width, height, fmt):
    if fmt == 115:
        return np.frombuffer(data, dtype='<f2').reshape(height,width,4).astype(np.float32)[...,:3]
    if fmt != 92:
        raise ValueError(f'unsupported capture format {fmt}')
    packed=np.frombuffer(data,dtype='<u4').reshape(height,width)
    channels=[]
    for shift,mantissa in ((0,6),(11,6),(22,5)):
        code=(packed>>shift)&((1<<(mantissa+5))-1)
        exp=(code>>mantissa).astype(np.int32); frac=(code&((1<<mantissa)-1)).astype(np.float32)/(1<<mantissa)
        value=np.ldexp(1+frac,exp-15)
        value=np.where(exp==0,np.ldexp(frac,-14),value)
        value=np.where(exp==31,np.where(frac==0,np.inf,np.nan),value)
        channels.append(value)
    return np.stack(channels,axis=-1)

def preview(rgb):
    x=np.maximum(np.nan_to_num(rgb,nan=0,posinf=65504,neginf=0),0)
    x=x/(1+x)
    x=np.where(x<=.0031308,12.92*x,1.055*np.power(x,1/2.4)-.055)
    return Image.fromarray(np.uint8(np.rint(np.clip(x,0,1)*255)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('capture');ap.add_argument('out');a=ap.parse_args()
    source=pathlib.Path(a.capture);root=source.parent;out=pathlib.Path(a.out)
    record=json.loads(source.read_text());out.mkdir(parents=True,exist_ok=False)
    arrays={};stats={};display_transforms={}
    for entry in record['files']:
        name=entry['file']
        if pathlib.PurePath(name).name!=name:raise ValueError('nonlocal capture path')
        raw=(root/name).read_bytes()
        if len(raw)!=entry['bytes'] or hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError(f'capture hash/size mismatch: {name}')
        rgb=decode(raw,entry['width'],entry['height'],entry['metal_pixel_format']);arrays[name]=rgb
        finite=rgb[np.isfinite(rgb)]
        stats[name]={'nonfinite_components':int(np.count_nonzero(~np.isfinite(rgb))),
            'min':float(finite.min()) if finite.size else None,'max':float(finite.max()) if finite.size else None}
        if record.get('color_mode')=='bounded-reinhard-srgb' and name.startswith('model-'):
            im=Image.fromarray(np.uint8(np.rint(np.clip(np.nan_to_num(rgb),0,1)*255)))
            display_transforms[name]='bounded model-space values displayed directly; chosen experimental color profile'
        else:
            im=preview(rgb);display_transforms[name]='fixed per-channel Reinhard followed by sRGB'
        im.save(out/(name+'.preview.png'))
    labels=(('game-input.bin','Before NR: captured upscaled game color'),('game-output.bin','Game output after NR attempt'))
    panels=[]
    for key,label in labels:
        im=preview(arrays[key]);im.thumbnail((960,600))
        panel=Image.new('RGB',(960,650),(20,20,20));panel.paste(im,((960-im.width)//2,45))
        d=ImageDraw.Draw(panel);d.text((15,12),label,fill='white');panels.append(panel)
    canvas=Image.new('RGB',(1920,700),(20,20,20));canvas.paste(panels[0],(0,0));canvas.paste(panels[1],(960,0))
    ImageDraw.Draw(canvas).text((15,665),'Diagnostic HDR previews: identical Reinhard + sRGB transform. Not game screenshots or quality acceptance.',fill='white')
    canvas.save(out/'comparison.png')
    delta=arrays['game-output.bin']-arrays['game-input.bin']
    stats['change_not_quality_score']={'mean_absolute_rgb_change':float(np.mean(np.abs(delta))),
        'max_absolute_rgb_change':float(np.max(np.abs(delta)))}
    stats['capture_manifest_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    stats['nr_export_completed']=record['nr_export_completed'];stats['presentation_verified']=False
    stats['display_transform']='fixed per-channel Reinhard followed by sRGB; no auto-exposure'
    stats['plane_display_transforms']=display_transforms
    (out/'preview.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(json.dumps(stats,indent=2))

if __name__=='__main__':main()
