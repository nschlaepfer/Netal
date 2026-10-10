#!/usr/bin/env python3
"""CPU diagnostic-math tests; synthetic inputs are never model/game evidence."""
import importlib.util
import pathlib
import numpy as np

path=pathlib.Path(__file__).with_name('analyze-nr-motion-pair.py')
spec=importlib.util.spec_from_file_location('motion_analysis',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
rng=np.random.default_rng(1529)
previous=rng.uniform(.02,3,(24,32,3)).astype(np.float32)
current=np.roll(previous,(1,2),axis=(0,1))
motion=np.zeros((24,32,2),np.float32);motion[:]=(-2,-1)
result=module.census(previous,current,motion,(1.,1.))
scores=result['scores']
assert scores['current_to_previous_input_pixels']['common_valid']['mean_abs_log1p_rgb']<1e-7
for name in ('zero_motion','wrong_sign','wrong_y_sign','wrong_game_output_scale_5_over_3',
             'wrong_model_scale_1p5_in_input_space','plus_half_texel_offset'):
    assert scores[name]['common_valid']['mean_abs_log1p_rgb']>.01,name
assert len({s['common_valid']['pixels'] for s in scores.values()})==1
assert not result['game_convention_verified'] and not result['temporal_admission']
yy,xx=np.mgrid[:24,:32]
sampled,ok=module.sample(previous,xx.astype(float),yy.astype(float))
assert ok.all() and np.array_equal(sampled,previous)
# Finite-mask intersection excludes invalid samples rather than clipping them
# into the scoring domain. Nonfinite motion cannot count as a zero-motion pixel.
motion[10,10]=np.nan
bad=module.census(previous,current,motion,(1.,1.))
assert bad['motion_nonfinite_pixels']==1
assert bad['scores']['current_to_previous_input_pixels']['common_valid']['pixels']==scores['current_to_previous_input_pixels']['common_valid']['pixels']-1
outside=np.full(xx.shape,1e9)
_,valid=module.sample(previous,outside,outside)
assert not valid.any()
print('PASS synthetic two-axis translation, wrong-sign/scale/center controls, shared scoring masks, identity sampling and nonfinite/out-of-bounds refusal; no game or temporal admission')
