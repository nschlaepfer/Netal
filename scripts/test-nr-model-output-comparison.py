#!/usr/bin/env python3
"""CPU-only sanity/negative cases for output statistics; no model or GPU."""
import importlib.util
import pathlib
import numpy as np
path=pathlib.Path(__file__).with_name('compare-nr-model-outputs.py')
spec=importlib.util.spec_from_file_location('output_metrics',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
a=np.zeros((2,3,4),dtype='<f2');b=a.copy()
r=m.metrics(a,b);assert r['bit_identical_fraction']==1 and r['all_finite_components']['rmse']==0
b[1,2,0]=2
r=m.metrics(a,b);assert r['differing_half_components']==1 and r['differing_rgb_pixels']==1
assert r['first_different_component_y_x_channel']==[1,2,0]
assert r['all_finite_components']['max_abs']==2 and abs(r['all_finite_components']['rmse']-np.sqrt(4/24))<1e-12
assert r['alpha_finite_components']['max_abs']==0 and r['candidate_rgb_outside_0_1_finite']==1
b=a.copy();b.view('<u2')[0,0,0]=0x8000
r=m.metrics(a,b);assert r['differing_half_components']==1 and r['finite_numeric_equal_components']==24
assert r['finite_half_representation_steps_max']==1 and r['candidate_special_values']['negative_zero']==1
b=a.copy();b[0,0,:]=[np.nan,np.inf,-np.inf,0]
r=m.metrics(a,b);assert r['nonfinite_pair_components_excluded_from_error']==3
assert r['candidate_special_values']=={'nan':1,'positive_inf':1,'negative_inf':1,'negative_zero':0}
a[:]=np.nan;b[:]=np.nan;r=m.metrics(a,b)
assert r['all_finite_components']['mean_abs'] is None and r['nonfinite_pair_components_excluded_from_error']==24
try:m.metrics(a.astype(np.float32),b)
except ValueError:pass
else:raise AssertionError('wrong dtype admitted')
try:m.metrics(a,b[:,:2])
except ValueError:pass
else:raise AssertionError('shape mismatch admitted')
print('PASS exact, changed RGB, alpha isolation, signed zero, NaN/Inf, empty finite domain, dtype/shape rejection; no model/GPU')
