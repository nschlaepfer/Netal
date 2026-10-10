#!/usr/bin/env python3
"""Read-only tests of Fable's recipe preflight/status helpers. Never calls run()."""
import argparse
import builtins
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from unittest.mock import patch


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--peer',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=a.peer/'scripts/nr-run-recipe.py';recipe=a.peer/'spec/nr/recipes/images3.json'
    source_pin=sha(source);recipe_pin=sha(recipe)
    spec=importlib.util.spec_from_file_location('peer_preflight',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    base=json.loads(recipe.read_text());rows=[]
    assert not m.preflight(base),'baseline recipe does not preflight'
    def check(name,edit):
        r=copy.deepcopy(base);edit(r)
        try:errors=m.preflight(r);rows.append({'case':name,'expected':'rejected','observed':'rejected' if errors else 'accepted','errors':errors})
        except Exception as e:rows.append({'case':name,'expected':'rejected','observed':'unstructured_exception','exception':type(e).__name__+': '+str(e)})
    check('missing-harness-pin',lambda r:r.pop('harness_executable_sha256'))
    check('missing-descriptor-pin',lambda r:r.pop('descriptor_sha256'))
    check('missing-input-pin',lambda r:r['frames'][1].pop('input_sha256'))
    check('missing-reference-pin',lambda r:r['frames'][1].pop('reference_sha256'))
    check('nonbinary-reset',lambda r:r.__setitem__('reset_sequence','1,2,0'))
    check('duplicate-replica-tag',lambda r:r['replicas'][1].__setitem__('tag',r['replicas'][0]['tag']))
    check('missing-encoder-pin',lambda r:r.pop('encoder_object_sha256'))
    check('invalid-schema',lambda r:r.__setitem__('schema','unrecognized.schema'))
    # A fully hashed virtual file with a nonzero byte after the first page.
    # Only this one path is mocked; no file is created or mutated.
    motion=bytes(4096)+b'\x01'+bytes(1920*1080*4-4097)
    virtual=str(a.output.parent/'virtual-motion-not-created.rg16le')
    r=copy.deepcopy(base);r['motion']={'file':virtual,'sha256':hashlib.sha256(motion).hexdigest()};r['encoder_sampler_variant']=0
    original_exists=m.os.path.exists;original_size=m.os.path.getsize
    def opened(path,*args,**kwargs):return io.BytesIO(motion) if str(path)==virtual else builtins.open(path,*args,**kwargs)
    with patch.object(m,'open',side_effect=opened,create=True),patch.object(m.os.path,'exists',side_effect=lambda path:True if str(path)==virtual else original_exists(path)),patch.object(m.os.path,'getsize',side_effect=lambda path:len(motion) if str(path)==virtual else original_size(path)):
        errors=m.preflight(r)
    rows.append({'case':'nonzero-motion-after-first-page','expected':'rejected','observed':'rejected' if errors else 'accepted','errors':errors,'virtual_fixture_only':True,'first_nonzero_byte':4096})
    fault=json.loads((a.peer/'build/nr_full/temporal-fault-v5/constant-late-mutation/receipt.json').read_text())
    statuses=[]
    for i in (1,2,3,4,5,6):
        frame=fault['frames'][i]
        try:observed=m.status_of(frame)
        except Exception as e:observed=type(e).__name__+': '+str(e)
        statuses.append({'frame':i,'encoder_code':frame.get('encoder',{}).get('code'),'body_code':frame.get('body',{}).get('code'),'result':frame['result'],'observed_status':observed})
    assert sha(source)==source_pin and sha(recipe)==recipe_pin
    r={'schema':'neuralstudio.nr.recipe-independent-review.v1','source':str(source),'source_sha256':source_pin,'recipe_sha256':recipe_pin,
       'baseline_preflight_passed':True,'mutations':rows,'retained_fault_statuses':statuses,
       'GPU_executed':False,'peer_files_changed':False,'virtual_motion_file_written':False,
       'limits':'Only preflight() and status_of() were called; no recipe run, GPU reservation, or peer source edit.',
       'reviewer_sha256':sha(Path(__file__))}
    r['receipt_root_sha256']=canonical(r);assert not a.output.exists();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))


if __name__=='__main__':main()
