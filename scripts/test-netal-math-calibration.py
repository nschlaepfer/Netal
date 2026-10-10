#!/usr/bin/env python3
"""Check signed corrections, band sharing, rejection and the narrow data-export exception."""
import importlib.util
from pathlib import Path
from netal_sr_calibration import BINARY_FILES, load, validate_blob
from public_source_policy import allowed, content_findings

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('packer',root/'scripts/pack-netal-math-calibration.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
def reject(operation):
    try:operation()
    except ValueError:return
    raise AssertionError('invalid input accepted')
base=[100]*8
rom,hist=module.pack([100,99,101,100]*2,base,'rcp',4)
assert rom==bytes([0b00100100]) and hist=={-1:2,0:4,1:2}
rom,hist=module.pack([98,99,100,101]*2,base,'rsq',4)
assert rom==bytes([0b10000111])*2 and hist[-2]==2
reject(lambda:module.pack([100,99,101,100,101,99,101,100],base,'rcp',4))
reject(lambda:module.pack([98]*8,base,'rcp',4))
reject(lambda:module.pack([103]*8,base,'rsq',4))
reject(lambda:module.pack([100]*4,base,'rsq',4))
files=load(root)
for relative in BINARY_FILES:
    data=(root/relative).read_bytes()
    assert allowed(relative) and content_findings(data,relative=relative)==[]
    changed=bytes([data[0]^1])+data[1:]
    reject(lambda:validate_blob(relative,changed))
    assert content_findings(changed,relative=relative)==['calibration-identity-mismatch']
    assert content_findings(data,relative='src/unapproved.bin')
assert len(files)==2 and not allowed('runtime/arithmetic-calibration/extra.bin')
assert not allowed('src/unreviewed-oracle.cu')
print('Calibration packing, rare -2 corrections, full data identities and export rejection checks passed')
