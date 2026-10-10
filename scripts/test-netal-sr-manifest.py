#!/usr/bin/env python3
"""Cross-language inventory integrity checks with synthetic, non-model files."""
from pathlib import Path
import hashlib,json,subprocess,tempfile
from netal_sr_runtime_manifest import SCHEMA,ledger_root
root=Path(__file__).resolve().parents[1];binary=root/'build/netal-tools/sr-manifest-probe';binary.parent.mkdir(parents=True,exist_ok=True)
subprocess.run(['xcrun','clang++','-std=c++20','-fobjc-arc','-Wall','-Wextra','-Werror','-framework','Foundation',str(root/'src/macos_host/netal_sr_manifest_probe.mm'),'-o',str(binary)],check=True)
subprocess.run(['codesign','--force','--sign','-',str(binary)],check=True,capture_output=True)
with tempfile.TemporaryDirectory() as temporary:
    folder=Path(temporary);asset=folder/'asset.bin';asset.write_bytes(b'original');rows=[{'path':'asset.bin','bytes':8,'sha256':hashlib.sha256(asset.read_bytes()).hexdigest()}]
    original={'schema':SCHEMA,'model':'DLSS_SR_Feature1_Quality_K','files':rows,'asset_root_sha256':ledger_root(rows)}
    def check(manifest,expected,full=False):
        (folder/'manifest.json').write_text(json.dumps(manifest))
        result=subprocess.run([str(binary),str(folder),*([] if full else ['--inventory-only'])],capture_output=True,text=True)
        assert (result.returncode==0)==expected,result.stdout
    check(original,True)
    check(original,False,full=True) # integrity of a toy file cannot establish an SR model contract
    bad=dict(original,asset_root_sha256='0'*64);check(bad,False)
    bad=dict(original,files=rows+rows);check(bad,False)
    asset.write_bytes(b'changed!');check(original,False)
    changed=[dict(rows[0],sha256=hashlib.sha256(asset.read_bytes()).hexdigest())]
    check(dict(original,files=changed),False) # updating only a row must not preserve the old root
    asset.write_bytes(b'original');(folder/'extra').write_text('unexpected');check(original,False);(folder/'extra').unlink()
    asset.rename(folder/'saved');asset.symlink_to(folder/'saved');check(original,False);asset.unlink();(folder/'saved').rename(asset)
    for path in ['../outside','/absolute','a//b','a/./b','a/../b','capture/data.bin','manifest.json']:
        try:ledger_root([dict(rows[0],path=path)])
        except ValueError:pass
        else:raise AssertionError(path)
    check(original,True)
print('Manifest inventory checks passed; model contract is tested separately with explicit local inputs')
