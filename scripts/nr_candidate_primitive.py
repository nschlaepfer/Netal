"""Explicit allowlist of independently tested encoder arithmetic artifacts."""
import hashlib
import json

LEGACY_ROOT='e478976e560da01d2fb449252b41e5cfbdcc1f0a0ed976274d4b11f74136862f'
FSCALE_ROOT='b4c58e3e5dfc17f9731a772ede468d92c297fd5a5d21a2608043f065db2964b5'
HEADERS={LEGACY_ROOT:'58ba2e7c43e96c39348b685957cf014050b096fdaaccdb1da4e086ccab2409cf',
         FSCALE_ROOT:'b28f1e12f6696ebe513d323ec018f6f86e89745b37f09884913a1f92bae3586b'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def validate_primitive(bundle,manifest):
    root=manifest.get('primitive_root')
    if root not in HEADERS:raise ValueError('unaccepted primitive root')
    for name in ('pre/source/nr_swin/qmma_fixed.metalh','encoder/source/qmma_fixed.metalh'):
        if sha(bundle/name)!=HEADERS[root]:raise ValueError('primitive source is not the tested header: '+name)
    if root==FSCALE_ROOT:
        proof=json.loads((bundle/'primitive-verification.json').read_text());body=dict(proof);actual=body.pop('receipt_root_sha256',None)
        canonical=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if actual!=root or canonical!=root or proof.get('result')!='pass':raise ValueError('primitive verification seal mismatch')
        if proof['files']['source/nr_swin/qmma_fixed.metalh']!=HEADERS[root]:raise ValueError('proof source mismatch')
    return root
