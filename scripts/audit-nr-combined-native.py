#!/usr/bin/env python3
"""Reuse the reviewed native auditor with an explicit combined-case reference provider."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--peer-root',type=Path,required=True);p.add_argument('--matrix',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];base=repo/'scripts/audit-nr-half-fma-regression.py'
    assert sha(base)=='1b002008ef0d6993bebd6a2ada9d763bcdcf7869497b1d3375e8d4605da251ff','reviewed audit engine changed'
    spec=importlib.util.spec_from_file_location('reviewed_native_audit',base);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    def combined_reference(root,name):
        assert name=='combined3'
        r=m.sealed(root/'build/nr_combined_sequence/packet-audit-v1.json')
        assert r['receipt_root_sha256']=='8fc7129e7dfa5b2457707796319bbc57c0706cf3ddb32f99fe7ae2a544fa4a06'
        assert r['RTX_matrix_root']=='45c6aedceb8a48eefd79961d37d657c3dfdeb973ca8fa9d91b2401ce6a3c5553'
        assert r['result']=='pass_reference_packet_only' and not r['full_model_Mac_parity']
        assert r['input_hashes']==['81ec1de5d1e76e6b55c4fd987b3732180c9720919fbd760dd5b4e8aa8cdd0342','d2db0ead99a6307a6504afed2d49867f0421e3665bb397bed8277ae4f2046d06','af31801008df2fbb537f365265271e716b26339b4c5a2fa2069c01c1a949e8b8']
        assert r['motion_sha256']=='776bec76c8d7fe489da918b80f9178d135acecc88b977212e43c370bbb0cd3aa'
        return r,3,[1,0,0],r['input_hashes'],r['motion_sha256'],True
    # Only the CPU comparison configuration changes. The reviewed native
    # provenance, state, byte, coverage, poison and source checks are reused.
    m.configuration=combined_reference
    assert not a.output.exists()
    try:
        r=m.audit(a.peer_root.resolve(),a.matrix.resolve(),'combined3')
        r['schema']='neuralstudio.nr.combined-native-independent-audit.v1'
        r['result']='pass_bounded_combined_image_motion_parity'
        r['packet_root']=r['RTX_matrix_root']
        r['RTX_matrix_root']='45c6aedceb8a48eefd79961d37d657c3dfdeb973ca8fa9d91b2401ce6a3c5553'
        r['limits'][0]='One three-frame original-to-checker-to-black sequence with the fixed signed-spatial motion field; not arbitrary changing motion or sequence length.'
        r['base_verifier_sha256']=r.pop('verifier_sha256')
    except Exception as e:r={'schema':'neuralstudio.nr.combined-native-independent-audit.v1','result':'fail','error':str(e)}
    r['verifier_sha256']=sha(Path(__file__));r['receipt_root_sha256']=canonical(r)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k not in ('artifacts','runs')},indent=2))
    if r['result']=='fail':raise SystemExit(3)


if __name__=='__main__':main()
