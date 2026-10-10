#!/usr/bin/env python3
"""Seal an assembled local SR runtime; does not supply missing assets or certify gameplay."""
import argparse
from pathlib import Path
from netal_sr_runtime_manifest import seal
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('runtime',type=Path);parser.add_argument('--replace',action='store_true')
args=parser.parse_args()
try:
    result=seal(args.runtime,args.replace)
except (ValueError,OSError) as error:parser.exit(1,str(error)+'\n')
print('result=pass assets='+str(len(result['files']))+' root='+result['asset_root_sha256']+' gameplay_validated=false')
