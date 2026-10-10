#!/usr/bin/env python3
"""Generate the three reconstructed SR math assets; correction ROMs remain separate."""
import argparse,hashlib,json,tempfile
from pathlib import Path
from netal_sr_math import generate


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
 output=args.output.absolute()
 if output.exists() or output.is_symlink():parser.error('output exists; choose a new directory')
 files=generate();output.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='.sr-math-',dir=output.parent) as temporary:
  stage=Path(temporary)/'math';stage.mkdir()
  for name,data in files.items():(stage/name).write_bytes(data)
  receipt={'schema':'netal.sr.generated-math.v1','source_sha256':hashlib.sha256(Path(__file__).with_name('netal_sr_math.py').read_bytes()).hexdigest(),
           'files':{name:{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()} for name,data in files.items()},
           'model_data_included':False,'complete_runtime':False,'remaining':['RCP correction ROM','RSQ correction ROM','general scalar math beyond sampled contracts']}
  (stage/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');stage.rename(output)
 print('result=pass generated=3 complete_runtime=false')


if __name__=='__main__':main()
