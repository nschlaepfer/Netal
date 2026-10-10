#!/usr/bin/env python3
"""Compile the available SR boundary sources with explicit provenance and math flags.

This does not provide arithmetic tables or all legacy fallbacks.
A successful build is not a redistribution clearance or inference-validation receipt.
"""
import argparse,hashlib,json,subprocess,sys,tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCES={
 'hiluma_input_poison_a':('src/sr_kernels/hiluma_input_depthreg.metal','translated-source-rights-review-required',['-fmetal-math-mode=safe','-fmetal-math-fp32-functions=precise']),
 'hiluma_output_exact_math_safe':('src/sr_kernels/hiluma_output_depthreg.metal','translated-source-rights-review-required',['-fmetal-math-mode=safe','-fmetal-math-fp32-functions=precise']),
 'hiluma_input_depthinv':('src/sr_kernels/hiluma_input_depthinv_fast.metal','translated-source-rights-review-required',['-ffast-math']),
 'hiluma_input_depthinv_exact':('src/sr_kernels/hiluma_input_depthinv_exact.metal','translated-source-rights-review-required',['-fmetal-math-mode=safe']),
 'netal_copy_exposure':('src/sr_kernels/netal_copy_exposure.metal','project-implementation-with-apache-2.0-helper-attribution', ['-std=metal4.0','-fno-fast-math']),
 'hiluma_output_depthinv_exact':('src/sr_kernels/hiluma_output_depthinv_exact.metal','translated-source-rights-review-required',['-fmetal-math-mode=safe']),
}


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output',required=True,type=Path)
 parser.add_argument('--only',choices=list(SOURCES),action='append')
 args=parser.parse_args();output=args.output.absolute()
 if output.exists() or output.is_symlink():parser.error('output exists; choose a new directory')
 output.parent.mkdir(parents=True,exist_ok=True)
 try:
  metal=Path(subprocess.check_output(['xcrun','--find','metal'],text=True).strip())
  linker=metal.with_name('metallib')
  if not linker.is_file():linker=Path(subprocess.check_output(['xcrun','--find','metallib'],text=True).strip())
  records=[]
  with tempfile.TemporaryDirectory(prefix='.boundary-kernels-',dir=output.parent) as tmp:
   stage=Path(tmp)/'output';stage.mkdir()
   for name in args.only or SOURCES:
    relative,provenance,flags=SOURCES[name];source=ROOT/relative;air=Path(tmp)/(name+'.air');library=stage/(name+'.metallib')
    subprocess.run([str(metal),*flags,'-c',str(source),'-o',str(air)],check=True)
    subprocess.run([str(linker),str(air),'-o',str(library)],check=True)
    records.append({'library':library.name,'source':relative,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                    'flags':flags,'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),'provenance':provenance})
   receipt={'schema':'netal.sr.boundary-kernels.v1','compiler':subprocess.check_output([str(metal),'--version'],text=True).strip(),
            'kernels':records,'complete_runtime':False,'inference_validated':False,'redistribution_cleared':False}
   (stage/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');stage.rename(output)
 except (OSError,subprocess.CalledProcessError) as error:
  print('result=fail '+str(error),file=sys.stderr);return 1
 print('result=pass compiled='+str(len(records))+' complete_runtime=false');return 0


if __name__=='__main__':raise SystemExit(main())
