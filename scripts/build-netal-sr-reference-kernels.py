#!/usr/bin/env python3
"""Build the retained translated Reference body kernels; input rights remain under review."""
import argparse,hashlib,json,subprocess,tempfile
from pathlib import Path
from netal_apple_build import add_metal_target_arguments, metal_toolchain
ROOT=Path(__file__).resolve().parents[1]
STAGES=['enc0','enc1','enc2','enc3','enc4','dec5','dec4','dec3','dec2','dec1','dec0']


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=STAGES,action='append');add_metal_target_arguments(parser);args=parser.parse_args();output=args.output.absolute()
    if output.exists() or output.is_symlink():parser.error('Output exists')
    selected=args.stage or STAGES
    if len(selected)!=len(set(selected)):parser.error('Duplicate stage')
    try: compiler,linker,target_flags,target_receipt=metal_toolchain(args.sdk,args.deployment_target)
    except ValueError as error: parser.error(str(error))
    output.parent.mkdir(parents=True,exist_ok=True);records=[]
    with tempfile.TemporaryDirectory(prefix='.sr-reference-',dir=output.parent) as temporary:
        stage_dir=Path(temporary)/'libraries';stage_dir.mkdir()
        for stage in selected:
            name=f'dltss_pwin_{stage}_layer';source=ROOT/'src/sr_kernels/reference'/(name+'.metal');air=Path(temporary)/(name+'.air');library=stage_dir/(name+'_exact.metallib')
            flags=['-fmetal-math-mode=safe','-fmetal-math-fp32-functions=precise']
            subprocess.run([str(compiler),*target_flags,*flags,'-c',str(source),'-o',str(air)],check=True)
            subprocess.run([str(linker),str(air),'-o',str(library)],check=True)
            records.append({'stage':stage,'entry':name,'source':str(source.relative_to(ROOT)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'library':library.name,'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),'flags':flags,'input_rights':'review-required'})
            print('compiled='+stage,flush=True)
        receipt={'schema':'netal.sr.reference-kernels.v1',**target_receipt,'compiler':subprocess.check_output([str(compiler),'--version'],text=True).strip(),'kernels':records,'complete_runtime':False,'inference_validated':False,'publication_clearance':False}
        (stage_dir/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');stage_dir.rename(output)
    print('result=pass kernels='+str(len(records)))


if __name__=='__main__':main()
