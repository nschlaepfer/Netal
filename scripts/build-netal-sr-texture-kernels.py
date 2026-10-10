#!/usr/bin/env python3
"""Build the model-independent SR texture converter and existing format extension."""
import argparse,hashlib,json,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCES={'dlss_sr_live_texture_convert':('src/sr_kernels/netal_live_texture_convert.metal',['-std=metal3.1','-fno-fast-math']),
         'netal_texture_import_ext':('src/macos_host/netal_texture_import_ext.metal',['-std=metal3.1','-O2'])}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();output=args.output.absolute()
    if output.exists() or output.is_symlink():parser.error('Output exists')
    compiler=Path(subprocess.check_output(['xcrun','--find','metal'],text=True).strip());linker=compiler.with_name('metallib')
    if not linker.is_file():linker=Path(subprocess.check_output(['xcrun','--find','metallib'],text=True).strip())
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.texture-kernels-',dir=output.parent) as temporary:
        stage=Path(temporary)/'libraries';stage.mkdir();records=[]
        for name,(relative,flags) in SOURCES.items():
            source=ROOT/relative;air=Path(temporary)/(name+'.air');library=stage/(name+'.metallib')
            subprocess.run([str(compiler),*flags,'-c',str(source),'-o',str(air)],check=True);subprocess.run([str(linker),str(air),'-o',str(library)],check=True)
            records.append({'source':relative,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'library':library.name,'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),'flags':flags})
        (stage/'build-receipt.json').write_text(json.dumps({'schema':'netal.sr.texture-kernels.v1','kernels':records,'compiler':subprocess.check_output([str(compiler),'--version'],text=True).strip(),'model_inputs':0,'gameplay_validated':False},indent=2)+'\n');stage.rename(output)
    print('result=pass libraries='+str(len(records)))

if __name__=='__main__':main()
