#!/usr/bin/env python3
"""Freeze a varied-color RTX harness; does not build, stage or execute it.

The original graph/runtime and depth/motion/reset descriptors stay unchanged.
Output poison tests are not internal pool write-coverage evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError('parent source changed: '+old[:80])
    return text.replace(old,new)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    a.output.mkdir(parents=True,exist_ok=False)
    source=repo/'src/macos_host/cuda_trace/dlssnr_cuda_create_trace.cpp';text=source.read_text()
    text=replace_once(text,'#include <vector>','#include <vector>\n#include "input_cases.h"')
    marker='''      output_host.assign(static_cast<size_t>(kWidth) * kHeight * 4, 0);'''
    replacement='''      char case_text[8]{}, poison_text[8]{};
      const DWORD case_len=GetEnvironmentVariableA("NEURALSTUDIO_NR_COLOR_CASE",case_text,sizeof(case_text));
      const DWORD poison_len=GetEnvironmentVariableA("NEURALSTUDIO_NR_OUTPUT_POISON",poison_text,sizeof(poison_text));
      if(case_len!=1 || case_text[0]<'0' || case_text[0]>'3' || poison_len!=2 ||
         (std::string(poison_text)!="a5" && std::string(poison_text)!="5a"))
        throw std::runtime_error("explicit NR color case and output poison required");
      const unsigned case_id=static_cast<unsigned>(case_text[0]-'0');
      const uint16_t poison=std::string(poison_text)=="a5"?0xa5a5u:0x5a5au;
      NrColorCase(color_host,kWidth,kHeight,case_id);
      output_host.assign(static_cast<size_t>(kWidth) * kHeight * 4, poison);
      std::cout << "runtime.color_case=" << case_id << '\\n';
      std::cout << "runtime.process_id=" << GetCurrentProcessId() << '\\n';
      std::cout << "runtime.output_poison=" << poison_text << '\\n';'''
    text=replace_once(text,marker,replacement)
    text=replace_once(text,'rgba16-tile-v1_depth-bands-v1_motion-zero\\n','rgba16-color-cases-v1_depth-bands-v1_motion-zero\\n')
    (a.output/'color_cases_trace.cpp').write_text(text)
    shutil.copyfile(repo/'src/nr_pre/input_cases.h',a.output/'input_cases.h')
    capture=(repo/'scripts/capture-rtx-feature18-cuda-oracle.ps1').read_text()
    recipe='rgba16-tile-v1_depth-bands-v1_motion-zero'
    if capture.count(recipe)!=2:raise ValueError('capture recipe changed')
    (a.output/'capture-color.ps1').write_text(capture.replace(recipe,'rgba16-color-cases-v1_depth-bands-v1_motion-zero'))
    for name in ('build-nr-color-job.ps1','run-nr-color-job.ps1'):shutil.copyfile(repo/'scripts'/name,a.output/name)
    sdk=repo.parent/'work/NVIDIA-DLSS/include';(a.output/'include').mkdir()
    # Only public compile-time SDK headers, no runtime/weights/expected outputs.
    for header in sorted(sdk.glob('*.h')):shutil.copyfile(header,a.output/'include'/header.name)
    if not (a.output/'include/nvsdk_ngx.h').is_file():raise ValueError('SDK headers missing')
    receipt={'schema':'neuralstudio.nr.color-case-source.v1','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
      'parent_harness_sha256':sha(source),'case_ids':{'0':'original tile control','1':'black RGB','2':'checker and thin stripes','3':'dense finite half patterns'},
      'width':1920,'height':1080,'reset':1,'depth_and_motion':'unchanged baseline',
      'expected_bytes_staged':False,'runtime_modified':False,'executed':False,
      'files':{str(x.relative_to(a.output)):sha(x) for x in sorted(a.output.rglob('*')) if x.is_file()},
      'scope':'Prepared source only. Baseline case0 must reproduce pinned input and RTX final output across A5/A5/5A/5A before new cases. No native Mac or internal-pool write-coverage claim.'}
    receipt['receipt_root_sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (a.output/'source-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['receipt_root_sha256'])

if __name__=='__main__':main()
