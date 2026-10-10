#!/usr/bin/env python3
"""Derive a two-evaluation, one-feature-handle RTX diagnostic from sealed source."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

PARENT='ce4f469852de91b8ede849a355770c9d10be5c8273c198426d374aaded411a1e'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def once(s,old,new):
    if s.count(old)!=1:raise ValueError('changed parent anchor '+old[:80])
    return s.replace(old,new)
def host_sequence(s):
    s=once(s,'#include <cstdint>','#include <algorithm>\n#include <cstdint>')
    s=once(s,'bool WriteIoArtifact(const char* filename,','static unsigned NrEvaluationIndex=0;\nbool WriteIoArtifact(const char* filename,')
    s=once(s,'const std::string path = std::string(directory) + "\\\\" + filename;','const std::string path = std::string(directory) + "\\\\" + (NrEvaluationIndex ? "frame-1-" : "") + filename;')
    start='    set_trace_phase("evaluate");'
    s=once(s,start,'''    char reset_text[8]{};
    const DWORD reset_len=GetEnvironmentVariableA("NEURALSTUDIO_NR_SECOND_RESET",reset_text,sizeof(reset_text));
    if(reset_len!=1||(reset_text[0]!='0'&&reset_text[0]!='1'))throw std::runtime_error("explicit second reset required");
    if(!real_cuda||output_host.empty())throw std::runtime_error("sequence requires real CUDA output");
    const uint16_t output_poison=output_host[0];
    for(NrEvaluationIndex=0;NrEvaluationIndex<2;++NrEvaluationIndex){
      const int reset=NrEvaluationIndex ? reset_text[0]-'0' : 1;
      parameters.SetI32("DLSSNR.Reset",reset);
      std::cout<<"runtime.sequence_frame="<<NrEvaluationIndex<<" reset="<<reset<<'\\n';
      std::fill(output_host.begin(),output_host.end(),output_poison);
      if(!CopyHostToArray(cu_memcpy_2d,output,output_host.data(),static_cast<size_t>(kWidth)*8,kHeight))
        throw std::runtime_error("sequence output re-poison failed");
'''+start)
    s=once(s,'        std::cout << "runtime.output_readback_bytes="','        if(NrEvaluationIndex==0){\n        std::cout << "runtime.output_readback_bytes="')
    s=once(s,'        output_readback_valid = statistics.nonfinite_elements == 0;','''        }
        std::cout<<"runtime.sequence_nonfinite="<<NrEvaluationIndex<<":"<<statistics.nonfinite_elements<<'\\n';
        output_readback_valid = statistics.nonfinite_elements == 0;''')
    errors='''    if (NVSDK_NGX_FAILED(evaluate_result)) {
      std::cerr << "error=feature18_evaluation_failed\\n";
      return 9;
    }
    if (!output_readback_valid) {
      std::cerr << "error=feature18_output_readback_invalid\\n";
      return 10;
    }'''
    s=once(s,errors,'')
    s=once(s,'    set_trace_phase("release");',errors+'''
    }
    std::cout<<"runtime.sequence_frames=2\\n";
    set_trace_phase("release");''')
    return s
def capture_sequence(s):
    s=once(s,"$ErrorActionPreference = 'Stop'","$ErrorActionPreference = 'Stop'\nif ($CaptureSnapshots -or $Fp16Analysis) { throw 'Two-frame diagnostic is native FP8, no snapshots' }")
    for old,new in [('$counts.launches -ne 156','$counts.launches -ne 312'),('$counts.argument_bindings -ne 156','$counts.argument_bindings -ne 312'),('$argumentDumps.Count -ne 156','$argumentDumps.Count -ne 312')]:s=once(s,old,new)
    s=once(s,"    'runtime.evaluate_result=0x1',","    'runtime.evaluate_result=0x1',\n    'runtime.sequence_frames=2',\n    'runtime.sequence_frame=0 reset=1',\n    'runtime.sequence_nonfinite=0:0',\n    'runtime.sequence_nonfinite=1:0',")
    s=once(s,'if ($Fp16Analysis) {\n    Require-OutputLine', 'Require-OutputLine ("runtime.sequence_frame=1 reset="+$env:NEURALSTUDIO_NR_SECOND_RESET)\nif ($Fp16Analysis) {\n    Require-OutputLine')
    s=once(s,"    'output.rgba16le',","    'frame-1-output.rgba16le',\n    'frame-1-input-color-after.rgba16le',\n    'frame-1-input-depth-after.r32le',\n    'frame-1-input-motion-after.rg16le',\n    'output.rgba16le',")
    s=once(s,'$outputHash = (Get-FileHash', '''foreach ($prefix in @('', 'frame-1-')) {
  foreach ($pair in @(@('input-color.rgba16le','input-color-after.rgba16le'),@('input-depth.r32le','input-depth-after.r32le'),@('input-motion.rg16le','input-motion-after.rg16le'))) {
    if ((Get-FileHash -LiteralPath (Join-Path $ArtifactDirectory $pair[0]) -Algorithm SHA256).Hash -ne (Get-FileHash -LiteralPath (Join-Path $ArtifactDirectory ($prefix+$pair[1])) -Algorithm SHA256).Hash) { throw 'Sequence input changed' }
  }
}
$secondOutputHash=(Get-FileHash -LiteralPath (Join-Path $ArtifactDirectory 'frame-1-output.rgba16le') -Algorithm SHA256).Hash.ToLowerInvariant()
$outputHash = (Get-FileHash''')
    s=once(s,'    snapshot_launch_limit = 24','    sequence_frames = 2\n    second_output_sha256 = $secondOutputHash\n    second_reset = [int]$env:NEURALSTUDIO_NR_SECOND_RESET\n    same_feature_handle = $true\n    output_repoisoned_each_evaluation = $true\n    snapshot_launch_limit = 0')
    s=once(s,'    snapshot_launch_ids = @(2,24)','    snapshot_launch_ids = @()')
    return s
def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    parent=json.loads((a.parent/'source-receipt.json').read_text());b=dict(parent);h=b.pop('receipt_root_sha256');assert canonical(b)==h==PARENT
    for name,digest in parent['files'].items():assert sha(a.parent/name)==digest,name
    host=host_sequence((a.parent/'color_cases_trace.cpp').read_text());capture=capture_sequence((a.parent/'capture-color.ps1').read_text())
    assert not a.output.exists();shutil.copytree(a.parent,a.output,ignore=shutil.ignore_patterns('source-receipt.json','run-nr-color-job.ps1'))
    (a.output/'color_cases_trace.cpp').write_text(host);(a.output/'capture-color.ps1').write_text(capture)
    shutil.copyfile(repo/'scripts/run-nr-reset-sequence.ps1',a.output/'run-nr-reset-sequence.ps1')
    r={'schema':'neuralstudio.nr.reset-sequence-source.v1','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'parent_source_root':h,'sequence_frames':2,'supported_reset_sequences':[[1,1],[1,0]],'same_feature_handle':True,'output_repoisoned_each_evaluation':True,'snapshot_capture':False,'expected_bytes_staged':False,'runtime_modified':False,'executed':False,'scope':'Same image, same model instance, two synchronized evaluations with per-evaluation input immutability and output re-poison. Initial [1,1] diagnostic precedes [1,0]. No Metal or temporal acceptance claim.','files':{str(p.relative_to(a.output)):sha(p) for p in sorted(a.output.rglob('*')) if p.is_file()}}
    r['receipt_root_sha256']=canonical(r);(a.output/'source-receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(r['receipt_root_sha256'])
if __name__=='__main__':main()
