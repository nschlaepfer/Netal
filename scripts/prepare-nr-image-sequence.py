#!/usr/bin/env python3
"""Prepare a three-call original->checker->black reference, with static control."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

PARENT='d5da3e2d6d3ff2ad2545bc1b50bb181ff389bdf8ee663cdb8f2800dcf00543e8'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def once(s,a,b):
    if s.count(a)!=1:raise ValueError('changed source anchor '+a[:90])
    return s.replace(a,b)
def host(s):
    s=once(s,'    std::vector<uint16_t> color_host;','    unsigned initial_color_case=0;\n    std::vector<uint16_t> color_host;')
    s=once(s,"      const unsigned case_id=static_cast<unsigned>(case_text[0]-'0');","      const unsigned case_id=static_cast<unsigned>(case_text[0]-'0');initial_color_case=case_id;")
    s=once(s,'    for(NrEvaluationIndex=0;NrEvaluationIndex<3;++NrEvaluationIndex){','''    char sequence_text[16]{};
    const DWORD sequence_len=GetEnvironmentVariableA("NEURALSTUDIO_NR_IMAGE_SEQUENCE",sequence_text,sizeof(sequence_text));
    if(sequence_len==0 || sequence_len>=sizeof(sequence_text) || (std::string(sequence_text)!="static" && std::string(sequence_text)!="mixed"))throw std::runtime_error("explicit static/mixed image sequence required");
    const bool mixed=std::string(sequence_text)=="mixed";
    if(mixed && initial_color_case!=0)throw std::runtime_error("mixed sequence starts with original image");
    for(NrEvaluationIndex=0;NrEvaluationIndex<3;++NrEvaluationIndex){
      const unsigned current_color_case=mixed?(NrEvaluationIndex==0?0u:(NrEvaluationIndex==1?2u:1u)):initial_color_case;
      if(NrEvaluationIndex>0){
        NrColorCase(color_host,kWidth,kHeight,current_color_case);
        if(!CopyHostToArray(cu_memcpy_2d,color,color_host.data(),static_cast<size_t>(kWidth)*8,kHeight) ||
           !WriteIoArtifact("input-color.rgba16le",color_host.data(),color_host.size()*2))throw std::runtime_error("per-frame original image update failed");
      }
      std::cout<<"runtime.sequence_color_case="<<NrEvaluationIndex<<":"<<current_color_case<<'\\n';''')
    return once(s,'rgba16-color-cases-v1_depth-bands-v1_motion-zero','rgba16-image-sequence-v1_depth-bands-v1_motion-zero')
def runner(s):
    if '$ImageSequence' in s:raise ValueError('already transformed')
    s=once(s,'[ValidateRange(0,2)][int]$ColorCase=0,',"[ValidateRange(0,2)][int]$ColorCase=0,[ValidateSet('static','mixed')][string]$ImageSequence='static',")
    s=once(s,'$env:NEURALSTUDIO_NR_COLOR_CASE=[string]$ColorCase;','$env:NEURALSTUDIO_NR_IMAGE_SEQUENCE=$ImageSequence;$env:NEURALSTUDIO_NR_COLOR_CASE=[string]$ColorCase;')
    return once(s,'color_case=$ColorCase;','color_case=$ColorCase;image_sequence=$ImageSequence;')
def capture(s):
    assert s.count('rgba16-color-cases-v1_depth-bands-v1_motion-zero')==2
    s=s.replace('rgba16-color-cases-v1_depth-bands-v1_motion-zero','rgba16-image-sequence-v1_depth-bands-v1_motion-zero')
    s=once(s,"    'frame-1-output.rgba16le',","    'frame-1-input-color.rgba16le',\n    'frame-2-input-color.rgba16le',\n    'frame-1-output.rgba16le',")
    s=once(s,'    if ((Get-FileHash -LiteralPath (Join-Path $ArtifactDirectory $pair[0]) -Algorithm SHA256).Hash -ne',"    $beforeName=if($pair[0] -eq 'input-color.rgba16le'){$prefix+$pair[0]}else{$pair[0]}\n    if ((Get-FileHash -LiteralPath (Join-Path $ArtifactDirectory $beforeName) -Algorithm SHA256).Hash -ne")
    return once(s,'    sequence_frames = 3','    sequence_frames = 3\n    image_sequence = $env:NEURALSTUDIO_NR_IMAGE_SEQUENCE')
def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    parent=json.loads((a.parent/'source-receipt.json').read_text());b=dict(parent);h=b.pop('receipt_root_sha256');assert canonical(b)==h==PARENT
    for n,d in parent['files'].items():assert sha(a.parent/n)==d,n
    assert not a.output.exists();shutil.copytree(a.parent,a.output,ignore=shutil.ignore_patterns('source-receipt.json'))
    for n,f in [('color_cases_trace.cpp',host),('run-nr-reset-sequence.ps1',runner),('capture-color.ps1',capture)]:(a.output/n).write_text(f((a.parent/n).read_text()))
    r={'schema':'neuralstudio.nr.image-sequence-source.v1','parent_source_root':h,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip(),'frames':3,'image_sequences':{'static':'initial recipe repeated','mixed':[0,2,1]},'depth_motion_descriptors':'unchanged; zero motion','proxy_model_unchanged':True,'input_color_updated_between_evaluations_only':True,'expected_bytes_staged':False,'executed':False,'files':{str(p.relative_to(a.output)):sha(p) for p in sorted(a.output.rglob('*')) if p.is_file()}}
    r['receipt_root_sha256']=canonical(r);(a.output/'source-receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(r['receipt_root_sha256'])
if __name__=='__main__':main()
