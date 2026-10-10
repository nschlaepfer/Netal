#!/usr/bin/env python3
"""Prepare read-only history observation plus two-call encoder boundary snapshots."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def once(s,a,b):
    if s.count(a)!=1:raise ValueError('source anchor changed '+a[:80])
    return s.replace(a,b)
def history_host(s):
    s=once(s,'    const uint16_t output_poison=output_host[0];', '''    const uint16_t output_poison=output_host[0];
    using HistoryReadback=int(WINAPI*)(unsigned,unsigned,void*,size_t);
    const auto history_readback=LoadFunction<HistoryReadback>(cuda_driver,"NeuralStudioCudaCaptureHistory");
    if(!history_readback)throw std::runtime_error("history capture export unavailable");
    std::vector<uint16_t> history_host(output_host.size());
    auto observe_history=[&](unsigned after){
      if(history_readback(NrEvaluationIndex,after,history_host.data(),history_host.size()*2)!=0 ||
         !WriteIoArtifact(after?"history-after.rgba16le":"history-before.rgba16le",history_host.data(),history_host.size()*2))
        throw std::runtime_error("read-only history capture failed");
    };''')
    s=once(s,'    set_trace_phase("evaluate");','    observe_history(0);\n    set_trace_phase("evaluate");')
    s=once(s,'    PrintResult("runtime.evaluate_result", evaluate_result);','    PrintResult("runtime.evaluate_result", evaluate_result);\n    if(NVSDK_NGX_SUCCEED(evaluate_result))observe_history(1);')
    return s
def sampler_trace(proxy):
    start=proxy.index('extern "C" __declspec(dllexport) CUresult WINAPI cuTexObjectCreate(')
    end=proxy.index('\nextern "C" BOOL WINAPI DllMain',start)
    section=proxy[start:end]
    section=once(section,'  char fields[192]{};', '''  // Prefix layout matches the CUDA_TEXTURE_DESC used by the pinned harness:
  // addressMode[3], filterMode, flags; all five fields are 32-bit.
  uint32_t sampler[5]{};
  if(texture!=nullptr)std::memcpy(sampler,texture,sizeof(sampler));
  char fields[320]{};''')
    section=once(section,'"\\"object\\":%llu,\\"array_id\\":%ld,\\"result\\":%d",', '"\\"object\\":%llu,\\"array_id\\":%ld,\\"address_modes\\":[%u,%u,%u],\\"filter_mode\\":%u,\\"flags\\":%u,\\"result\\":%d",')
    section=once(section,'array != nullptr ? array->id : 0, result);','array != nullptr ? array->id : 0, sampler[0],sampler[1],sampler[2],sampler[3],sampler[4],result);')
    return proxy[:start]+section+proxy[end:]
def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--post-boundary',action='store_true');a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    parent=json.loads((a.parent/'source-receipt.json').read_text());b=dict(parent);h=b.pop('receipt_root_sha256');assert canonical(b)==h=='08e67d7bded75e9f15e69db00f940cd45142137123573a471427f15448601130'
    for n,v in parent['files'].items():assert sha(a.parent/n)==v
    assert not a.output.exists();shutil.copytree(a.parent,a.output,ignore=shutil.ignore_patterns('source-receipt.json'))
    s=history_host((a.parent/'color_cases_trace.cpp').read_text());(a.output/'color_cases_trace.cpp').write_text(s)
    proxy=(a.parent/'prefix_proxy.cpp').read_text();proxy=once(proxy,'if (launch_id != 2 && launch_id != 24) return;','if (launch_id != 2 && launch_id != 24 && launch_id != 158 && launch_id != 180) return;')
    if a.post_boundary:proxy=once(proxy,'&& launch_id != 180) return;','&& launch_id != 180 && launch_id != 310) return;')
    proxy=sampler_trace(proxy);proxy+='\n#include "nr_history_readback.inl"\n';(a.output/'prefix_proxy.cpp').write_text(proxy)
    shutil.copyfile(repo/'src/nr_probes/nr_history_readback.inl',a.output/'nr_history_readback.inl')
    cap=(a.parent/'capture-color.ps1').read_text()
    cap=once(cap,"if ($CaptureSnapshots -or $Fp16Analysis) { throw 'Two-frame diagnostic is native FP8, no snapshots' }","if (-not $CaptureSnapshots -or $Fp16Analysis) { throw 'Native FP8 history capture requires bounded snapshots' }")
    for x,y in [('$counts.snapshots -ne 2','$counts.snapshots -ne 4'),('$snapshotFiles.Count -ne 2','$snapshotFiles.Count -ne 4'),('Expected 2 prefix pool snapshots','Expected 4 bounded history-chain pool snapshots'),('Expected 2 prefix snapshot delta files','Expected 4 bounded history-chain snapshot delta files'),('snapshot_launch_limit = 0','snapshot_launch_limit = 180'),('snapshot_launch_ids = @()','snapshot_launch_ids = @(2,24,158,180)')]:cap=once(cap,x,y)
    cap=once(cap,"    'frame-1-output.rgba16le',","    'history-before.rgba16le',\n    'history-after.rgba16le',\n    'frame-1-history-before.rgba16le',\n    'frame-1-history-after.rgba16le',\n    'frame-1-output.rgba16le',")
    cap=once(cap,'$failedCalls = @($events | Where-Object {', '''$historyEvents=@($events | Where-Object { $_.call -eq 'history_readback' })
if ($historyEvents.Count -ne 4) { throw 'Expected four read-only history observations' }
for ($i=0;$i -lt 4;$i++) {
  $e=$historyEvents[$i]; if ($e.frame -ne [math]::Floor($i/2) -or $e.after -ne ($i%2) -or $e.array_id -ne 1 -or $e.bytes -ne 16588800 -or $e.result -ne 0) { throw 'History event contract differs' }
}
if ((Get-FileHash -LiteralPath (Join-Path $ArtifactDirectory 'history-after.rgba16le') -Algorithm SHA256).Hash -ne (Get-FileHash -LiteralPath (Join-Path $ArtifactDirectory 'frame-1-history-before.rgba16le') -Algorithm SHA256).Hash) { throw 'History mutated between completed evaluations' }
$failedCalls = @($events | Where-Object {''')
    if a.post_boundary:
        for old,new in [('$counts.snapshots -ne 4','$counts.snapshots -ne 5'),('$snapshotFiles.Count -ne 4','$snapshotFiles.Count -ne 5'),('Expected 4 bounded history-chain','Expected 5 bounded history-chain'),('snapshot_launch_limit = 180','snapshot_launch_limit = 310'),('snapshot_launch_ids = @(2,24,158,180)','snapshot_launch_ids = @(2,24,158,180,310)')]:
            if old=='Expected 4 bounded history-chain':
                assert cap.count(old)==2;cap=cap.replace(old,new)
            else:cap=once(cap,old,new)
    (a.output/'capture-color.ps1').write_text(cap)
    run=(a.parent/'run-nr-reset-sequence.ps1').read_text();run=once(run,"'-ArmRealExecution','-ArtifactDirectory'","'-ArmRealExecution','-CaptureSnapshots','-ArtifactDirectory'");(a.output/'run-nr-reset-sequence.ps1').write_text(run)
    r={'schema':'neuralstudio.nr.history-capture-source.v1','parent_source_root':h,'snapshot_launch_ids':[2,24,158,180],'history_array_id':1,'history_shape':[1920,1080,4],'history_format':'RGBA16F','history_readback_direction':'device_to_host_only','expected_bytes_staged':False,'runtime_modified':False,'executed':False,'remote_compile_validated':False,'files':{str(p.relative_to(a.output)):sha(p) for p in sorted(a.output.rglob('*')) if p.is_file()},'scope':'Unexecuted observation prototype. Four bounded pool snapshots plus before/after internal history. Must preserve existing [1,1] and [1,0] output signatures and repeat-process/poison checks before use as a temporal reference.'}
    if a.post_boundary:r['snapshot_launch_ids'].append(310);r['scope']=r['scope'].replace('Four bounded','Five bounded')
    r['receipt_root_sha256']=canonical(r);(a.output/'source-receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(r['receipt_root_sha256'])
if __name__=='__main__':main()
