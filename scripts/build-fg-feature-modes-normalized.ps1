param([Parameter(Mandatory=$true)][string]$JobRoot)
$ErrorActionPreference='Stop';$JobRoot=[IO.Path]::GetFullPath($JobRoot)
if(-not $JobRoot.StartsWith('D:\CodexLab\')){throw 'invalid job root'}
$builder=Join-Path $JobRoot 'source\build-fg-nvapi-repeat.py'
$module21=Join-Path $JobRoot 'source\module-000021.ptx.txt';$module21Receipt=Join-Path $JobRoot 'source\module21-receipt.json'
$modes=@('core','hudless','hudless-ui-alpha','hudless-ui-color-alpha','distortion','output-real','disable-output','menu','reset-every-frame','subrect')
$rows=@()
foreach($mode in $modes){
 $a=Join-Path $JobRoot "output\$mode-poison-a5-trace";$b=Join-Path $JobRoot "output\$mode-poison-5a-trace";$out=Join-Path $JobRoot "output\normalized-v2-$mode.json"
 if(-not(Test-Path -LiteralPath $out)){
  & python $builder --trace-a (Join-Path $a 'nvapi-trace.jsonl') --dumps-a (Join-Path $a 'dumps') --volatile-a (Join-Path $a 'nvapi-volatile.jsonl') --trace-b (Join-Path $b 'nvapi-trace.jsonl') --dumps-b (Join-Path $b 'dumps') --volatile-b (Join-Path $b 'nvapi-volatile.jsonl') --module21-ptx $module21 --module21-ptx-receipt $module21Receipt --output $out | Out-Null
  if($LASTEXITCODE -ne 0){throw "normalization failed: $mode"}
 }
 $r=Get-Content $out -Raw|ConvertFrom-Json
 $rows+=[ordered]@{mode=$mode;receipt_root_sha256=$r.receipt_root_sha256;launch_count=$r.handle_normalized_parameters.launch_comparison_count;address_free_exact_count=$r.handle_normalized_parameters.address_free_exact_count;structural_graph_repeat=$r.claims.structural_graph_repeat;address_free_parameters_exact=$r.claims.address_free_parameters_exact;blockers=@($r.blockers)}
}
$receipt=[ordered]@{schema='neuralstudio.fg.feature-modes-normalized.v1';rows=$rows;address_fields_recorded=$false;result='pass'}
$receipt|ConvertTo-Json -Depth 8|Set-Content (Join-Path $JobRoot 'feature-modes-normalized.json') -Encoding UTF8
$receipt|ConvertTo-Json -Compress -Depth 8
