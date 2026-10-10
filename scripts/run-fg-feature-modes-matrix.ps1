[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$JobRoot,
  [Parameter(Mandatory=$true)][string]$RunnerDirectory,
  [Parameter(Mandatory=$true)][string]$TraceRunner
)
$ErrorActionPreference='Stop'
$JobRoot=[IO.Path]::GetFullPath($JobRoot)
if(-not $JobRoot.StartsWith('D:\CodexLab\')){throw 'JobRoot must be beneath D:\CodexLab'}
$rows=@(
  @{name='core'},
  @{name='hudless';Hudless=$true},
  @{name='hudless-ui-alpha';Hudless=$true;UiAlpha=$true},
  @{name='hudless-ui-color-alpha';Hudless=$true;UiColorAlpha=$true},
  @{name='distortion';BidirectionalDistortion=$true},
  @{name='output-real';OutputReal=$true},
  @{name='disable-output';DisableInterpolationOutput=$true},
  @{name='menu';MenuDetection=$true},
  @{name='reset-every-frame';ResetEveryFrame=$true},
  @{name='subrect';SubrectX=32;SubrectY=24;SubrectWidth=512;SubrectHeight=288},
  @{name='mfg2-index1';MultiFrameCount=2;MultiFrameIndex=1},
  @{name='mfg2-index2';MultiFrameCount=2;MultiFrameIndex=2},
  @{name='mfg3-index1';MultiFrameCount=3;MultiFrameIndex=1},
  @{name='mfg3-index3';MultiFrameCount=3;MultiFrameIndex=3}
)
$outcomes=@()
foreach($row in $rows){
 foreach($poison in @(165,90)){
  $suffix=if($poison -eq 165){'a5'}else{'5a'}
  $runId="$($row.name)-poison-$suffix"
  $arguments=@{JobRoot=$JobRoot;RunnerDirectory=$RunnerDirectory;RunId=$runId;OutputPoisonByte=$poison;Width=640;Height=360;RenderedFrames=2}
  foreach($key in $row.Keys){if($key -ne 'name'){$arguments[$key]=$row[$key]}}
  & $TraceRunner @arguments | Out-Null
  $exit=$LASTEXITCODE
  $processPath=Join-Path $JobRoot "receipts\$runId-process.json"
  $process=Get-Content -LiteralPath $processPath -Raw|ConvertFrom-Json
  $hostPath=Join-Path $JobRoot "output\$runId\fg-run-receipt.json"
  $hostReceipt=if(Test-Path -LiteralPath $hostPath){Get-Content -LiteralPath $hostPath -Raw|ConvertFrom-Json}else{$null}
  $hostHash=if($hostReceipt){(Get-FileHash $hostPath -Algorithm SHA256).Hash.ToLowerInvariant()}else{$null}
  $profile=if($hostReceipt){$hostReceipt.resource_profile}else{$null}
  $outputHash=if($hostReceipt){$hostReceipt.frames[-1].output_sha256}else{$null}
  $outcomes+=[ordered]@{mode=$row.name;poison_byte=$poison;run_id=$runId;exit_code=$exit;process_receipt_sha256=(Get-FileHash $processPath -Algorithm SHA256).Hash.ToLowerInvariant();host_receipt_sha256=$hostHash;resource_profile=$profile;generated_output_sha256=$outputHash;trace_sha256=$process.trace_sha256;trace_lines=$process.trace_lines;error=$process.error}
 }
}
$receipt=[ordered]@{schema='neuralstudio.fg.feature-modes-matrix-run.v1';job_id=(Split-Path -Leaf $JobRoot);runner_directory=$RunnerDirectory;rows=$outcomes;address_fields_recorded=$false;result='complete'}
$receipt|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $JobRoot 'feature-modes-matrix.json') -Encoding UTF8
$receipt|ConvertTo-Json -Compress -Depth 8
