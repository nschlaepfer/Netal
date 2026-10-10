param([Parameter(Mandatory=$true)][string]$JobRoot,[Parameter(Mandatory=$true)][string]$RunnerDirectory,[Parameter(Mandatory=$true)][string]$TraceRunner,[string]$RunId='distortion-descriptors-v3',[int]$OutputPoisonByte=90)
$ErrorActionPreference='Stop';$JobRoot=[IO.Path]::GetFullPath($JobRoot);if(-not $JobRoot.StartsWith('D:\CodexLab\')){throw 'invalid job'}
$d=Join-Path $JobRoot ("descriptors\"+$RunId);New-Item -ItemType Directory -Path $d,(Join-Path $JobRoot 'logs'),(Join-Path $JobRoot 'receipts') -Force|Out-Null
$env:NEURALSTUDIO_FG_D3D12_DESCRIPTOR_TRACE=Join-Path $d 'semantic.jsonl';$env:NEURALSTUDIO_FG_D3D12_DESCRIPTOR_VOLATILE_TRACE=Join-Path $d 'volatile.jsonl'
& $TraceRunner -JobRoot $JobRoot -RunnerDirectory $RunnerDirectory -RunId $RunId -OutputPoisonByte $OutputPoisonByte -BidirectionalDistortion -CaptureRuntimeAllocations -ExpectedBaselineOutputSha256 '07a87435b2e7d79540dcec737b9abb6354e68e8595b4df29ca6f8d92cce54383'|Out-Null
if($LASTEXITCODE -ne 0){throw 'distortion descriptor run failed'}
$s=(Get-Content $env:NEURALSTUDIO_FG_D3D12_DESCRIPTOR_TRACE|Select-Object -Last 1|ConvertFrom-Json);if(-not $s.capture_healthy){throw 'descriptor capture unhealthy'}
$r=[ordered]@{schema='neuralstudio.fg.distortion-descriptor-capture.v1';job_id=(Split-Path -Leaf $JobRoot);semantic_sha256=(Get-FileHash $env:NEURALSTUDIO_FG_D3D12_DESCRIPTOR_TRACE -Algorithm SHA256).Hash.ToLowerInvariant();summary=$s;result='pass'};$r|ConvertTo-Json -Depth 8|Set-Content (Join-Path $JobRoot 'output\descriptor-receipt.json') -Encoding UTF8;$r|ConvertTo-Json -Compress -Depth 8
