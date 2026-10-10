[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$JobRoot,[ValidateSet('a5','5a')][string]$Poison='a5',[ValidateRange(0,1)][int]$SecondReset=1,[ValidatePattern('^[a-z0-9-]+$')][string]$Label='pilot-reset11-a5')
$ErrorActionPreference='Stop'
$JobRoot=[IO.Path]::GetFullPath($JobRoot).TrimEnd('\')
if (-not $JobRoot.StartsWith('D:\CodexLab\nr-color-reset-')) { throw 'Isolated reset job required' }
if ((Get-PSDrive D).Free -lt 10GB) { throw 'Storage floor' }
$source=Join-Path $JobRoot 'source';$build=Join-Path $JobRoot 'build';$out=Join-Path $JobRoot $Label
if (Test-Path -LiteralPath $out) { throw 'Preserve existing run' }
$b=Get-Content -LiteralPath (Join-Path $JobRoot 'build-receipt.json') -Raw | ConvertFrom-Json
if ($b.exit_code -ne 0) { throw 'Build failed' }
foreach ($p in $b.files.PSObject.Properties) { if ((Get-FileHash -LiteralPath (Join-Path $build $p.Name) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $p.Value) { throw 'Binary drift' } }
$env:TEMP=Join-Path $JobRoot 'temp';$env:TMP=$env:TEMP;$env:CUDA_CACHE_PATH=Join-Path $JobRoot 'cuda-cache'
$env:NEURALSTUDIO_NR_COLOR_CASE='0';$env:NEURALSTUDIO_NR_OUTPUT_POISON=$Poison;$env:NEURALSTUDIO_NR_SECOND_RESET=[string]$SecondReset
$start=(Get-Date).ToUniversalTime().ToString('o');$gpu=(& nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader)
$argsCapture=@('-NoLogo','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',(Join-Path $source 'capture-color.ps1'),'-ArmRealExecution','-ArtifactDirectory',$out,'-RuntimeBinary',(Join-Path $build 'nvngx_dlssnr.dll'),'-HarnessBinary',(Join-Path $build 'color-cases.exe'),'-ProxyBinary',(Join-Path $build 'nvcuda.dll'),'-ThunkBinary',(Join-Path $build 'nvngx.dll'))
& powershell.exe @argsCapture *> (Join-Path $JobRoot ($Label+'.log'));$exitCode=$LASTEXITCODE
$record=[ordered]@{schema='neuralstudio.nr.reset-sequence-job.v1';started_utc=$start;ended_utc=(Get-Date).ToUniversalTime().ToString('o');command=$argsCapture;exit_code=$exitCode;gpu_before=$gpu;gpu_after=(& nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader);resets=@(1,$SecondReset);poison=$Poison;build_receipt_sha256=(Get-FileHash -LiteralPath (Join-Path $JobRoot 'build-receipt.json') -Algorithm SHA256).Hash.ToLowerInvariant();result='execution_failed';performance_benchmark=$false;temporal_acceptance=$false}
if ($exitCode -eq 0) {
  $cap=Get-Content -LiteralPath (Join-Path $out 'capture-receipt.json') -Raw | ConvertFrom-Json
  $record.capture_receipt_sha256=(Get-FileHash -LiteralPath (Join-Path $out 'capture-receipt.json') -Algorithm SHA256).Hash.ToLowerInvariant()
  $record.first_output_sha256=$cap.output_sha256;$record.second_output_sha256=$cap.second_output_sha256
  $record.first_matches_control=($cap.output_sha256 -eq 'a07835db4be8469dad1e3d3befcc495be396790f89408be62ddb8a5a710585a2')
  $record.second_equals_first=($cap.second_output_sha256 -eq $cap.output_sha256)
  $record.result=if ($record.first_matches_control) {'observed'} else {'control_counterexample'}
}
$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $JobRoot ($Label+'-receipt.json')) -Encoding UTF8
$record | ConvertTo-Json -Depth 8
if ($exitCode -ne 0) { throw 'Capture failed; all logs preserved' }
