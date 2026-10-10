[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$JobRoot,[ValidateRange(0,3)][int]$CaseId=0)
$ErrorActionPreference='Stop'
$JobRoot=[IO.Path]::GetFullPath($JobRoot).TrimEnd('\')
if (-not $JobRoot.StartsWith('D:\CodexLab\nr-color-')) { throw 'Isolated D: color job required' }
$source=Join-Path $JobRoot 'source'; $build=Join-Path $JobRoot 'build'
$buildReceipt=Get-Content -LiteralPath (Join-Path $JobRoot 'build-receipt.json') -Raw | ConvertFrom-Json
if ($buildReceipt.exit_code -ne 0) { throw 'Build did not pass' }
foreach ($entry in $buildReceipt.files.PSObject.Properties) {
  if ((Get-FileHash -LiteralPath (Join-Path $build $entry.Name) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value) { throw "Binary drift: $($entry.Name)" }
}
if ($CaseId -ne 0) {
  $control=Get-Content -LiteralPath (Join-Path $JobRoot 'case-0-receipt.json') -Raw | ConvertFrom-Json
  if ($control.result -ne 'pass' -or -not $control.pinned_control_equal -or $control.harness_sha256 -ne $buildReceipt.files.'color-cases.exe') { throw 'Case0 control required first' }
}
$receiptPath=Join-Path $JobRoot "case-$CaseId-receipt.json"
if (Test-Path -LiteralPath $receiptPath) { throw 'Preserve existing case receipt' }
$env:TEMP=Join-Path $JobRoot 'temp';$env:TMP=$env:TEMP;$env:CUDA_CACHE_PATH=Join-Path $JobRoot 'cuda-cache'
$env:NEURALSTUDIO_NR_COLOR_CASE=[string]$CaseId
$inputHashes=@('81ec1de5d1e76e6b55c4fd987b3732180c9720919fbd760dd5b4e8aa8cdd0342','af31801008df2fbb537f365265271e716b26339b4c5a2fa2069c01c1a949e8b8','d2db0ead99a6307a6504afed2d49867f0421e3665bb397bed8277ae4f2046d06','598bb0d29e73b30a2fb74074a8e582b7231e151920bdb52f71d436a0a3c9654e')
$started=(Get-Date).ToUniversalTime().ToString('o');$runs=@();$pids=@();$before=(& nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader)
foreach ($label in @('a5-a','a5-b','5a-a','5a-b')) {
  if ((Get-PSDrive D).Free -lt 10GB) { throw 'D: space below10GiB; preserve prior runs' }
  $env:NEURALSTUDIO_NR_OUTPUT_POISON=$label.Substring(0,2)
  $out=Join-Path $JobRoot "case-$CaseId-$label"
  if (Test-Path -LiteralPath $out) { throw "Preserve prior run: $out" }
  $captureArguments=@('-NoLogo','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',(Join-Path $source 'capture-color.ps1'),'-ArmRealExecution','-ArtifactDirectory',$out,'-RuntimeBinary',(Join-Path $build 'nvngx_dlssnr.dll'),'-HarnessBinary',(Join-Path $build 'color-cases.exe'),'-ProxyBinary',(Join-Path $build 'nvcuda.dll'),'-ThunkBinary',(Join-Path $build 'nvngx.dll'))
  & powershell.exe @captureArguments *> (Join-Path $JobRoot "case-$CaseId-$label.log")
  $exitCode=$LASTEXITCODE
  if ($exitCode -ne 0) { throw "Case run failed: $CaseId $label exit$exitCode; preserve log" }
  $r=Get-Content -LiteralPath (Join-Path $out 'capture-receipt.json') -Raw | ConvertFrom-Json
  $lines=Get-Content -LiteralPath (Join-Path $out 'host-output.txt')
  $pidLine=@($lines | Where-Object { $_ -match '^runtime.process_id=' })
  if ($pidLine.Count -ne 1) { throw 'Missing process identity' }
  $processIdValue=[int]($pidLine[0].Split('=')[1])
  if ($pids -contains $processIdValue) { throw 'Repeated native process identity' };$pids+=$processIdValue
  if ($lines -notcontains "runtime.color_case=$CaseId" -or $lines -notcontains "runtime.output_poison=$($label.Substring(0,2))") { throw 'Case/poison identity missing' }
  $inputHash=(Get-FileHash -LiteralPath (Join-Path $out 'input-color.rgba16le') -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($inputHash -ne $inputHashes[$CaseId]) { throw 'Input differs from tested CPU generator' }
  if ($CaseId -eq 0 -and $r.output_sha256 -ne 'a07835db4be8469dad1e3d3befcc495be396790f89408be62ddb8a5a710585a2') { throw 'Control output differs from accepted RTX frame' }
  $runs += [ordered]@{label=$label;pid=$processIdValue;input_sha256=$inputHash;output_sha256=$r.output_sha256;counts=$r.counts;capture_receipt_sha256=(Get-FileHash -LiteralPath (Join-Path $out 'capture-receipt.json') -Algorithm SHA256).Hash.ToLowerInvariant()}
}
if (@($runs | ForEach-Object { $_.output_sha256 } | Select-Object -Unique).Count -ne 1) { throw 'Fresh-process/output-poison invariance failed' }
$receipt=[ordered]@{schema='neuralstudio.nr.color-case-rtx.v1';case=$CaseId;started_utc=$started;ended_utc=(Get-Date).ToUniversalTime().ToString('o');gpu_before=$before;gpu_after=(& nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader);harness_sha256=$buildReceipt.files.'color-cases.exe';build_receipt_sha256=(Get-FileHash -LiteralPath (Join-Path $JobRoot 'build-receipt.json') -Algorithm SHA256).Hash.ToLowerInvariant();runs=$runs;pinned_control_equal=($CaseId -eq 0);output_poison_invariance=$true;internal_pool_write_coverage_proven=$false;mac_parity_proven=$false;performance_benchmark=$false;result='pass'}
$receipt | ConvertTo-Json -Depth 9 | Set-Content -LiteralPath $receiptPath -Encoding UTF8
$receipt | ConvertTo-Json -Depth 9
