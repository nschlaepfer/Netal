param(
    [Parameter(Mandatory = $true)][string]$JobRoot,
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [Parameter(Mandatory = $true)][string]$AttemptId,
    [string]$BaseCommit = 'working-tree-source-pins'
)
$ErrorActionPreference = 'Stop'
$started = (Get-Date).ToUniversalTime().ToString('o')
$JobRoot = [IO.Path]::GetFullPath($JobRoot)
$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
if (-not $JobRoot.StartsWith('D:\CodexLab\')) { throw 'JobRoot must be beneath D:\CodexLab' }
$source = Join-Path $RepositoryRoot 'src\fg_early_status_l10_l12\rtx_oracle.cpp'
$contract = Join-Path $RepositoryRoot 'src\fg_early_status_l10_l12\contract.h'
$harness = Join-Path $RepositoryRoot 'scripts\build-fg-early-status-l10-l12-rtx.ps1'
$verifier = Join-Path $RepositoryRoot 'scripts\verify-fg-early-status-l10-l12.py'
$module = Join-Path $JobRoot 'input\cu-module-000027.bin'
$attemptRoot = Join-Path $JobRoot (Join-Path 'output' $AttemptId)
$buildRoot = Join-Path $attemptRoot 'build'
foreach ($required in @($source,$contract,$harness,$verifier,$module)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Missing status input: $required" }
}
if (Test-Path -LiteralPath $attemptRoot) { throw "Attempt exists: $attemptRoot" }
if ((Get-Item $module).Length -ne 4024 -or (Get-FileHash $module -Algorithm SHA256).Hash -ne 'C8B8C082139005C1BF059F156827C4E331F9A6579DED1B2C9646198B1CC2ABF7') { throw 'module27 identity mismatch' }
$cudaPath = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3' }
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
New-Item -ItemType Directory -Path $buildRoot | Out-Null
$binary = Join-Path $buildRoot 'fg-early-status-l10-l12-rtx.exe'
$object = Join-Path $buildRoot 'fg-early-status-l10-l12-rtx.obj'
$compileLog = Join-Path $buildRoot 'compile.log'
$command = '"{0}" >nul && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /I"{1}" /Fo"{2}" /c "{3}" >"{4}" 2>&1 && link.exe /nologo /OUT:"{5}" "{2}" "{6}" bcrypt.lib >>"{4}" 2>&1' -f $vcvars,(Join-Path $cudaPath 'include'),$object,$source,$compileLog,$binary,(Join-Path $cudaPath 'lib\x64\cuda.lib')
& $env:ComSpec /d /s /c $command
if ($LASTEXITCODE -ne 0) { throw "Status build failed; see $compileLog" }
$receipts = @()
$processResults = @()
foreach ($mode in @('reset','generated')) {
  foreach ($poison in @('a5','5a')) {
    foreach ($ordinal in @('a','b')) {
      $runId = "status-$mode-poison-$poison-process-$ordinal"
      $runRoot = Join-Path $attemptRoot $runId
      New-Item -ItemType Directory -Path $runRoot | Out-Null
      $process = Start-Process -FilePath $binary -ArgumentList @($module,$runRoot,$runId,$poison,$mode) -NoNewWindow -Wait -PassThru -RedirectStandardOutput (Join-Path $runRoot 'stdout.log') -RedirectStandardError (Join-Path $runRoot 'stderr.log')
      if ($process.ExitCode -ne 0) { throw "Status run failed: $runId" }
      $receipt = Get-Content (Join-Path $runRoot 'run-receipt.json') -Raw | ConvertFrom-Json
      if ($receipt.result -ne 'pass') { throw "Status receipt failed: $runId" }
      $receipts += $receipt
      $processResults += [ordered]@{run_id=$runId;process_id=$process.Id;exit_code=$process.ExitCode}
    }
  }
}
$pids = @($receipts.process_id | Sort-Object -Unique)
if ($pids.Count -ne 8) { throw 'Status fresh-process gate failed' }
$modes = [ordered]@{}
foreach ($mode in @('reset','generated')) {
    $selected = @($receipts | Where-Object {$_.mode -eq $mode})
    $hashes = @($selected.output_sha256 | Sort-Object -Unique)
    if ($selected.Count -ne 4 -or $hashes.Count -ne 1) { throw "Status repeat gate failed: $mode" }
    $modes[$mode] = [ordered]@{canonical_launch=[int]$selected[0].canonical_launch;value=[int]$selected[0].actual_value;output_sha256=$hashes[0]}
}
$negativeRoot = Join-Path $attemptRoot 'unsupported-mode-negative'
New-Item -ItemType Directory -Path $negativeRoot | Out-Null
$negative = Start-Process -FilePath $binary -ArgumentList @($module,$negativeRoot,'unsupported-mode-negative','a5','unsupported') -NoNewWindow -Wait -PassThru -RedirectStandardOutput (Join-Path $negativeRoot 'stdout.log') -RedirectStandardError (Join-Path $negativeRoot 'stderr.log')
if ($negative.ExitCode -ne 1 -or (Test-Path (Join-Path $negativeRoot 'run-receipt.json'))) { throw 'Unsupported status mode did not fail closed' }
$rig = [ordered]@{
 schema='neuralstudio.fg.early-status-l10-l12-rig.v1';job_id=(Split-Path -Leaf $JobRoot);attempt_id=$AttemptId;base_commit=$BaseCommit;started_utc=$started;ended_utc=(Get-Date).ToUniversalTime().ToString('o');gpu=(& nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader,nounits|Select-Object -First 1).Trim();cuda_toolkit=(&(Join-Path $cudaPath 'bin\nvcc.exe') --version|Select-Object -Last 1).Trim();module_sha256=(Get-FileHash $module -Algorithm SHA256).Hash.ToLowerInvariant();source_sha256=(Get-FileHash $source -Algorithm SHA256).Hash.ToLowerInvariant();contract_sha256=(Get-FileHash $contract -Algorithm SHA256).Hash.ToLowerInvariant();harness_sha256=(Get-FileHash $harness -Algorithm SHA256).Hash.ToLowerInvariant();verifier_sha256=(Get-FileHash $verifier -Algorithm SHA256).Hash.ToLowerInvariant();binary_sha256=(Get-FileHash $binary -Algorithm SHA256).Hash.ToLowerInvariant();run_count=$receipts.Count;fresh_process_count=$pids.Count;process_results=$processResults;modes=$modes;unsupported_mode_negative=[ordered]@{process_id=$negative.Id;exit_code=$negative.ExitCode;receipt_created=$false;result='rejected-before-gpu-execution'};tensor_input_count=0;expected_output_fed_to_execution=$false;archived_intermediates_consumed=$false;address_fields_recorded=$false;result='pass'
}
$utf8=New-Object System.Text.UTF8Encoding($false)
$path=Join-Path $attemptRoot 'rig-receipt.json'
[IO.File]::WriteAllText($path,(($rig|ConvertTo-Json -Depth 8)+"`n"),$utf8)
$backendPath=Join-Path $attemptRoot 'backend-receipt.json'
& python $verifier backend --backend rtx --root $attemptRoot --output $backendPath *> (Join-Path $attemptRoot 'backend-verification.log')
if ($LASTEXITCODE -ne 0) { throw 'Status RTX backend verification failed' }
$backendHash=(Get-FileHash $backendPath -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText((Join-Path $attemptRoot 'backend-receipt.sha256'),($backendHash+"  backend-receipt.json`n"),$utf8)
Write-Host "attempt_root=$attemptRoot"
Write-Host "rig_receipt_sha256=$((Get-FileHash $path -Algorithm SHA256).Hash.ToLowerInvariant())"
Write-Host "backend_receipt_sha256=$backendHash"
Write-Host 'result=pass'
