[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$JobRoot,
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [Parameter(Mandatory = $true)][string]$AttemptId,
    [string]$BaseCommit = 'working-tree-source-pins'
)
$ErrorActionPreference = 'Stop'
$JobRoot = [IO.Path]::GetFullPath($JobRoot)
$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
if (-not $JobRoot.StartsWith('D:\CodexLab\', [StringComparison]::OrdinalIgnoreCase)) {
    throw "JobRoot must be beneath D:\CodexLab: $JobRoot"
}
$source = Join-Path $RepositoryRoot 'src\fg_early_l19\rtx_nvapi_oracle.cpp'
$contract = Join-Path $RepositoryRoot 'src\fg_early_l19\contract.h'
$baseSource = Join-Path $RepositoryRoot 'src\fg_compositor_l87\rtx_nvapi_oracle.cpp'
$baseContract = Join-Path $RepositoryRoot 'src\fg_compositor_l87\compositor_l87_contract.h'
$harness = Join-Path $RepositoryRoot 'scripts\build-fg-early-l19-rtx.ps1'
$verifier = Join-Path $RepositoryRoot 'scripts\verify-fg-early-l19.py'
$module = Join-Path $JobRoot 'input\cu-module-000005.bin'
$attemptRoot = Join-Path $JobRoot (Join-Path 'output' $AttemptId)
$buildRoot = Join-Path $attemptRoot 'build'
foreach ($required in @($source,$contract,$baseSource,$baseContract,$harness,$verifier,$module)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Missing L19 input: $required" }
}
if (Test-Path -LiteralPath $attemptRoot) { throw "Attempt exists: $attemptRoot" }
if ((Get-Item -LiteralPath $module).Length -ne 14448 -or
    (Get-FileHash -LiteralPath $module -Algorithm SHA256).Hash -ne
        '3F2308A693730B44C00E064ED25DAB3B73A420F2E92D215EF733C4E10A30439C') {
    throw 'Module5 identity mismatch'
}
$cudaPath = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3' }
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
New-Item -ItemType Directory -Path $buildRoot | Out-Null
$env:TEMP = $buildRoot
$env:TMP = $buildRoot
$binary = Join-Path $buildRoot 'fg-early-l19-rtx.exe'
$object = Join-Path $buildRoot 'fg-early-l19-rtx.obj'
$compileLog = Join-Path $buildRoot 'compile.log'
$started = (Get-Date).ToUniversalTime().ToString('o')
$compileCommand = '"{0}" >nul && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /Fo"{1}" /c "{2}" >"{3}" 2>&1 && link.exe /nologo /OUT:"{4}" "{1}" bcrypt.lib d3d12.lib dxgi.lib >>"{3}" 2>&1' -f $vcvars,$object,$source,$compileLog,$binary
& $env:ComSpec /d /s /c $compileCommand
if ($LASTEXITCODE -ne 0) { throw "L19 build failed; see $compileLog" }

$receipts = @()
$processResults = @()
foreach ($fixture in @('first-frame','fan-in')) {
  foreach ($poison in @('a5','5a')) {
    foreach ($ordinal in @('a','b')) {
      $runId = "l19-$fixture-poison-$poison-process-$ordinal"
      $runRoot = Join-Path $attemptRoot $runId
      New-Item -ItemType Directory -Path $runRoot | Out-Null
      $process = Start-Process -FilePath $binary -ArgumentList @($module,$runRoot,$runId,$poison,$fixture) -NoNewWindow -Wait -PassThru -RedirectStandardOutput (Join-Path $runRoot 'stdout.log') -RedirectStandardError (Join-Path $runRoot 'stderr.log')
      if ($process.ExitCode -ne 0) { throw "L19 run failed: $runId exit=$($process.ExitCode)" }
      $receipt = Get-Content -LiteralPath (Join-Path $runRoot 'run-receipt.json') -Raw | ConvertFrom-Json
      if ($receipt.result -ne 'pass' -or $receipt.input_mismatched_bytes -ne 0 -or
          $receipt.buffer_canary_mismatched_bytes -ne 0 -or
          $receipt.texture_canary_mismatched_bytes -ne 0) {
          throw "L19 receipt rejected: $runId"
      }
      $receipts += $receipt
      $processResults += [ordered]@{run_id=$runId;process_id=$process.Id;exit_code=$process.ExitCode}
    }
  }
}
$processIds = @($receipts.process_id | Sort-Object -Unique)
if ($processIds.Count -ne 8) { throw 'L19 fresh-process gate failed' }
$scenarios = [ordered]@{}
foreach ($fixture in @('first-frame','fan-in')) {
    $selected = @($receipts | Where-Object {$_.fixture -eq $fixture})
    $tensorHashes = @($selected.tensor_output_sha256 | Sort-Object -Unique)
    $maskHashes = @($selected.mask_output_sha256 | Sort-Object -Unique)
    $vectorHashes = @($selected.vector_output_sha256 | Sort-Object -Unique)
    if ($selected.Count -ne 4 -or $tensorHashes.Count -ne 1 -or
        $maskHashes.Count -ne 1 -or $vectorHashes.Count -ne 1) {
        throw "L19 repeat/poison invariance failed: $fixture"
    }
    $scenarios[$fixture] = [ordered]@{tensor_output_sha256=$tensorHashes[0];mask_output_sha256=$maskHashes[0];vector_output_sha256=$vectorHashes[0]}
}
$negativeRoot = Join-Path $attemptRoot 'unsupported-fixture-negative'
New-Item -ItemType Directory -Path $negativeRoot | Out-Null
$negative = Start-Process -FilePath $binary -ArgumentList @($module,$negativeRoot,'unsupported-fixture','a5','unsupported') -NoNewWindow -Wait -PassThru -RedirectStandardOutput (Join-Path $negativeRoot 'stdout.log') -RedirectStandardError (Join-Path $negativeRoot 'stderr.log')
if ($negative.ExitCode -ne 1 -or (Test-Path -LiteralPath (Join-Path $negativeRoot 'run-receipt.json'))) {
    throw 'L19 unsupported fixture did not fail closed'
}
$rig = [ordered]@{
 schema='neuralstudio.fg.early-l19-rig.v1';job_id=(Split-Path -Leaf $JobRoot);attempt_id=$AttemptId;base_commit=$BaseCommit;started_utc=$started;ended_utc=(Get-Date).ToUniversalTime().ToString('o');gpu=(& nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader,nounits|Select-Object -First 1).Trim();cuda_toolkit=(&(Join-Path $cudaPath 'bin\nvcc.exe') --version|Select-Object -Last 1).Trim();module_sha256=(Get-FileHash $module -Algorithm SHA256).Hash.ToLowerInvariant();source_sha256=(Get-FileHash $source -Algorithm SHA256).Hash.ToLowerInvariant();contract_sha256=(Get-FileHash $contract -Algorithm SHA256).Hash.ToLowerInvariant();base_source_sha256=(Get-FileHash $baseSource -Algorithm SHA256).Hash.ToLowerInvariant();base_contract_sha256=(Get-FileHash $baseContract -Algorithm SHA256).Hash.ToLowerInvariant();harness_sha256=(Get-FileHash $harness -Algorithm SHA256).Hash.ToLowerInvariant();verifier_sha256=(Get-FileHash $verifier -Algorithm SHA256).Hash.ToLowerInvariant();binary_sha256=(Get-FileHash $binary -Algorithm SHA256).Hash.ToLowerInvariant();run_count=$receipts.Count;fresh_process_count=$processIds.Count;process_results=$processResults;scenarios=$scenarios;unsupported_fixture_negative=[ordered]@{process_id=$negative.Id;exit_code=$negative.ExitCode;receipt_created=$false;result='rejected-before-gpu-execution'};input_generator='fg-early-l19-independent-ten-inputs-v1';allocation29_capture_bytes_consumed=$false;expected_output_fed_to_execution=$false;archived_intermediates_consumed=$false;address_fields_recorded=$false;result='pass'
}
$utf8=New-Object System.Text.UTF8Encoding($false)
$rigPath=Join-Path $attemptRoot 'rig-receipt.json'
[IO.File]::WriteAllText($rigPath,(($rig|ConvertTo-Json -Depth 8)+"`n"),$utf8)
$backendPath=Join-Path $attemptRoot 'backend-receipt.json'
& python $verifier backend --backend rtx-d3d12-nvapi --root $attemptRoot --output $backendPath *> (Join-Path $attemptRoot 'backend-verification.log')
if ($LASTEXITCODE -ne 0) { throw 'L19 backend verification failed' }
$backendHash=(Get-FileHash $backendPath -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host "attempt_root=$attemptRoot"
Write-Host "rig_receipt_sha256=$((Get-FileHash $rigPath -Algorithm SHA256).Hash.ToLowerInvariant())"
Write-Host "backend_receipt_sha256=$backendHash"
Write-Host 'result=pass'
