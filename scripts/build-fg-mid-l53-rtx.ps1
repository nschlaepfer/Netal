[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$JobRoot,
    [string]$AttemptId,
    [string]$RepositoryRoot,
    [string]$BaseCommit = '498d69c'
)

$ErrorActionPreference = 'Stop'
if (-not $RepositoryRoot) { $RepositoryRoot = Split-Path -Parent $PSScriptRoot }
if (-not $AttemptId) {
    $AttemptId = 'attempt-' + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
}
$JobRoot = [IO.Path]::GetFullPath($JobRoot)
$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
$source = Join-Path $RepositoryRoot 'src\fg_mid_l53\rtx_oracle.cpp'
$contract = Join-Path $RepositoryRoot 'src\fg_mid_l53\contract.h'
$verifier = Join-Path $RepositoryRoot 'scripts\verify-fg-mid-l53.py'
$baseSource = Join-Path $RepositoryRoot 'src\fg_conv\rtx_oracle.cpp'
$harness = Join-Path $RepositoryRoot 'scripts\build-fg-mid-l53-rtx.ps1'
$module = Join-Path $JobRoot 'inputs\cu-module-000012.bin'
$attemptRoot = Join-Path $JobRoot (Join-Path 'output' $AttemptId)
$buildRoot = Join-Path $attemptRoot 'build'
foreach ($required in @($source, $contract, $baseSource, $harness, $verifier, $module)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Missing L53 input: $required"
    }
}
if (Test-Path -LiteralPath $attemptRoot) { throw "Attempt exists: $attemptRoot" }
if ((Get-Item -LiteralPath $module).Length -ne 98408 -or
    (Get-FileHash -LiteralPath $module -Algorithm SHA256).Hash -ne
        '5A8E0284AAB8AC14FC82B0504BBEEF25D2FCE1D13A1C11D8BDB3F91FEE8145FC') {
    throw 'L53 module identity mismatch'
}

$cudaPath = $env:CUDA_PATH
if (-not $cudaPath) {
    $cudaPath = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
}
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
New-Item -ItemType Directory -Path $buildRoot | Out-Null
$binary = Join-Path $buildRoot 'fg-mid-l53-rtx.exe'
$object = Join-Path $buildRoot 'fg-mid-l53-rtx.obj'
$compileLog = Join-Path $buildRoot 'compile.log'
$compileCommand = '"{0}" >nul && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /I"{1}" /Fo"{2}" /c "{3}" >"{4}" 2>&1 && link.exe /nologo /OUT:"{5}" "{2}" "{6}" bcrypt.lib >>"{4}" 2>&1' -f `
    $vcvars, (Join-Path $cudaPath 'include'), $object, $source, $compileLog,
    $binary, (Join-Path $cudaPath 'lib\x64\cuda.lib')
& $env:ComSpec /d /s /c $compileCommand
if ($LASTEXITCODE -ne 0) { throw "L53 build failed; see $compileLog" }

$receipts = @()
foreach ($scenario in @('small-flow', 'pixel-flow')) {
  foreach ($poison in @('a5', '5a')) {
    foreach ($ordinal in @('a', 'b')) {
        $runId = "l53-$scenario-poison-$poison-process-$ordinal"
        $runRoot = Join-Path $attemptRoot $runId
        New-Item -ItemType Directory -Path $runRoot | Out-Null
        $process = Start-Process -FilePath $binary `
            -ArgumentList @($module, $runRoot, $runId, $poison, $scenario) `
            -NoNewWindow -Wait -PassThru `
            -RedirectStandardOutput (Join-Path $runRoot 'stdout.log') `
            -RedirectStandardError (Join-Path $runRoot 'stderr.log')
        if ($process.ExitCode -ne 0) {
            throw "L53 run failed: $runId exit=$($process.ExitCode)"
        }
        $receipts += Get-Content -LiteralPath `
            (Join-Path $runRoot 'run-receipt.json') -Raw | ConvertFrom-Json
    }
  }
}

$negativeRoot = Join-Path $attemptRoot 'unsupported-scenario-negative'
New-Item -ItemType Directory -Path $negativeRoot | Out-Null
$negative = Start-Process -FilePath $binary `
    -ArgumentList @($module, $negativeRoot, 'unsupported-scenario-negative', 'a5', 'unsupported') `
    -NoNewWindow -Wait -PassThru `
    -RedirectStandardOutput (Join-Path $negativeRoot 'stdout.log') `
    -RedirectStandardError (Join-Path $negativeRoot 'stderr.log')
if ($negative.ExitCode -ne 1 -or
    (Test-Path -LiteralPath (Join-Path $negativeRoot 'run-receipt.json'))) {
    throw "Unsupported L53 scenario did not fail closed: exit=$($negative.ExitCode)"
}
$processIds = @($receipts | ForEach-Object { [int64]$_.process_id } | Sort-Object -Unique)
if ($processIds.Count -ne 8) { throw 'L53 fresh-process identity gate failed' }
$scenarioOutputs = [ordered]@{}
foreach ($scenario in @('small-flow', 'pixel-flow')) {
    $selected = @($receipts | Where-Object { $_.scenario -eq $scenario })
    $outputAHashes = @($selected | ForEach-Object { $_.output_sha256[0] } | Sort-Object -Unique)
    $outputBHashes = @($selected | ForEach-Object { $_.output_sha256[1] } | Sort-Object -Unique)
    if ($selected.Count -ne 4 -or $outputAHashes.Count -ne 1 -or
        $outputBHashes.Count -ne 1) {
        throw "L53 fresh-process/two-poison invariance failed: $scenario"
    }
    $scenarioOutputs[$scenario] = [ordered]@{
        output_a_sha256 = $outputAHashes[0]
        output_b_sha256 = $outputBHashes[0]
        sentinel_u64_count = @([int64]$selected[0].sentinel_u64_count[0], [int64]$selected[0].sentinel_u64_count[1])
    }
}

$receipt = [ordered]@{
    schema = 'neuralstudio.fg.mid-l53-rig.v1'
    job_id = Split-Path -Leaf $JobRoot
    attempt_id = $AttemptId
    local_base_commit = $BaseCommit
    gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total,pci.bus_id --format=csv,noheader,nounits | Select-Object -First 1).Trim()
    cuda_toolkit = (& (Join-Path $cudaPath 'bin\nvcc.exe') --version | Select-Object -Last 1).Trim()
    module_sha256 = (Get-FileHash -LiteralPath $module -Algorithm SHA256).Hash.ToLowerInvariant()
    source_sha256 = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
    contract_sha256 = (Get-FileHash -LiteralPath $contract -Algorithm SHA256).Hash.ToLowerInvariant()
    harness_sha256 = (Get-FileHash -LiteralPath $harness -Algorithm SHA256).Hash.ToLowerInvariant()
    verifier_sha256 = (Get-FileHash -LiteralPath $verifier -Algorithm SHA256).Hash.ToLowerInvariant()
    binary_sha256 = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant()
    run_count = $receipts.Count
    fresh_process_count = $processIds.Count
    scenarios = $scenarioOutputs
    unsupported_scenario_negative = [ordered]@{
        process_id = $negative.Id
        exit_code = $negative.ExitCode
        receipt_created = $false
        result = 'rejected-before-gpu-execution'
    }
    expected_frame_bytes_read = $false
    archived_intermediates_read = $false
    address_fields_recorded = $false
    result = 'pass'
}
$utf8 = New-Object System.Text.UTF8Encoding($false)
$receiptPath = Join-Path $attemptRoot 'rig-receipt.json'
[IO.File]::WriteAllText($receiptPath, (($receipt | ConvertTo-Json -Depth 5) + "`n"), $utf8)
$backendPath = Join-Path $attemptRoot 'backend-receipt.json'
& python $verifier backend --backend rtx --root $attemptRoot --output $backendPath *> (Join-Path $attemptRoot 'backend-verification.log')
if ($LASTEXITCODE -ne 0) { throw 'L53 RTX backend verification failed' }
$backendHash = (Get-FileHash -LiteralPath $backendPath -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText((Join-Path $attemptRoot 'backend-receipt.sha256'), ($backendHash + "  backend-receipt.json`n"), $utf8)
Write-Host "attempt_root=$attemptRoot"
Write-Host "rig_receipt_sha256=$((Get-FileHash -LiteralPath $receiptPath -Algorithm SHA256).Hash.ToLowerInvariant())"
Write-Host "backend_receipt_sha256=$backendHash"
Write-Host "small_flow_output_a_sha256=$($scenarioOutputs['small-flow'].output_a_sha256)"
Write-Host "small_flow_output_b_sha256=$($scenarioOutputs['small-flow'].output_b_sha256)"
Write-Host "pixel_flow_output_a_sha256=$($scenarioOutputs['pixel-flow'].output_a_sha256)"
Write-Host "pixel_flow_output_b_sha256=$($scenarioOutputs['pixel-flow'].output_b_sha256)"
Write-Host 'result=pass'
