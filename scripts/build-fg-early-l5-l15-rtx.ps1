[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$JobRoot,
    [string]$AttemptId = 'oracle-v1',
    [string]$RepositoryRoot,
    [string]$BaseCommit = '384913a'
)
$ErrorActionPreference = 'Stop'
if (-not $RepositoryRoot) { $RepositoryRoot = Join-Path $JobRoot 'repo' }
$JobRoot = [IO.Path]::GetFullPath($JobRoot)
$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
if (-not $JobRoot.StartsWith('D:\CodexLab\', [StringComparison]::OrdinalIgnoreCase)) {
    throw "JobRoot must be beneath D:\CodexLab: $JobRoot"
}
$source = Join-Path $RepositoryRoot 'src\fg_early_l5_l15\rtx_oracle.cpp'
$contract = Join-Path $RepositoryRoot 'src\fg_early_l5_l15\contract.h'
$baseSource = Join-Path $RepositoryRoot 'src\fg_conv\rtx_oracle.cpp'
$harness = Join-Path $RepositoryRoot 'scripts\build-fg-early-l5-l15-rtx.ps1'
$module = Join-Path $JobRoot 'inputs\cu-module-000013.bin'
$attemptRoot = Join-Path $JobRoot (Join-Path 'output' $AttemptId)
$buildRoot = Join-Path $attemptRoot 'build'
foreach ($required in @($source, $contract, $baseSource, $harness, $module)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Missing L5/L15 input: $required"
    }
}
if (Test-Path -LiteralPath $attemptRoot) { throw "Attempt exists: $attemptRoot" }
if ((Get-Item -LiteralPath $module).Length -ne 13088 -or
    (Get-FileHash -LiteralPath $module -Algorithm SHA256).Hash -ne
        'FD0665969DA887DD92540B673009E0361A6C1DE74FE54BADF48F300B1FCCBBF1') {
    throw 'Module13 identity mismatch'
}
$cudaPath = $env:CUDA_PATH
if (-not $cudaPath) { $cudaPath = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3' }
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
New-Item -ItemType Directory -Path $buildRoot | Out-Null
$env:TEMP = $buildRoot
$env:TMP = $buildRoot
$binary = Join-Path $buildRoot 'fg-early-l5-l15-rtx.exe'
$object = Join-Path $buildRoot 'fg-early-l5-l15-rtx.obj'
$compileLog = Join-Path $buildRoot 'compile.log'
$started = (Get-Date).ToUniversalTime().ToString('o')
$compileCommand = '"{0}" >nul && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /I"{1}" /Fo"{2}" /c "{3}" >"{4}" 2>&1 && link.exe /nologo /OUT:"{5}" "{2}" "{6}" bcrypt.lib >>"{4}" 2>&1' -f $vcvars,(Join-Path $cudaPath 'include'),$object,$source,$compileLog,$binary,(Join-Path $cudaPath 'lib\x64\cuda.lib')
& $env:ComSpec /d /s /c $compileCommand
if ($LASTEXITCODE -ne 0) { throw "L5/L15 build failed; see $compileLog" }
$receipts = @()
foreach ($phase in @('reset', 'generated')) {
    $launch = if ($phase -eq 'reset') { 'l5' } else { 'l15' }
    foreach ($scenario in @('equal-high-tie', 'priority-collision')) {
        foreach ($poison in @('a5', '5a')) {
            foreach ($ordinal in @('a', 'b')) {
                $runId = "$launch-$phase-$scenario-poison-$poison-process-$ordinal"
                $runRoot = Join-Path $attemptRoot $runId
                New-Item -ItemType Directory -Path $runRoot | Out-Null
                $process = Start-Process -FilePath $binary -ArgumentList @($module, $runRoot, $runId, $poison, $phase, $scenario) -NoNewWindow -Wait -PassThru -RedirectStandardOutput (Join-Path $runRoot 'stdout.log') -RedirectStandardError (Join-Path $runRoot 'stderr.log')
                if ($process.ExitCode -ne 0) {
                    throw "L5/L15 run failed: $runId exit=$($process.ExitCode)"
                }
                $receipt = Get-Content -LiteralPath (Join-Path $runRoot 'run-receipt.json') -Raw | ConvertFrom-Json
                if ($receipt.result -ne 'pass' -or $receipt.input_mismatched_bytes -ne 0 -or
                    $receipt.atomic_canary_mismatched_bytes -ne 0 -or
                    $receipt.mask_guard_row_mismatched_bytes -ne 0) {
                    throw "L5/L15 run receipt rejected: $runId"
                }
                $receipts += $receipt
            }
        }
    }
}
$processIds = @($receipts | ForEach-Object { [int64]$_.process_id } | Sort-Object -Unique)
if ($processIds.Count -ne 16) { throw 'Fresh-process gate failed' }
$summaries = @()
foreach ($phase in @('reset', 'generated')) {
    foreach ($scenario in @('equal-high-tie', 'priority-collision')) {
        $selected = @($receipts | Where-Object { $_.phase -eq $phase -and $_.scenario -eq $scenario })
        $atomic = @($selected | ForEach-Object { $_.atomic_output_sha256 } | Sort-Object -Unique)
        $mask = @($selected | ForEach-Object { $_.mask_output_sha256 } | Sort-Object -Unique)
        if ($selected.Count -ne 4 -or $atomic.Count -ne 1 -or $mask.Count -ne 1) {
            throw "Poison/process invariance failed: $phase/$scenario"
        }
        $summaries += [ordered]@{
            phase = $phase
            scenario = $scenario
            atomic_output_sha256 = $atomic[0]
            mask_output_sha256 = $mask[0]
            atomic_sentinel_u64 = [int64]$selected[0].atomic_sentinel_u64
            atomic_candidate_u64 = [int64]$selected[0].atomic_candidate_u64
            atomic_unique_high_words = [int64]$selected[0].atomic_unique_high_words
            atomic_unique_low_words = [int64]$selected[0].atomic_unique_low_words
            mask_zero_pixels = [int64]$selected[0].mask_zero_pixels
            mask_one_pixels = [int64]$selected[0].mask_one_pixels
        }
    }
}
$completed = (Get-Date).ToUniversalTime().ToString('o')
$receipt = [ordered]@{
    schema = 'neuralstudio.fg.early-l5-l15-rig.v1'
    job_id = Split-Path -Leaf $JobRoot
    attempt_id = $AttemptId
    local_base_commit = $BaseCommit
    started_utc = $started
    completed_utc = $completed
    gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total,pci.bus_id --format=csv,noheader,nounits | Select-Object -First 1).Trim()
    cuda_toolkit = (& (Join-Path $cudaPath 'bin\nvcc.exe') --version | Select-Object -Last 1).Trim()
    module_sha256 = (Get-FileHash -LiteralPath $module -Algorithm SHA256).Hash.ToLowerInvariant()
    source_sha256 = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
    contract_sha256 = (Get-FileHash -LiteralPath $contract -Algorithm SHA256).Hash.ToLowerInvariant()
    shared_helper_sha256 = (Get-FileHash -LiteralPath $baseSource -Algorithm SHA256).Hash.ToLowerInvariant()
    harness_sha256 = (Get-FileHash -LiteralPath $harness -Algorithm SHA256).Hash.ToLowerInvariant()
    binary_sha256 = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant()
    command = '16 fresh processes: L5/L15 x tie/priority x A5/5A x repeat'
    run_count = $receipts.Count
    fresh_process_count = $processIds.Count
    summaries = $summaries
    input_immutable = $true
    canaries_intact = $true
    expected_or_archived_ingress = $false
    address_fields_recorded = $false
    result = 'pass'
}
$utf8 = New-Object System.Text.UTF8Encoding($false)
$receiptPath = Join-Path $attemptRoot 'rig-receipt.json'
[IO.File]::WriteAllText($receiptPath, (($receipt | ConvertTo-Json -Depth 7) + "`n"), $utf8)
Write-Host "attempt_root=$attemptRoot"
Write-Host "rig_receipt_sha256=$((Get-FileHash -LiteralPath $receiptPath -Algorithm SHA256).Hash.ToLowerInvariant())"
Write-Host 'result=pass'
