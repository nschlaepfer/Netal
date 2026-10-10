[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$JobRoot,
    [string]$AttemptId,
    [string]$RepositoryRoot,
    [string]$BaseCommit = '8540f35'
)
$ErrorActionPreference = 'Stop'
if (-not $RepositoryRoot) { $RepositoryRoot = Join-Path $JobRoot 'repo' }
if (-not $AttemptId) { $AttemptId = 'attempt-' + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ') }
$JobRoot = [IO.Path]::GetFullPath($JobRoot)
$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
if (-not $JobRoot.StartsWith('D:\CodexLab\', [StringComparison]::OrdinalIgnoreCase)) {
    throw "JobRoot must be a new D:\CodexLab job: $JobRoot"
}
$source = Join-Path $RepositoryRoot 'src\fg_early_l11\rtx_nvapi_oracle.cpp'
$contract = Join-Path $RepositoryRoot 'src\fg_early_l11\contract.h'
$baseSource = Join-Path $RepositoryRoot 'src\fg_compositor_l87\rtx_nvapi_oracle.cpp'
$baseContract = Join-Path $RepositoryRoot 'src\fg_compositor_l87\compositor_l87_contract.h'
$harness = Join-Path $RepositoryRoot 'scripts\build-fg-early-l11-rtx.ps1'
$module = Join-Path $JobRoot 'inputs\cu-module-000004.bin'
$attemptRoot = Join-Path $JobRoot (Join-Path 'output' $AttemptId)
$buildRoot = Join-Path $attemptRoot 'build'
foreach ($required in @($source, $contract, $baseSource, $baseContract, $harness, $module)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Missing L11 input: $required"
    }
}
if (Test-Path -LiteralPath $attemptRoot) { throw "Attempt exists: $attemptRoot" }
if ((Get-Item -LiteralPath $module).Length -ne 4512 -or
    (Get-FileHash -LiteralPath $module -Algorithm SHA256).Hash -ne
        'C11CD7474562E20762D272162C94378A17F8F77C7DE8206D695E6C469383E840') {
    throw 'Module identity mismatch'
}
$cudaPath = $env:CUDA_PATH
if (-not $cudaPath) { $cudaPath = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3' }
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
New-Item -ItemType Directory -Path $buildRoot | Out-Null
$env:TEMP = $buildRoot
$env:TMP = $buildRoot
$binary = Join-Path $buildRoot 'fg-early-l11-rtx.exe'
$object = Join-Path $buildRoot 'fg-early-l11-rtx.obj'
$compileLog = Join-Path $buildRoot 'compile.log'
$started = (Get-Date).ToUniversalTime().ToString('o')
$compileCommand = '"{0}" >nul && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /Fo"{1}" /c "{2}" >"{3}" 2>&1 && link.exe /nologo /OUT:"{4}" "{1}" bcrypt.lib d3d12.lib dxgi.lib >>"{3}" 2>&1' -f $vcvars, $object, $source, $compileLog, $binary
& $env:ComSpec /d /s /c $compileCommand
if ($LASTEXITCODE -ne 0) { throw "L11 build failed; see $compileLog" }
$receipts = @()
foreach ($poison in @('a5', '5a')) {
    foreach ($ordinal in @('a', 'b')) {
        $runId = "l11-poison-$poison-process-$ordinal"
        $runRoot = Join-Path $attemptRoot $runId
        New-Item -ItemType Directory -Path $runRoot | Out-Null
        $process = Start-Process -FilePath $binary -ArgumentList @($module, $runRoot, $runId, $poison) -NoNewWindow -Wait -PassThru -RedirectStandardOutput (Join-Path $runRoot 'stdout.log') -RedirectStandardError (Join-Path $runRoot 'stderr.log')
        if ($process.ExitCode -ne 0) {
            throw "L11 run failed: $runId exit=$($process.ExitCode)"
        }
        $receipt = Get-Content -LiteralPath (Join-Path $runRoot 'run-receipt.json') -Raw | ConvertFrom-Json
        if ($receipt.result -ne 'pass' -or $receipt.input_mismatched_bytes -ne 0 -or
            $receipt.declared_region_mismatched_bytes -ne 0 -or
            $receipt.outside_region_mismatched_bytes -ne 0 -or
            $receipt.canary_mismatched_bytes -ne 0) {
            throw "L11 run receipt rejected: $runId"
        }
        $receipts += $receipt
    }
}
$processIds = @($receipts | ForEach-Object { [int64]$_.process_id } | Sort-Object -Unique)
$regionHashes = @($receipts | ForEach-Object { $_.region_output_sha256 } | Sort-Object -Unique)
$inputHashes = @($receipts | ForEach-Object { $_.input_pre_sha256 } | Sort-Object -Unique)
if ($processIds.Count -ne 4 -or $regionHashes.Count -ne 1 -or $inputHashes.Count -ne 1) {
    throw 'Fresh-process/two-poison/input invariance failed'
}
$completed = (Get-Date).ToUniversalTime().ToString('o')
$rigReceipt = [ordered]@{
    schema = 'neuralstudio.fg.early-l11-rig.v1'
    job_id = Split-Path -Leaf $JobRoot
    attempt_id = $AttemptId
    local_base_commit = $BaseCommit
    started_utc = $started
    completed_utc = $completed
    gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total,pci.bus_id --format=csv,noheader,nounits | Select-Object -First 1).Trim()
    cuda_toolkit = (& (Join-Path $cudaPath 'bin\nvcc.exe') --version | Select-Object -Last 1).Trim()
    module = [ordered]@{
        bytes = (Get-Item -LiteralPath $module).Length
        sha256 = (Get-FileHash -LiteralPath $module -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    sources = @(
        [ordered]@{ file = 'rtx_nvapi_oracle.cpp'; sha256 = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant() },
        [ordered]@{ file = 'contract.h'; sha256 = (Get-FileHash -LiteralPath $contract -Algorithm SHA256).Hash.ToLowerInvariant() },
        [ordered]@{ file = 'fg_compositor_l87/rtx_nvapi_oracle.cpp'; sha256 = (Get-FileHash -LiteralPath $baseSource -Algorithm SHA256).Hash.ToLowerInvariant() },
        [ordered]@{ file = 'fg_compositor_l87/compositor_l87_contract.h'; sha256 = (Get-FileHash -LiteralPath $baseContract -Algorithm SHA256).Hash.ToLowerInvariant() },
        [ordered]@{ file = 'build-fg-early-l11-rtx.ps1'; sha256 = (Get-FileHash -LiteralPath $harness -Algorithm SHA256).Hash.ToLowerInvariant() }
    )
    binary_sha256 = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant()
    command = 'four fresh fg-early-l11-rtx.exe processes; poison a5,a5,5a,5a; independently generated input'
    run_count = $receipts.Count
    fresh_process_count = $processIds.Count
    poison_bytes = @(165, 90)
    region_output_sha256 = $regionHashes[0]
    two_poison_region_invariant = $true
    input_immutable = $true
    zero_canary_mismatches = $true
    expected_or_archived_ingress = $false
    address_fields_recorded = $false
    result = 'pass'
}
$utf8 = New-Object System.Text.UTF8Encoding($false)
$receiptPath = Join-Path $attemptRoot 'rig-receipt.json'
[IO.File]::WriteAllText($receiptPath, (($rigReceipt | ConvertTo-Json -Depth 6) + "`n"), $utf8)
Write-Host "attempt_root=$attemptRoot"
Write-Host "rig_receipt_sha256=$((Get-FileHash -LiteralPath $receiptPath -Algorithm SHA256).Hash.ToLowerInvariant())"
Write-Host "region_output_sha256=$($regionHashes[0])"
Write-Host 'result=pass'
