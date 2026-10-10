[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$JobRoot,
    [string]$AttemptId = 'oracle-v1',
    [string]$RepositoryRoot,
    [string]$BaseCommit = '956f1df'
)

$ErrorActionPreference = 'Stop'
if (-not $RepositoryRoot) { $RepositoryRoot = Split-Path -Parent $PSScriptRoot }
$JobRoot = [IO.Path]::GetFullPath($JobRoot)
$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
$source = Join-Path $RepositoryRoot 'src\fg_early_l9\rtx_oracle.cpp'
$contract = Join-Path $RepositoryRoot 'src\fg_early_l9\contract.h'
$baseSource = Join-Path $RepositoryRoot 'src\fg_conv\rtx_oracle.cpp'
$harness = Join-Path $RepositoryRoot 'scripts\build-fg-early-l9-rtx.ps1'
$module = Join-Path $JobRoot 'inputs\cu-module-000014.bin'
$attemptRoot = Join-Path $JobRoot (Join-Path 'output' $AttemptId)
$buildRoot = Join-Path $attemptRoot 'build'
foreach ($required in @($source, $contract, $baseSource, $harness, $module)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Missing L9 input: $required"
    }
}
if (Test-Path -LiteralPath $attemptRoot) { throw "Attempt exists: $attemptRoot" }
if ((Get-Item -LiteralPath $module).Length -ne 4584 -or
    (Get-FileHash -LiteralPath $module -Algorithm SHA256).Hash -ne
        '540F686FFCD4508212B19296992A69ED81CA95772FD229D0A252905C8015419C') {
    throw 'L9 module identity mismatch'
}
$cudaPath = $env:CUDA_PATH
if (-not $cudaPath) { $cudaPath = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3' }
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
New-Item -ItemType Directory -Path $buildRoot | Out-Null
$binary = Join-Path $buildRoot 'fg-early-l9-rtx.exe'
$object = Join-Path $buildRoot 'fg-early-l9-rtx.obj'
$compileLog = Join-Path $buildRoot 'compile.log'
$compileCommand = '"{0}" >nul && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /I"{1}" /Fo"{2}" /c "{3}" >"{4}" 2>&1 && link.exe /nologo /OUT:"{5}" "{2}" "{6}" bcrypt.lib >>"{4}" 2>&1' -f `
    $vcvars, (Join-Path $cudaPath 'include'), $object, $source, $compileLog,
    $binary, (Join-Path $cudaPath 'lib\x64\cuda.lib')
& $env:ComSpec /d /s /c $compileCommand
if ($LASTEXITCODE -ne 0) { throw "L9 build failed; see $compileLog" }

$receipts = @()
foreach ($poison in @('a5', '5a')) {
    foreach ($ordinal in @('a', 'b')) {
        $runId = "l9-poison-$poison-process-$ordinal"
        $runRoot = Join-Path $attemptRoot $runId
        New-Item -ItemType Directory -Path $runRoot | Out-Null
        $process = Start-Process -FilePath $binary `
            -ArgumentList @($module, $runRoot, $runId, $poison) `
            -NoNewWindow -Wait -PassThru `
            -RedirectStandardOutput (Join-Path $runRoot 'stdout.log') `
            -RedirectStandardError (Join-Path $runRoot 'stderr.log')
        if ($process.ExitCode -ne 0) {
            throw "L9 run failed: $runId exit=$($process.ExitCode)"
        }
        $receipts += Get-Content -LiteralPath `
            (Join-Path $runRoot 'run-receipt.json') -Raw | ConvertFrom-Json
    }
}
$processIds = @($receipts | ForEach-Object { [int64]$_.process_id } | Sort-Object -Unique)
$outputHashes = @($receipts | ForEach-Object { $_.output_sha256 } | Sort-Object -Unique)
if ($processIds.Count -ne 4 -or $outputHashes.Count -ne 1) {
    throw 'L9 fresh-process/two-poison invariance failed'
}
$receipt = [ordered]@{
    schema = 'neuralstudio.fg.early-l9-rig.v1'
    job_id = Split-Path -Leaf $JobRoot
    attempt_id = $AttemptId
    local_base_commit = $BaseCommit
    gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total,pci.bus_id --format=csv,noheader,nounits | Select-Object -First 1).Trim()
    cuda_toolkit = (& (Join-Path $cudaPath 'bin\nvcc.exe') --version | Select-Object -Last 1).Trim()
    module_sha256 = (Get-FileHash -LiteralPath $module -Algorithm SHA256).Hash.ToLowerInvariant()
    source_sha256 = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
    contract_sha256 = (Get-FileHash -LiteralPath $contract -Algorithm SHA256).Hash.ToLowerInvariant()
    harness_sha256 = (Get-FileHash -LiteralPath $harness -Algorithm SHA256).Hash.ToLowerInvariant()
    binary_sha256 = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant()
    run_count = 4
    fresh_process_count = 4
    output_sha256 = $outputHashes[0]
    result = 'pass'
}
$utf8 = New-Object System.Text.UTF8Encoding($false)
$receiptPath = Join-Path $attemptRoot 'rig-receipt.json'
[IO.File]::WriteAllText($receiptPath, (($receipt | ConvertTo-Json -Depth 5) + "`n"), $utf8)
Write-Host "attempt_root=$attemptRoot"
Write-Host "rig_receipt_sha256=$((Get-FileHash -LiteralPath $receiptPath -Algorithm SHA256).Hash.ToLowerInvariant())"
Write-Host "output_sha256=$($outputHashes[0])"
Write-Host 'result=pass'
