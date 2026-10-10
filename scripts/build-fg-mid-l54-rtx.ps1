[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$JobRoot,
    [string]$AttemptId,
    [string]$RepositoryRoot,
    [string]$BaseCommit = '98754cb'
)
$ErrorActionPreference = 'Stop'
if (-not $RepositoryRoot) { $RepositoryRoot = Split-Path -Parent $PSScriptRoot }
if (-not $AttemptId) { $AttemptId = 'attempt-' + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ') }
$JobRoot = [IO.Path]::GetFullPath($JobRoot)
$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
$source = Join-Path $RepositoryRoot 'src\fg_mid_l54\rtx_oracle.cpp'
$contract = Join-Path $RepositoryRoot 'src\fg_mid_l54\contract.h'
$baseSource = Join-Path $RepositoryRoot 'src\fg_conv\rtx_oracle.cpp'
$harness = Join-Path $RepositoryRoot 'scripts\build-fg-mid-l54-rtx.ps1'
$module = Join-Path $JobRoot 'inputs\cu-module-000026.bin'
$attemptRoot = Join-Path $JobRoot (Join-Path 'output' $AttemptId)
$buildRoot = Join-Path $attemptRoot 'build'
foreach ($required in @($source, $contract, $baseSource, $harness, $module)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Missing L54 input: $required" }
}
if (Test-Path -LiteralPath $attemptRoot) { throw "Attempt exists: $attemptRoot" }
if ((Get-Item -LiteralPath $module).Length -ne 7544 -or (Get-FileHash -LiteralPath $module -Algorithm SHA256).Hash -ne 'D8D41B87895A3119F24DB4A8D4CB1EC4F73613B507E3D1D4DE5DEA0F36FE5DC5') { throw 'Module identity mismatch' }
$cudaPath = $env:CUDA_PATH
if (-not $cudaPath) { $cudaPath = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3' }
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
New-Item -ItemType Directory -Path $buildRoot | Out-Null
$binary = Join-Path $buildRoot 'fg-mid-l54-rtx.exe'
$object = Join-Path $buildRoot 'fg-mid-l54-rtx.obj'
$compileLog = Join-Path $buildRoot 'compile.log'
$compileCommand = '"{0}" >nul && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /I"{1}" /Fo"{2}" /c "{3}" >"{4}" 2>&1 && link.exe /nologo /OUT:"{5}" "{2}" "{6}" bcrypt.lib >>"{4}" 2>&1' -f $vcvars, (Join-Path $cudaPath 'include'), $object, $source, $compileLog, $binary, (Join-Path $cudaPath 'lib\x64\cuda.lib')
& $env:ComSpec /d /s /c $compileCommand
if ($LASTEXITCODE -ne 0) { throw "L54 build failed; see $compileLog" }
$receipts = @()
foreach ($poison in @('a5', '5a')) {
    foreach ($ordinal in @('a', 'b')) {
        $runId = "l54-poison-$poison-process-$ordinal"
        $runRoot = Join-Path $attemptRoot $runId
        New-Item -ItemType Directory -Path $runRoot | Out-Null
        $process = Start-Process -FilePath $binary -ArgumentList @($module, $runRoot, $runId, $poison) -NoNewWindow -Wait -PassThru -RedirectStandardOutput (Join-Path $runRoot 'stdout.log') -RedirectStandardError (Join-Path $runRoot 'stderr.log')
        if ($process.ExitCode -ne 0) { throw "L54 run failed: $runId exit=$($process.ExitCode)" }
        $receipts += Get-Content -LiteralPath (Join-Path $runRoot 'run-receipt.json') -Raw | ConvertFrom-Json
    }
}
$processIds = @($receipts | ForEach-Object { [int64]$_.process_id } | Sort-Object -Unique)
$outputHashes = @($receipts | ForEach-Object { $_.output_sha256 } | Sort-Object -Unique)
if ($processIds.Count -ne 4 -or $outputHashes.Count -ne 1) { throw 'Fresh-process/two-poison invariance failed' }
$rigReceipt = [ordered]@{
    schema = 'neuralstudio.fg.mid-l54-rig.v1'
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
    run_count = $receipts.Count
    fresh_process_count = $processIds.Count
    output_sha256 = $outputHashes[0]
    result = 'pass'
}
$utf8 = New-Object System.Text.UTF8Encoding($false)
$receiptPath = Join-Path $attemptRoot 'rig-receipt.json'
[IO.File]::WriteAllText($receiptPath, (($rigReceipt | ConvertTo-Json -Depth 5) + "`n"), $utf8)
Write-Host "attempt_root=$attemptRoot"
Write-Host "rig_receipt_sha256=$((Get-FileHash -LiteralPath $receiptPath -Algorithm SHA256).Hash.ToLowerInvariant())"
Write-Host "output_sha256=$($outputHashes[0])"
Write-Host 'result=pass'
