[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$JobRoot,
    [string]$AttemptId,
    [string]$RepositoryRoot
)

$ErrorActionPreference = 'Stop'
$expectedModuleHash = '7F9C1F110E8EC9347AA968BC9019E5FFC8839057987A2E5676A0F64637C8DF2D'
$expectedModuleBytes = 16512
$started = (Get-Date).ToUniversalTime().ToString('o')
if (-not $RepositoryRoot) {
    $RepositoryRoot = Split-Path -Parent $PSScriptRoot
}
if (-not $AttemptId) {
    $AttemptId = 'attempt-' + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
}
$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
$JobRoot = [IO.Path]::GetFullPath($JobRoot)
$source = Join-Path $RepositoryRoot 'src\fg_conv\rtx_oracle.cpp'
$verifier = Join-Path $RepositoryRoot 'scripts\verify-fg-conv-l20.py'
$module = Join-Path $JobRoot 'inputs\cu-module-000028.bin'
$attemptRoot = Join-Path $JobRoot (Join-Path 'output' $AttemptId)
$buildRoot = Join-Path $attemptRoot 'build'

foreach ($required in @($source, $verifier, $module)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required FG convolution input is missing: $required"
    }
}
if (Test-Path -LiteralPath $attemptRoot) {
    throw "Attempt directory already exists; preserve it: $attemptRoot"
}
$moduleItem = Get-Item -LiteralPath $module
$moduleHash = (Get-FileHash -LiteralPath $module -Algorithm SHA256).Hash
if ($moduleItem.Length -ne $expectedModuleBytes -or $moduleHash -ne $expectedModuleHash) {
    throw "Selected module identity mismatch: bytes=$($moduleItem.Length) sha256=$moduleHash"
}

$cudaPath = $env:CUDA_PATH
if (-not $cudaPath) {
    $cudaPath = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
}
$cudaInclude = Join-Path $cudaPath 'include'
$cudaLibrary = Join-Path $cudaPath 'lib\x64\cuda.lib'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $installation) {
    throw 'Visual Studio C++ x64 build tools were not found.'
}
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'

New-Item -ItemType Directory -Path $buildRoot | Out-Null
$binary = Join-Path $buildRoot 'fg-conv-l20-rtx.exe'
$object = Join-Path $buildRoot 'fg-conv-l20-rtx.obj'
$compileLog = Join-Path $buildRoot 'compile.log'
$common = '/nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8'
$compileCommand = '"{0}" >nul && cl.exe {1} /I"{2}" /Fo"{3}" /c "{4}" >"{5}" 2>&1 && link.exe /nologo /OUT:"{6}" "{3}" "{7}" bcrypt.lib >>"{5}" 2>&1' -f `
    $vcvars, $common, $cudaInclude, $object, $source, $compileLog, $binary, $cudaLibrary
& $env:ComSpec /d /s /c $compileCommand
if ($LASTEXITCODE -ne 0) {
    throw "FG convolution CUDA Driver build failed; see $compileLog"
}

$scenarios = @('spatial-map', 'one-hot', 'bias')
$poisons = @('a5', '5a')
$runReceipts = @()
foreach ($scenario in $scenarios) {
    foreach ($poison in $poisons) {
        foreach ($ordinal in @('a', 'b')) {
            $runId = "$scenario-poison-$poison-process-$ordinal"
            $runRoot = Join-Path $attemptRoot $runId
            New-Item -ItemType Directory -Path $runRoot | Out-Null
            $process = Start-Process -FilePath $binary `
                -ArgumentList @($module, $runRoot, $runId, $poison, $scenario) `
                -NoNewWindow -Wait -PassThru `
                -RedirectStandardOutput (Join-Path $runRoot 'stdout.log') `
                -RedirectStandardError (Join-Path $runRoot 'stderr.log')
            if ($process.ExitCode -ne 0) {
                throw "FG convolution run $runId failed with exit code $($process.ExitCode)"
            }
            $receipt = Get-Content -LiteralPath (Join-Path $runRoot 'run-receipt.json') -Raw | ConvertFrom-Json
            if ($receipt.result -ne 'pass') {
                throw "FG convolution run receipt failed: $runId"
            }
            $runReceipts += $receipt
        }
    }
}

$processIds = @($runReceipts | ForEach-Object { [int64]$_.process_id } | Sort-Object -Unique)
if ($processIds.Count -ne 12) {
    throw "Fresh-process gate failed: unique process count=$($processIds.Count)"
}
foreach ($scenario in $scenarios) {
    $scenarioReceipts = @($runReceipts | Where-Object { $_.scenario -eq $scenario })
    $outputHashes = @($scenarioReceipts | ForEach-Object { $_.output_sha256 } | Sort-Object -Unique)
    if ($outputHashes.Count -ne 1) {
        throw "Two-process/two-poison output invariance failed for $scenario"
    }
}

$gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits | Select-Object -First 1).Trim()
$cudaVersion = (& (Join-Path $cudaPath 'bin\nvcc.exe') --version | Select-Object -Last 1).Trim()
$rigReceipt = [ordered]@{
    schema = 'neuralstudio.fg.conv-l20-rig.v1'
    job_id = Split-Path -Leaf $JobRoot
    attempt_id = $AttemptId
    started_utc = $started
    ended_utc = (Get-Date).ToUniversalTime().ToString('o')
    gpu = $gpu
    cuda_toolkit = $cudaVersion
    visual_studio = $installation
    selected_module = [ordered]@{
        relative_path = 'inputs/cu-module-000028.bin'
        token = 28
        bytes = $moduleItem.Length
        sha256 = $moduleHash.ToLowerInvariant()
        transport = 'cuModuleLoadData-exact-selected-module'
    }
    launch = [ordered]@{
        order = 20
        function = 'k_conv_fp16_nhwc'
        grid = @(10, 8, 1)
        block = @(32, 4, 1)
        packed_parameter_bytes = 48
        abi_scalars = @(160, 96, 162, 98, 160, 96)
    }
    source = [ordered]@{
        relative_path = 'src/fg_conv/rtx_oracle.cpp'
        bytes = (Get-Item -LiteralPath $source).Length
        sha256 = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    binary = [ordered]@{
        relative_path = "output/$AttemptId/build/fg-conv-l20-rtx.exe"
        bytes = (Get-Item -LiteralPath $binary).Length
        sha256 = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    scenarios = $scenarios
    run_count = $runReceipts.Count
    fresh_process_count = $processIds.Count
    output_sha256_by_scenario = [ordered]@{
        'spatial-map' = ($runReceipts | Where-Object { $_.scenario -eq 'spatial-map' } | Select-Object -First 1).output_sha256
        'one-hot' = ($runReceipts | Where-Object { $_.scenario -eq 'one-hot' } | Select-Object -First 1).output_sha256
        'bias' = ($runReceipts | Where-Object { $_.scenario -eq 'bias' } | Select-Object -First 1).output_sha256
    }
    unused_weight_gap_two_poison_probe = $true
    address_fields_recorded = $false
    result = 'pass'
}
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$rigReceiptPath = Join-Path $attemptRoot 'rig-receipt.json'
[IO.File]::WriteAllText($rigReceiptPath, (($rigReceipt | ConvertTo-Json -Depth 8) + "`n"), $utf8NoBom)
$rigReceiptHash = (Get-FileHash -LiteralPath $rigReceiptPath -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText((Join-Path $attemptRoot 'rig-receipt.sha256'), ($rigReceiptHash + "  rig-receipt.json`n"), $utf8NoBom)
$backendReceipt = Join-Path $attemptRoot 'backend-receipt.json'
& python $verifier backend --backend rtx --root $attemptRoot --module $module --output $backendReceipt
if ($LASTEXITCODE -ne 0) {
    throw "FG convolution RTX backend verification failed with exit code $LASTEXITCODE"
}
$backendHash = (Get-FileHash -LiteralPath $backendReceipt -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText((Join-Path $attemptRoot 'backend-receipt.sha256'), ($backendHash + "  backend-receipt.json`n"), $utf8NoBom)
Write-Host "attempt_root=$attemptRoot"
Write-Host "rig_receipt_sha256=$rigReceiptHash"
Write-Host "backend_receipt_sha256=$backendHash"
Write-Host 'result=pass'
