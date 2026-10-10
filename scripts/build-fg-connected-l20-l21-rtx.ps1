[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$JobRoot,
    [string]$AttemptId,
    [string]$RepositoryRoot,
    [string]$BaseCommit = '32234a7433ae3e2eddf43519a00a6145a9713a82'
)

$ErrorActionPreference = 'Stop'
$expectedConvHash = '7F9C1F110E8EC9347AA968BC9019E5FFC8839057987A2E5676A0F64637C8DF2D'
$expectedPoolHash = 'A29A3D7041CBE7E44707E74DDDF7AFF61871A2397D1C097EF913BD753A149977'
$expectedWeightHash = '44595B02307A6B934A124ABE476CFD327440E2ACBA7F5ED69F73195CA743E3D9'
$started = (Get-Date).ToUniversalTime().ToString('o')
if (-not $RepositoryRoot) {
    $RepositoryRoot = Split-Path -Parent $PSScriptRoot
}
if (-not $AttemptId) {
    $AttemptId = 'attempt-' + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
}
$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
$JobRoot = [IO.Path]::GetFullPath($JobRoot)
$source = Join-Path $RepositoryRoot 'src\fg_connected_l20_l21\rtx_oracle.cpp'
$verifier = Join-Path $RepositoryRoot 'scripts\verify-fg-connected-l20-l21.py'
$harness = Join-Path $RepositoryRoot 'scripts\build-fg-connected-l20-l21-rtx.ps1'
$convModule = Join-Path $JobRoot 'inputs\cu-module-000028.bin'
$poolModule = Join-Path $JobRoot 'inputs\cu-module-000029.bin'
$weights = Join-Path $JobRoot 'inputs\default-token-000063.bin'
$attemptRoot = Join-Path $JobRoot (Join-Path 'output' $AttemptId)
$buildRoot = Join-Path $attemptRoot 'build'

foreach ($required in @($source, $verifier, $harness, $convModule, $poolModule, $weights)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required connected-group input is missing: $required"
    }
}
if (Test-Path -LiteralPath $attemptRoot) {
    throw "Attempt directory already exists; preserve it: $attemptRoot"
}
$convHash = (Get-FileHash -LiteralPath $convModule -Algorithm SHA256).Hash
$poolHash = (Get-FileHash -LiteralPath $poolModule -Algorithm SHA256).Hash
$weightHash = (Get-FileHash -LiteralPath $weights -Algorithm SHA256).Hash
if ((Get-Item -LiteralPath $convModule).Length -ne 16512 -or $convHash -ne $expectedConvHash) {
    throw "Selected convolution module identity mismatch"
}
if ((Get-Item -LiteralPath $poolModule).Length -ne 2032 -or $poolHash -ne $expectedPoolHash) {
    throw "Selected pooling module identity mismatch"
}
if ((Get-Item -LiteralPath $weights).Length -ne 56320 -or $weightHash -ne $expectedWeightHash) {
    throw "Live weight identity mismatch"
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
$binary = Join-Path $buildRoot 'fg-connected-l20-l21-rtx.exe'
$object = Join-Path $buildRoot 'fg-connected-l20-l21-rtx.obj'
$compileLog = Join-Path $buildRoot 'compile.log'
$common = '/nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8'
$compileCommand = '"{0}" >nul && cl.exe {1} /I"{2}" /Fo"{3}" /c "{4}" >"{5}" 2>&1 && link.exe /nologo /OUT:"{6}" "{3}" "{7}" bcrypt.lib >>"{5}" 2>&1' -f `
    $vcvars, $common, $cudaInclude, $object, $source, $compileLog, $binary, $cudaLibrary
& $env:ComSpec /d /s /c $compileCommand
if ($LASTEXITCODE -ne 0) {
    throw "Connected-group CUDA Driver build failed; see $compileLog"
}

$runReceipts = @()
foreach ($poison in @('a5', '5a')) {
    foreach ($ordinal in @('a', 'b')) {
        $runId = "connected-poison-$poison-process-$ordinal"
        $runRoot = Join-Path $attemptRoot $runId
        New-Item -ItemType Directory -Path $runRoot | Out-Null
        $process = Start-Process -FilePath $binary `
            -ArgumentList @($convModule, $poolModule, $weights, $runRoot, $runId, $poison, '0') `
            -NoNewWindow -Wait -PassThru `
            -RedirectStandardOutput (Join-Path $runRoot 'stdout.log') `
            -RedirectStandardError (Join-Path $runRoot 'stderr.log')
        if ($process.ExitCode -ne 0) {
            throw "Connected-group run $runId failed with exit code $($process.ExitCode)"
        }
        $receipt = Get-Content -LiteralPath (Join-Path $runRoot 'run-receipt.json') -Raw | ConvertFrom-Json
        if ($receipt.result -ne 'pass') {
            throw "Connected-group receipt failed: $runId"
        }
        $runReceipts += $receipt
    }
}
$processIds = @($runReceipts | ForEach-Object { [int64]$_.process_id } | Sort-Object -Unique)
if ($processIds.Count -ne 4) {
    throw "Fresh-process gate failed: unique process count=$($processIds.Count)"
}
$intermediateHashes = @($runReceipts | ForEach-Object { $_.intermediate_sha256 } | Sort-Object -Unique)
$outputHashes = @($runReceipts | ForEach-Object { $_.output_sha256 } | Sort-Object -Unique)
if ($intermediateHashes.Count -ne 1 -or $outputHashes.Count -ne 1) {
    throw 'Two-process/two-poison connected output invariance failed'
}

$backendReceipt = Join-Path $attemptRoot 'backend-receipt.json'
& python $verifier backend --backend rtx --root $attemptRoot --output $backendReceipt
if ($LASTEXITCODE -ne 0) {
    throw "Connected-group backend verification failed with exit code $LASTEXITCODE"
}
$gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total,pci.bus_id --format=csv,noheader,nounits | Select-Object -First 1).Trim()
$cudaVersion = (& (Join-Path $cudaPath 'bin\nvcc.exe') --version | Select-Object -Last 1).Trim()
$rigReceipt = [ordered]@{
    schema = 'neuralstudio.fg.connected-l20-l21-rig.v1'
    job_id = Split-Path -Leaf $JobRoot
    attempt_id = $AttemptId
    local_base_commit = $BaseCommit
    started_utc = $started
    ended_utc = (Get-Date).ToUniversalTime().ToString('o')
    gpu = $gpu
    cuda_toolkit = $cudaVersion
    visual_studio = $installation
    inputs = @(
        [ordered]@{ relative_path = 'inputs/cu-module-000028.bin'; bytes = 16512; sha256 = $convHash.ToLowerInvariant() },
        [ordered]@{ relative_path = 'inputs/cu-module-000029.bin'; bytes = 2032; sha256 = $poolHash.ToLowerInvariant() },
        [ordered]@{ relative_path = 'inputs/default-token-000063.bin'; bytes = 56320; sha256 = $weightHash.ToLowerInvariant(); provenance = 'post-evaluate-0-fence-pre-evaluate-1' }
    )
    source = [ordered]@{
        relative_path = 'src/fg_connected_l20_l21/rtx_oracle.cpp'
        bytes = (Get-Item -LiteralPath $source).Length
        sha256 = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    harness = @(
        [ordered]@{
            relative_path = 'scripts/build-fg-connected-l20-l21-rtx.ps1'
            bytes = (Get-Item -LiteralPath $harness).Length
            sha256 = (Get-FileHash -LiteralPath $harness -Algorithm SHA256).Hash.ToLowerInvariant()
        },
        [ordered]@{
            relative_path = 'scripts/verify-fg-connected-l20-l21.py'
            bytes = (Get-Item -LiteralPath $verifier).Length
            sha256 = (Get-FileHash -LiteralPath $verifier -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    )
    binary = [ordered]@{
        relative_path = "output/$AttemptId/build/fg-connected-l20-l21-rtx.exe"
        bytes = (Get-Item -LiteralPath $binary).Length
        sha256 = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    selected_launches = @(20, 21)
    run_count = $runReceipts.Count
    fresh_process_count = $processIds.Count
    intermediate_sha256 = $intermediateHashes[0]
    output_sha256 = $outputHashes[0]
    expected_output_fed_to_execution = $false
    archived_intermediate_fed_to_execution = $false
    address_fields_recorded = $false
    result = 'pass'
}
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$rigReceiptPath = Join-Path $attemptRoot 'rig-receipt.json'
[IO.File]::WriteAllText($rigReceiptPath, (($rigReceipt | ConvertTo-Json -Depth 8) + "`n"), $utf8NoBom)
$rigHash = (Get-FileHash -LiteralPath $rigReceiptPath -Algorithm SHA256).Hash.ToLowerInvariant()
$backendHash = (Get-FileHash -LiteralPath $backendReceipt -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText((Join-Path $attemptRoot 'rig-receipt.sha256'), ($rigHash + "  rig-receipt.json`n"), $utf8NoBom)
[IO.File]::WriteAllText((Join-Path $attemptRoot 'backend-receipt.sha256'), ($backendHash + "  backend-receipt.json`n"), $utf8NoBom)
Write-Host "attempt_root=$attemptRoot"
Write-Host "rig_receipt_sha256=$rigHash"
Write-Host "backend_receipt_sha256=$backendHash"
Write-Host 'result=pass'
