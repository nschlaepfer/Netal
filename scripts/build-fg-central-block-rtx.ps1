param(
    [Parameter(Mandatory = $true)][string]$JobRoot,
    [Parameter(Mandatory = $true)][string]$AttemptId
)

$ErrorActionPreference = 'Stop'
$started = (Get-Date).ToUniversalTime().ToString('o')
$expectedModuleHash = '184542107e98be3dc6229251c19cae4ecc44a7cc1b78cbf741b7fb3b8d81f060'
$expectedModuleBytes = 5752
$module = Join-Path $JobRoot 'input\cu-module-000061.bin'
$source = Join-Path $JobRoot 'src\rtx_oracle.cpp'
$verifier = Join-Path $JobRoot 'src\verify-fg-central-block.py'
$attemptRoot = Join-Path (Join-Path $JobRoot 'output') $AttemptId
$buildRoot = Join-Path $attemptRoot 'build'

foreach ($required in @($module, $source, $verifier)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required staged file is missing: $required"
    }
}
if (Test-Path -LiteralPath $attemptRoot) {
    throw "Attempt directory already exists; preserve it and choose a new AttemptId: $attemptRoot"
}
$moduleItem = Get-Item -LiteralPath $module
$moduleHash = (Get-FileHash -LiteralPath $module -Algorithm SHA256).Hash.ToLowerInvariant()
if ($moduleItem.Length -ne $expectedModuleBytes -or $moduleHash -ne $expectedModuleHash) {
    throw "Selected module identity mismatch: bytes=$($moduleItem.Length) sha256=$moduleHash"
}

$cudaPath = $env:CUDA_PATH
if (-not $cudaPath) {
    $cudaPath = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
}
$cudaInclude = Join-Path $cudaPath 'include'
$cudaLibrary = Join-Path $cudaPath 'lib\x64\cuda.lib'
foreach ($required in @((Join-Path $cudaInclude 'cuda.h'), $cudaLibrary)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "CUDA Driver development input is missing: $required"
    }
}
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
    throw 'Visual Studio Installer vswhere.exe was not found.'
}
$installation = & $vswhere -latest -products '*' `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $installation) {
    throw 'Visual Studio C++ x64 build tools were not found.'
}
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path -LiteralPath $vcvars -PathType Leaf)) {
    throw "vcvars64.bat was not found: $vcvars"
}

New-Item -ItemType Directory -Path $buildRoot | Out-Null
$binary = Join-Path $buildRoot 'fg-central-block-rtx.exe'
$object = Join-Path $buildRoot 'fg-central-block-rtx.obj'
$compileLog = Join-Path $buildRoot 'compile.log'
$common = '/nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8'
$compileCommand = '"{0}" >nul && cl.exe {1} /I"{2}" /Fo"{3}" /c "{4}" >"{5}" 2>&1 && link.exe /nologo /OUT:"{6}" "{3}" "{7}" bcrypt.lib >>"{5}" 2>&1' -f `
    $vcvars, $common, $cudaInclude, $object, $source, $compileLog, $binary, $cudaLibrary
& $env:ComSpec /d /s /c $compileCommand
if ($LASTEXITCODE -ne 0) {
    throw "FG central-block CUDA Driver build failed with exit code $LASTEXITCODE; see $compileLog"
}

$cases = @(
    [ordered]@{ id = 'poison-a5-process-a'; poison = 'a5' },
    [ordered]@{ id = 'poison-a5-process-b'; poison = 'a5' },
    [ordered]@{ id = 'poison-5a-process-a'; poison = '5a' },
    [ordered]@{ id = 'poison-5a-process-b'; poison = '5a' }
)
$runReceipts = @()
foreach ($case in $cases) {
    $runRoot = Join-Path $attemptRoot $case.id
    New-Item -ItemType Directory -Path $runRoot | Out-Null
    $stdout = Join-Path $runRoot 'stdout.log'
    $stderr = Join-Path $runRoot 'stderr.log'
    $process = Start-Process -FilePath $binary `
        -ArgumentList @($module, $runRoot, $case.id, $case.poison) `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    if ($process.ExitCode -ne 0) {
        throw "FG central-block RTX run $($case.id) failed with exit code $($process.ExitCode); see $stderr"
    }
    $receiptPath = Join-Path $runRoot 'run-receipt.json'
    if (-not (Test-Path -LiteralPath $receiptPath -PathType Leaf)) {
        throw "FG central-block RTX run did not produce a receipt: $receiptPath"
    }
    $receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
    if ($receipt.result -ne 'pass') {
        throw "FG central-block RTX run receipt did not pass: $($case.id)"
    }
    $runReceipts += $receipt
}
$processIds = @($runReceipts | ForEach-Object { [int64]$_.process_id } | Sort-Object -Unique)
if ($processIds.Count -ne 4) {
    throw "Fresh-process gate failed: unique process count=$($processIds.Count)"
}

$gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader,nounits | Select-Object -First 1).Trim()
$cudaVersion = (& (Join-Path $cudaPath 'bin\nvcc.exe') --version | Select-Object -Last 1).Trim()
$sourceArtifacts = @($source, $verifier) | ForEach-Object {
    [ordered]@{
        relative_path = [IO.Path]::GetFileName($_)
        bytes = (Get-Item -LiteralPath $_).Length
        sha256 = (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$rigReceipt = [ordered]@{
    schema = 'neuralstudio.fg.central-block-rig.v1'
    job_id = [IO.Path]::GetFileName($JobRoot.TrimEnd('\'))
    attempt_id = $AttemptId
    started_utc = $started
    ended_utc = (Get-Date).ToUniversalTime().ToString('o')
    exit_code = 0
    gpu = $gpu
    cuda_toolkit = $cudaVersion
    visual_studio = $installation
    selected_module = [ordered]@{
        job_relative_path = 'input/cu-module-000061.bin'
        token = 61
        bytes = $moduleItem.Length
        sha256 = $moduleHash
        embedded_code = 'sm_89-ptx-no-elf'
        transport = 'cuModuleLoadDataEx-exact-selected-module'
    }
    launch = [ordered]@{
        order = 71
        functions = @('k_central_block_original', 'k_central_block')
        grid = @(40, 23, 1)
        block = @(16, 16, 1)
        dynamic_shared_bytes = 0
    }
    source_artifacts = $sourceArtifacts
    binary = [ordered]@{
        relative_path = "output/$AttemptId/build/fg-central-block-rtx.exe"
        bytes = (Get-Item -LiteralPath $binary).Length
        sha256 = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    command = 'four fresh processes; each executes original then optimized entry with identical generated inputs'
    run_ids = @($cases | ForEach-Object { $_.id })
    fresh_process_count = $processIds.Count
    address_fields_recorded = $false
    result = 'pass'
}
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$rigReceiptPath = Join-Path $attemptRoot 'rig-receipt.json'
[IO.File]::WriteAllText(
    $rigReceiptPath,
    (($rigReceipt | ConvertTo-Json -Depth 8) + "`n"),
    $utf8NoBom
)
$rigReceiptHash = (Get-FileHash -LiteralPath $rigReceiptPath -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText(
    (Join-Path $attemptRoot 'rig-receipt.sha256'),
    ($rigReceiptHash + "  rig-receipt.json`n"),
    $utf8NoBom
)

$backendReceipt = Join-Path $attemptRoot 'backend-receipt.json'
& python $verifier backend --backend rtx --root $attemptRoot --module $module --output $backendReceipt
if ($LASTEXITCODE -ne 0) {
    throw "FG central-block RTX backend verification failed with exit code $LASTEXITCODE"
}
$backendHash = (Get-FileHash -LiteralPath $backendReceipt -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText(
    (Join-Path $attemptRoot 'backend-receipt.sha256'),
    ($backendHash + "  backend-receipt.json`n"),
    $utf8NoBom
)
Write-Host "attempt_root=$attemptRoot"
Write-Host "rig_receipt_sha256=$rigReceiptHash"
Write-Host "backend_receipt_sha256=$backendHash"
Write-Host 'result=pass'
