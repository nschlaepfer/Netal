param(
    [Parameter(Mandatory = $true)] [string] $JobRoot,
    [Parameter(Mandatory = $true)] [string] $RepositoryRoot,
    [Parameter(Mandatory = $true)] [string] $AttemptId
)

$ErrorActionPreference = 'Stop'
$started = (Get-Date).ToUniversalTime().ToString('o')
$sourceRoot = Join-Path $RepositoryRoot 'src\fg_postprocess_l85'
$oracleSource = Join-Path $sourceRoot 'rtx_oracle.cpp'
$probeSource = Join-Path $sourceRoot 'rtx_div_probe.cu'
$verifier = Join-Path $RepositoryRoot 'scripts\verify-fg-postprocess-l85.py'
$modulePath = Join-Path $JobRoot 'inputs\cu-module-000018.bin'
$romPath = Join-Path $JobRoot 'inputs\sm120-rcp-correction-rom.bin'
$attemptRoot = Join-Path $JobRoot (Join-Path 'output\fg-postprocess-l85-v1' $AttemptId)
$buildRoot = Join-Path $attemptRoot 'build'

foreach ($required in @($oracleSource, $probeSource, $verifier, $modulePath, $romPath)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required launch-85 input is missing: $required"
    }
}
if (Test-Path -LiteralPath $attemptRoot) {
    throw "Attempt already exists; preserve it and choose a new AttemptId: $attemptRoot"
}
$module = Get-Item -LiteralPath $modulePath
$moduleHash = (Get-FileHash -LiteralPath $modulePath -Algorithm SHA256).Hash
if ($module.Length -ne 15128 -or $moduleHash -ne '07877ED3DCEC1977A5DDFD7843314A5044ED9559BB2D222713906AFDFC609D1C') {
    throw "Selected launch-85 module identity mismatch: bytes=$($module.Length) sha256=$moduleHash"
}
$rom = Get-Item -LiteralPath $romPath
$romHash = (Get-FileHash -LiteralPath $romPath -Algorithm SHA256).Hash
if ($rom.Length -ne 2097152 -or $romHash -ne '7CF4F8CECE6B18B07605EDC9CB601A153346A79B10BF3B93C7EC058D57D29750') {
    throw "SM120 RCP architecture ROM identity mismatch: bytes=$($rom.Length) sha256=$romHash"
}

$cudaPath = $env:CUDA_PATH
if (-not $cudaPath) {
    $cudaPath = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
}
$cudaInclude = Join-Path $cudaPath 'include'
$cudaLibrary = Join-Path $cudaPath 'lib\x64\cuda.lib'
$nvcc = Join-Path $cudaPath 'bin\nvcc.exe'
foreach ($required in @((Join-Path $cudaInclude 'cuda.h'), $cudaLibrary, $nvcc)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "CUDA input is missing: $required"
    }
}
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
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
$oracleBinary = Join-Path $buildRoot 'fg-postprocess-l85-rtx.exe'
$oracleObject = Join-Path $buildRoot 'fg-postprocess-l85-rtx.obj'
$oracleLog = Join-Path $buildRoot 'oracle-compile.log'
$oracleCommon = '/nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8'
$oracleCommand = '"{0}" >nul && cl.exe {1} /I"{2}" /Fo"{3}" /c "{4}" >"{5}" 2>&1 && link.exe /nologo /OUT:"{6}" "{3}" "{7}" bcrypt.lib >>"{5}" 2>&1' -f `
    $vcvars, $oracleCommon, $cudaInclude, $oracleObject, $oracleSource, `
    $oracleLog, $oracleBinary, $cudaLibrary
& $env:ComSpec /d /s /c $oracleCommand
if ($LASTEXITCODE -ne 0) {
    throw "Launch-85 selected-module build failed with exit code $LASTEXITCODE; see $oracleLog"
}

$probeBinary = Join-Path $buildRoot 'fg-postprocess-l85-div-probe.exe'
$probeLog = Join-Path $buildRoot 'div-probe-compile.log'
$probeCommand = '"{0}" >nul && "{1}" -std=c++20 -arch=sm_120 -O2 -Xcompiler="/W4 /WX /wd4211 /EHsc /MD /permissive-" "{2}" bcrypt.lib -o "{3}" >"{4}" 2>&1' -f `
    $vcvars, $nvcc, $probeSource, $probeBinary, $probeLog
& $env:ComSpec /d /s /c $probeCommand
if ($LASTEXITCODE -ne 0) {
    throw "Launch-85 div.approx probe build failed with exit code $LASTEXITCODE; see $probeLog"
}

$cases = @(
    [ordered]@{ id = 'poison-a5-process-a'; poison = 'a5' },
    [ordered]@{ id = 'poison-a5-process-b'; poison = 'a5' },
    [ordered]@{ id = 'poison-5a-process-a'; poison = '5a' },
    [ordered]@{ id = 'poison-5a-process-b'; poison = '5a' }
)
$processResults = @()
$runReceipts = @()
$probeReceipts = @()
foreach ($case in $cases) {
    $runRoot = Join-Path $attemptRoot $case.id
    New-Item -ItemType Directory -Path $runRoot | Out-Null

    $probeStdout = Join-Path $runRoot 'div-probe-stdout.log'
    $probeStderr = Join-Path $runRoot 'div-probe-stderr.log'
    $probeProcess = Start-Process -FilePath $probeBinary `
        -ArgumentList @($runRoot, $case.id) -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $probeStdout -RedirectStandardError $probeStderr
    if ($probeProcess.ExitCode -ne 0) {
        throw "Launch-85 div probe $($case.id) failed with exit code $($probeProcess.ExitCode); see $probeStderr"
    }
    $probeReceiptPath = Join-Path $runRoot 'div-probe-receipt.json'
    $probeReceipt = Get-Content -LiteralPath $probeReceiptPath -Raw | ConvertFrom-Json
    if ($probeReceipt.result -ne 'pass') {
        throw "Launch-85 div probe receipt did not pass: $($case.id)"
    }
    $probeReceipts += $probeReceipt

    $oracleStdout = Join-Path $runRoot 'stdout.log'
    $oracleStderr = Join-Path $runRoot 'stderr.log'
    $oracleProcess = Start-Process -FilePath $oracleBinary `
        -ArgumentList @($modulePath, $runRoot, $case.id, $case.poison) `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $oracleStdout -RedirectStandardError $oracleStderr
    if ($oracleProcess.ExitCode -ne 0) {
        throw "Launch-85 selected-module run $($case.id) failed with exit code $($oracleProcess.ExitCode); see $oracleStderr"
    }
    $runReceiptPath = Join-Path $runRoot 'run-receipt.json'
    $runReceipt = Get-Content -LiteralPath $runReceiptPath -Raw | ConvertFrom-Json
    if ($runReceipt.result -ne 'pass') {
        throw "Launch-85 selected-module receipt did not pass: $($case.id)"
    }
    $runReceipts += $runReceipt
    $processResults += [ordered]@{
        run_id = $case.id
        div_probe_process_id = $probeProcess.Id
        div_probe_exit_code = $probeProcess.ExitCode
        selected_module_process_id = $oracleProcess.Id
        selected_module_exit_code = $oracleProcess.ExitCode
    }
}
$oraclePids = @($runReceipts | ForEach-Object { [int64]$_.process_id } | Sort-Object -Unique)
$probePids = @($probeReceipts | ForEach-Object { [int64]$_.process_id } | Sort-Object -Unique)
if ($oraclePids.Count -ne 4 -or $probePids.Count -ne 4) {
    throw "Fresh-process gate failed: selected=$($oraclePids.Count) div_probe=$($probePids.Count)"
}

$gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader,nounits | Select-Object -First 1).Trim()
$cudaVersion = (& $nvcc --version | Select-Object -Last 1).Trim()
$sourceArtifacts = @($oracleSource, $probeSource, $verifier, $MyInvocation.MyCommand.Path) | ForEach-Object {
    [ordered]@{
        relative_path = [IO.Path]::GetFileName($_)
        bytes = (Get-Item -LiteralPath $_).Length
        sha256 = (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$rigReceipt = [ordered]@{
    schema = 'neuralstudio.fg.postprocess-l85-rig.v1'
    job_id = Split-Path -Leaf $JobRoot
    attempt_id = $AttemptId
    started_utc = $started
    ended_utc = (Get-Date).ToUniversalTime().ToString('o')
    gpu = $gpu
    cuda_toolkit = $cudaVersion
    visual_studio = $installation
    selected_module = [ordered]@{
        job_relative_path = 'inputs/cu-module-000018.bin'
        token = 18
        bytes = 15128
        sha256 = $moduleHash.ToLowerInvariant()
        transport = 'cuModuleLoadDataEx-exact-selected-module'
    }
    selected_launch_token = 85
    address_free_parameter_sha256 = 'ac5ec30d58d9b7a47fc229acce7e3672af982298ba25e577ec4886bcdd929927'
    process_results = $processResults
    source_artifacts = $sourceArtifacts
    binaries = @(
        [ordered]@{ relative_path = 'build/fg-postprocess-l85-rtx.exe'; bytes = (Get-Item $oracleBinary).Length; sha256 = (Get-FileHash $oracleBinary -Algorithm SHA256).Hash.ToLowerInvariant() },
        [ordered]@{ relative_path = 'build/fg-postprocess-l85-div-probe.exe'; bytes = (Get-Item $probeBinary).Length; sha256 = (Get-FileHash $probeBinary -Algorithm SHA256).Hash.ToLowerInvariant() }
    )
    selected_module_fresh_process_count = $oraclePids.Count
    div_probe_fresh_process_count = $probePids.Count
    external_expected_tensors_consumed = $false
    archived_intermediates_consumed = $false
    address_fields_recorded = $false
    result = 'pass'
}
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$rigReceiptPath = Join-Path $attemptRoot 'rig-receipt.json'
[IO.File]::WriteAllText($rigReceiptPath, (($rigReceipt | ConvertTo-Json -Depth 10) + "`n"), $utf8NoBom)
$rigHash = (Get-FileHash -LiteralPath $rigReceiptPath -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText((Join-Path $attemptRoot 'rig-receipt.sha256'), ($rigHash + "  rig-receipt.json`n"), $utf8NoBom)

$backendReceipt = Join-Path $attemptRoot 'backend-receipt.json'
$backendLog = Join-Path $attemptRoot 'backend-verification.log'
& python $verifier backend --backend rtx --root $attemptRoot --output $backendReceipt --rcp-rom $romPath *> $backendLog
if ($LASTEXITCODE -ne 0) {
    throw "Launch-85 RTX backend verification failed with exit code $LASTEXITCODE; see $backendLog"
}
$backendHash = (Get-FileHash -LiteralPath $backendReceipt -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText((Join-Path $attemptRoot 'backend-receipt.sha256'), ($backendHash + "  backend-receipt.json`n"), $utf8NoBom)

Write-Host "attempt_root=$attemptRoot"
Write-Host "rig_receipt_sha256=$rigHash"
Write-Host "backend_receipt_sha256=$backendHash"
Write-Host 'result=pass'
