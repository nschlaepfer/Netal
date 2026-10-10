param(
    [Parameter(Mandatory = $true)] [string] $JobRoot,
    [Parameter(Mandatory = $true)] [string] $RepositoryRoot,
    [Parameter(Mandatory = $true)] [string] $AttemptId
)

$ErrorActionPreference = 'Stop'
$started = (Get-Date).ToUniversalTime().ToString('o')
$sourceRoot = Join-Path $RepositoryRoot 'src\fg_postprocess_l86'
$oracleSource = Join-Path $sourceRoot 'rtx_oracle.cpp'
$bfindSource = Join-Path $sourceRoot 'rtx_bfind_probe.cu'
$modulePath = Join-Path $JobRoot 'inputs\cu-module-000019.bin'
$attemptRoot = Join-Path $JobRoot (Join-Path 'output\fg-postprocess-l86-v1' $AttemptId)
$buildRoot = Join-Path $attemptRoot 'build'

foreach ($required in @($oracleSource, $bfindSource, $modulePath)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required launch-86 input is missing: $required"
    }
}
if (Test-Path -LiteralPath $attemptRoot) {
    throw "Attempt already exists; preserve it and choose a new AttemptId: $attemptRoot"
}
$module = Get-Item -LiteralPath $modulePath
$moduleHash = (Get-FileHash -LiteralPath $modulePath -Algorithm SHA256).Hash
if ($module.Length -ne 57792 -or $moduleHash -ne '143C6712C99DB4A5792DF697D68178AFEEA3657F4F0EEB12F905C90ABF38A1B0') {
    throw "Selected launch-86 module identity mismatch: bytes=$($module.Length) sha256=$moduleHash"
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
$oracleBinary = Join-Path $buildRoot 'fg-postprocess-l86-rtx.exe'
$oracleObject = Join-Path $buildRoot 'fg-postprocess-l86-rtx.obj'
$oracleLog = Join-Path $buildRoot 'oracle-compile.log'
$oracleCommon = '/nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8'
$oracleCommand = '"{0}" >nul && cl.exe {1} /I"{2}" /Fo"{3}" /c "{4}" >"{5}" 2>&1 && link.exe /nologo /OUT:"{6}" "{3}" "{7}" bcrypt.lib >>"{5}" 2>&1' -f `
    $vcvars, $oracleCommon, $cudaInclude, $oracleObject, $oracleSource, `
    $oracleLog, $oracleBinary, $cudaLibrary
& $env:ComSpec /d /s /c $oracleCommand
if ($LASTEXITCODE -ne 0) {
    throw "Launch-86 selected-module build failed with exit code $LASTEXITCODE; see $oracleLog"
}

$bfindBinary = Join-Path $buildRoot 'fg-postprocess-l86-bfind-probe.exe'
$bfindLog = Join-Path $buildRoot 'bfind-compile.log'
$bfindCommand = '"{0}" >nul && "{1}" -std=c++20 -arch=sm_120 -O2 -Xcompiler="/W4 /WX /EHsc /MD /permissive-" "{2}" -o "{3}" >"{4}" 2>&1' -f `
    $vcvars, $nvcc, $bfindSource, $bfindBinary, $bfindLog
& $env:ComSpec /d /s /c $bfindCommand
if ($LASTEXITCODE -ne 0) {
    throw "Launch-86 bfind probe build failed with exit code $LASTEXITCODE; see $bfindLog"
}

$bfindRoot = Join-Path $attemptRoot 'bfind-proof'
New-Item -ItemType Directory -Path $bfindRoot | Out-Null
$bfindStdout = Join-Path $bfindRoot 'stdout.log'
$bfindStderr = Join-Path $bfindRoot 'stderr.log'
$bfindProcess = Start-Process -FilePath $bfindBinary -ArgumentList @($bfindRoot) `
    -NoNewWindow -Wait -PassThru -RedirectStandardOutput $bfindStdout `
    -RedirectStandardError $bfindStderr
if ($bfindProcess.ExitCode -ne 0) {
    throw "Launch-86 bfind proof failed with exit code $($bfindProcess.ExitCode); see $bfindStderr"
}
$bfindReceipt = Get-Content -LiteralPath (Join-Path $bfindRoot 'bfind-receipt.json') -Raw | ConvertFrom-Json
if ($bfindReceipt.output_u32 -ne 2 -or $bfindReceipt.result -ne 'pass') {
    throw 'Launch-86 bfind proof did not establish 0x20000000 -> 2.'
}

$cases = @(
    [ordered]@{ id = 'pyramid-a-poison-a5'; fixture = 'pyramid-a'; poison = 'a5' },
    [ordered]@{ id = 'pyramid-b-poison-a5'; fixture = 'pyramid-b'; poison = 'a5' },
    [ordered]@{ id = 'pyramid-a-poison-5a'; fixture = 'pyramid-a'; poison = '5a' },
    [ordered]@{ id = 'pyramid-b-poison-5a'; fixture = 'pyramid-b'; poison = '5a' }
)
$processResults = @()
$runReceipts = @()
foreach ($case in $cases) {
    $runRoot = Join-Path $attemptRoot $case.id
    New-Item -ItemType Directory -Path $runRoot | Out-Null
    $stdout = Join-Path $runRoot 'stdout.log'
    $stderr = Join-Path $runRoot 'stderr.log'
    $process = Start-Process -FilePath $oracleBinary `
        -ArgumentList @($modulePath, $runRoot, $case.id, $case.fixture, $case.poison, '0') `
        -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr
    if ($process.ExitCode -ne 0) {
        throw "Launch-86 selected-module run $($case.id) failed with exit code $($process.ExitCode); see $stderr"
    }
    $receipt = Get-Content -LiteralPath (Join-Path $runRoot 'run-receipt.json') -Raw | ConvertFrom-Json
    if ($receipt.result -ne 'pass') {
        throw "Launch-86 selected-module receipt did not pass: $($case.id)"
    }
    $runReceipts += $receipt
    $processResults += [ordered]@{
        run_id = $case.id
        process_id = $process.Id
        exit_code = $process.ExitCode
    }
}
$oraclePids = @($runReceipts | ForEach-Object { [int64]$_.process_id } | Sort-Object -Unique)
if ($oraclePids.Count -ne 4) {
    throw "Fresh-process gate failed: selected=$($oraclePids.Count)"
}

foreach ($fixture in @('pyramid-a', 'pyramid-b')) {
    $a5 = $runReceipts | Where-Object { $_.fixture -eq $fixture -and $_.poison_byte -eq 165 }
    $fiveA = $runReceipts | Where-Object { $_.fixture -eq $fixture -and $_.poison_byte -eq 90 }
    if ($a5.level0_after_sha256 -ne $fiveA.level0_after_sha256) {
        throw "Two-poison invariance failed for $fixture"
    }
}

$negativeRoot = Join-Path $attemptRoot 'reject-mode-1'
New-Item -ItemType Directory -Path $negativeRoot | Out-Null
$negativeProcess = Start-Process -FilePath $oracleBinary `
    -ArgumentList @($modulePath, $negativeRoot, 'reject-mode-1', 'pyramid-a', 'a5', '1') `
    -NoNewWindow -Wait -PassThru `
    -RedirectStandardOutput (Join-Path $negativeRoot 'stdout.log') `
    -RedirectStandardError (Join-Path $negativeRoot 'stderr.log')
if ($negativeProcess.ExitCode -ne 4) {
    throw "Mode-1 rejection gate failed: exit=$($negativeProcess.ExitCode)"
}
if (Test-Path -LiteralPath (Join-Path $negativeRoot 'run-receipt.json')) {
    throw 'Rejected mode unexpectedly produced a live-execution receipt.'
}

$gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader,nounits | Select-Object -First 1).Trim()
$cudaVersion = (& $nvcc --version | Select-Object -Last 1).Trim()
$sourceArtifacts = @($oracleSource, $bfindSource, $MyInvocation.MyCommand.Path) | ForEach-Object {
    [ordered]@{
        relative_path = [IO.Path]::GetFileName($_)
        bytes = (Get-Item -LiteralPath $_).Length
        sha256 = (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$rigReceipt = [ordered]@{
    schema = 'neuralstudio.fg.postprocess-l86-rig.v1'
    job_id = Split-Path -Leaf $JobRoot
    attempt_id = $AttemptId
    started_utc = $started
    ended_utc = (Get-Date).ToUniversalTime().ToString('o')
    gpu = $gpu
    cuda_toolkit = $cudaVersion
    visual_studio = $installation
    selected_module = [ordered]@{
        job_relative_path = 'inputs/cu-module-000019.bin'
        token = 19
        bytes = 57792
        sha256 = $moduleHash.ToLowerInvariant()
        transport = 'cuModuleLoadDataEx-exact-selected-module'
    }
    selected_launch_token = 86
    raw_parameter_sha256 = 'faea46ab664868a3ad3bcdcf66dbb7350fddd82fb8d65031b2ea2cec12eb8925'
    address_free_parameter_sha256 = '8f2b991ee0a0288d0f79777881720354e691b920393d921ff53a30b7f27d5fc9'
    mode = 0
    bfind_constant_proof = $bfindReceipt
    process_results = $processResults
    output_hashes = @($runReceipts | ForEach-Object {
        [ordered]@{ run_id = $_.run_id; fixture = $_.fixture; poison_byte = $_.poison_byte; sha256 = $_.level0_after_sha256 }
    })
    fresh_process_count = $oraclePids.Count
    poison_values = @(90, 165)
    independent_fixture_count = 2
    two_poison_output_invariance = $true
    readonly_level_mismatched_bytes = [int64](($runReceipts | Measure-Object -Property readonly_level_mismatched_bytes -Sum).Sum)
    preservation_mismatched_bytes = [int64](($runReceipts | Measure-Object -Property preservation_mismatched_bytes -Sum).Sum)
    mode_1_rejected = $negativeProcess.ExitCode -eq 4
    source_artifacts = $sourceArtifacts
    binaries = @(
        [ordered]@{ relative_path = 'build/fg-postprocess-l86-rtx.exe'; bytes = (Get-Item $oracleBinary).Length; sha256 = (Get-FileHash $oracleBinary -Algorithm SHA256).Hash.ToLowerInvariant() },
        [ordered]@{ relative_path = 'build/fg-postprocess-l86-bfind-probe.exe'; bytes = (Get-Item $bfindBinary).Length; sha256 = (Get-FileHash $bfindBinary -Algorithm SHA256).Hash.ToLowerInvariant() }
    )
    external_expected_tensors_consumed = $false
    archived_intermediates_consumed = $false
    address_fields_recorded = $false
    result = 'pass'
}
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$rigReceiptPath = Join-Path $attemptRoot 'rig-receipt.json'
[IO.File]::WriteAllText($rigReceiptPath, (($rigReceipt | ConvertTo-Json -Depth 12) + "`n"), $utf8NoBom)
$rigHash = (Get-FileHash -LiteralPath $rigReceiptPath -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText((Join-Path $attemptRoot 'rig-receipt.sha256'), ($rigHash + "  rig-receipt.json`n"), $utf8NoBom)

Write-Host "attempt_root=$attemptRoot"
Write-Host "rig_receipt_sha256=$rigHash"
Write-Host 'result=pass'
