param(
    [Parameter(Mandatory = $true)][string]$JobRoot,
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [Parameter(Mandatory = $true)][string]$AttemptId
)

$ErrorActionPreference = 'Stop'
$started = (Get-Date).ToUniversalTime().ToString('o')
$source = Join-Path $RepositoryRoot 'src\fg_postprocess_l84\rtx_nvapi_oracle.cpp'
$sigmoidSource = Join-Path $RepositoryRoot 'src\fg_postprocess_l84\rtx_sigmoid_probe.cu'
$verifier = Join-Path $RepositoryRoot 'scripts\verify-fg-postprocess-l84.py'
$module = Join-Path $JobRoot 'input\cu-module-000017.bin'
$attemptRoot = Join-Path $JobRoot (Join-Path 'output\fg-postprocess-l84-v1' $AttemptId)
$buildRoot = Join-Path $attemptRoot 'build'
if (-not $JobRoot.StartsWith('D:\CodexLab\')) { throw 'JobRoot must be beneath D:\CodexLab' }
foreach ($required in @($source, $sigmoidSource, $verifier, $module)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Missing launch-84 input: $required"
    }
}
if (Test-Path -LiteralPath $attemptRoot) {
    throw "Attempt already exists; preserve it and choose another id: $attemptRoot"
}
$moduleItem = Get-Item -LiteralPath $module
$moduleHash = (Get-FileHash -LiteralPath $module -Algorithm SHA256).Hash.ToLowerInvariant()
if ($moduleItem.Length -ne 33936 -or $moduleHash -ne '3d8c6aeba3a82fdeb99672243a3d392776bd1742576ab44e8ef42af105c2fd76') {
    throw "Selected module identity mismatch: bytes=$($moduleItem.Length) sha256=$moduleHash"
}

$cudaPath = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3' }
$cudaInclude = Join-Path $cudaPath 'include'
$cudaLibrary = Join-Path $cudaPath 'lib\x64\cuda.lib'
$nvcc = Join-Path $cudaPath 'bin\nvcc.exe'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $installation) { throw 'Visual Studio C++ build tools were not found' }
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
foreach ($required in @((Join-Path $cudaInclude 'cuda.h'), $cudaLibrary, $nvcc, $vcvars)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Toolchain input missing: $required" }
}

New-Item -ItemType Directory -Path $buildRoot | Out-Null
$binary = Join-Path $buildRoot 'fg-postprocess-l84-rtx.exe'
$object = Join-Path $buildRoot 'fg-postprocess-l84-rtx.obj'
$compileLog = Join-Path $buildRoot 'compile.log'
$common = '/nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8'
$command = '"{0}" >nul && cl.exe {1} /I"{2}" /Fo"{3}" /c "{4}" >"{5}" 2>&1 && link.exe /nologo /OUT:"{6}" "{3}" "{7}" bcrypt.lib d3d12.lib dxgi.lib >>"{5}" 2>&1' -f $vcvars, $common, $cudaInclude, $object, $source, $compileLog, $binary, $cudaLibrary
& $env:ComSpec /d /s /c $command
if ($LASTEXITCODE -ne 0) { throw "Launch-84 build failed ($LASTEXITCODE); see $compileLog" }

$sigmoidBinary = Join-Path $buildRoot 'fg-postprocess-l84-sigmoid-probe.exe'
$sigmoidCompileLog = Join-Path $buildRoot 'sigmoid-compile.log'
$sigmoidCommand = '"{0}" >nul && "{1}" -std=c++20 -arch=sm_120 -O2 -Xcompiler="/W4 /WX /wd4211 /EHsc /MD /permissive-" "{2}" bcrypt.lib -o "{3}" >"{4}" 2>&1' -f $vcvars, $nvcc, $sigmoidSource, $sigmoidBinary, $sigmoidCompileLog
& $env:ComSpec /d /s /c $sigmoidCommand
if ($LASTEXITCODE -ne 0) { throw "Launch-84 sigmoid probe build failed ($LASTEXITCODE); see $sigmoidCompileLog" }

$cases = @(
    [ordered]@{ id = 'open-poison-a5'; fixture = 'open'; poison = 'a5' },
    [ordered]@{ id = 'open-poison-5a'; fixture = 'open'; poison = '5a' },
    [ordered]@{ id = 'islands-poison-a5'; fixture = 'islands'; poison = 'a5' },
    [ordered]@{ id = 'islands-poison-5a'; fixture = 'islands'; poison = '5a' }
)
$processResults = @()
$receipts = @()
foreach ($case in $cases) {
    $runRoot = Join-Path $attemptRoot $case.id
    New-Item -ItemType Directory -Path $runRoot | Out-Null
    $stdout = Join-Path $runRoot 'stdout.log'
    $stderr = Join-Path $runRoot 'stderr.log'
    $process = Start-Process -FilePath $binary -ArgumentList @($module, $runRoot, $case.id, $case.fixture, $case.poison) -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    if ($process.ExitCode -ne 0) { throw "Launch-84 run failed: $($case.id), exit=$($process.ExitCode); see $stderr" }
    $receiptPath = Join-Path $runRoot 'run-receipt.json'
    $receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
    if ($receipt.result -ne 'pass') { throw "Launch-84 receipt failed: $($case.id)" }
    $receipts += $receipt
    $processResults += [ordered]@{ run_id = $case.id; process_id = $process.Id; exit_code = $process.ExitCode }
}
$pids = @($receipts | ForEach-Object { [int64]$_.process_id } | Sort-Object -Unique)
if ($pids.Count -ne 4) { throw "Fresh-process gate failed: $($pids.Count) unique processes" }
foreach ($fixture in @('open', 'islands')) {
    $fixtureReceipts = @($receipts | Where-Object { $_.fixture -eq $fixture })
    if ($fixtureReceipts.Count -ne 2 -or $fixtureReceipts[0].output_root_sha256 -ne $fixtureReceipts[1].output_root_sha256) {
        throw "Two-poison output invariance failed for $fixture"
    }
}

$negativeRoot = Join-Path $attemptRoot 'unsupported-fixture-negative'
New-Item -ItemType Directory -Path $negativeRoot | Out-Null
$negativeProcess = Start-Process -FilePath $binary `
    -ArgumentList @($module, $negativeRoot, 'unsupported-fixture-negative', 'unsupported', 'a5') `
    -NoNewWindow -Wait -PassThru `
    -RedirectStandardOutput (Join-Path $negativeRoot 'stdout.log') `
    -RedirectStandardError (Join-Path $negativeRoot 'stderr.log')
if ($negativeProcess.ExitCode -ne 1) {
    throw "Unsupported fixture did not fail closed before execution: exit=$($negativeProcess.ExitCode)"
}

$sigmoidReceipts = @()
$sigmoidProcesses = @()
foreach ($ordinal in @('a', 'b')) {
    $runId = "sigmoid-process-$ordinal"
    $runRoot = Join-Path $attemptRoot $runId
    New-Item -ItemType Directory -Path $runRoot | Out-Null
    $process = Start-Process -FilePath $sigmoidBinary -ArgumentList @($runRoot, $runId) -NoNewWindow -Wait -PassThru -RedirectStandardOutput (Join-Path $runRoot 'stdout.log') -RedirectStandardError (Join-Path $runRoot 'stderr.log')
    if ($process.ExitCode -ne 0) { throw "Sigmoid probe failed: $runId, exit=$($process.ExitCode)" }
    $receipt = Get-Content -LiteralPath (Join-Path $runRoot 'sigmoid-receipt.json') -Raw | ConvertFrom-Json
    if ($receipt.result -ne 'pass') { throw "Sigmoid receipt failed: $runId" }
    $sigmoidReceipts += $receipt
    $sigmoidProcesses += [ordered]@{ run_id = $runId; process_id = $process.Id; exit_code = $process.ExitCode }
}
$sigmoidPids = @($sigmoidReceipts | ForEach-Object { [int64]$_.process_id } | Sort-Object -Unique)
if ($sigmoidPids.Count -ne 2 -or $sigmoidReceipts[0].lut_sha256 -ne $sigmoidReceipts[1].lut_sha256) {
    throw 'Fresh-process sigmoid LUT repeat gate failed'
}

$rigReceipt = [ordered]@{
    schema = 'neuralstudio.fg.postprocess-l84-rig.v1'
    job_id = Split-Path -Leaf $JobRoot
    attempt_id = $AttemptId
    started_utc = $started
    ended_utc = (Get-Date).ToUniversalTime().ToString('o')
    gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader,nounits | Select-Object -First 1).Trim()
    cuda_toolkit = (& $nvcc --version | Select-Object -Last 1).Trim()
    visual_studio = $installation
    selected_module = [ordered]@{ relative_path = 'input/cu-module-000017.bin'; token = 17; bytes = 33936; sha256 = $moduleHash; transport = 'cuModuleLoadData-exact-selected-module' }
    selected_launch_token = 84
    raw_live_parameter_sha256 = '31cfc2016f76ad79fe92717d4bb3b38ceb6c8b3f14968f6b5f1e2f66503e548a'
    process_results = $processResults
    sigmoid_process_results = $sigmoidProcesses
    sigmoid_lut = [ordered]@{ bytes = 262144; sha256 = $sigmoidReceipts[0].lut_sha256; full_half_bit_domain = $true; repeat_equal = $true }
    sources = @(
        [ordered]@{ relative_path = 'src/fg_postprocess_l84/rtx_nvapi_oracle.cpp'; bytes = (Get-Item $source).Length; sha256 = (Get-FileHash $source -Algorithm SHA256).Hash.ToLowerInvariant() },
        [ordered]@{ relative_path = 'src/fg_postprocess_l84/rtx_sigmoid_probe.cu'; bytes = (Get-Item $sigmoidSource).Length; sha256 = (Get-FileHash $sigmoidSource -Algorithm SHA256).Hash.ToLowerInvariant() },
        [ordered]@{ relative_path = 'scripts/verify-fg-postprocess-l84.py'; bytes = (Get-Item $verifier).Length; sha256 = (Get-FileHash $verifier -Algorithm SHA256).Hash.ToLowerInvariant() }
    )
    binaries = @(
        [ordered]@{ relative_path = 'build/fg-postprocess-l84-rtx.exe'; bytes = (Get-Item $binary).Length; sha256 = (Get-FileHash $binary -Algorithm SHA256).Hash.ToLowerInvariant() },
        [ordered]@{ relative_path = 'build/fg-postprocess-l84-sigmoid-probe.exe'; bytes = (Get-Item $sigmoidBinary).Length; sha256 = (Get-FileHash $sigmoidBinary -Algorithm SHA256).Hash.ToLowerInvariant() }
    )
    fresh_process_count = $pids.Count
    fixture_count = 2
    poison_bytes = @(165, 90)
    every_fixture_two_poison_invariant = $true
    unsupported_fixture_negative = [ordered]@{ process_id = $negativeProcess.Id; exit_code = $negativeProcess.ExitCode; receipt_created = (Test-Path (Join-Path $negativeRoot 'run-receipt.json')); result = 'rejected-before-gpu-execution' }
    sigmoid_lut_fresh_process_count = $sigmoidPids.Count
    expected_output_fed_to_execution = $false
    archived_intermediates_consumed = $false
    address_fields_recorded = $false
    result = 'pass'
}
$utf8 = New-Object System.Text.UTF8Encoding($false)
$rigPath = Join-Path $attemptRoot 'rig-receipt.json'
[IO.File]::WriteAllText($rigPath, (($rigReceipt | ConvertTo-Json -Depth 10) + "`n"), $utf8)
$rigHash = (Get-FileHash $rigPath -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText((Join-Path $attemptRoot 'rig-receipt.sha256'), ($rigHash + "  rig-receipt.json`n"), $utf8)
$backendPath = Join-Path $attemptRoot 'backend-receipt.json'
$backendLog = Join-Path $attemptRoot 'backend-verification.log'
& python $verifier backend --backend rtx --root $attemptRoot --output $backendPath *> $backendLog
if ($LASTEXITCODE -ne 0) { throw "Launch-84 backend verification failed; see $backendLog" }
$backendHash = (Get-FileHash $backendPath -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText((Join-Path $attemptRoot 'backend-receipt.sha256'), ($backendHash + "  backend-receipt.json`n"), $utf8)
Write-Host "attempt_root=$attemptRoot"
Write-Host "rig_receipt_sha256=$rigHash"
Write-Host "backend_receipt_sha256=$backendHash"
Write-Host 'result=pass'
