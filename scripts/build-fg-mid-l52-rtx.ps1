param(
    [Parameter(Mandatory = $true)][string]$JobRoot,
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [Parameter(Mandatory = $true)][string]$AttemptId
)

$ErrorActionPreference = 'Stop'
$started = (Get-Date).ToUniversalTime().ToString('o')
$source = Join-Path $RepositoryRoot 'src\fg_mid_l52\rtx_oracle.cpp'
$verifier = Join-Path $RepositoryRoot 'scripts\verify-fg-mid-l52.py'
$module = Join-Path $JobRoot 'input\cu-module-000006.bin'
$attemptRoot = Join-Path $JobRoot (Join-Path 'output\fg-mid-l52-v1' $AttemptId)
$buildRoot = Join-Path $attemptRoot 'build'
if (-not $JobRoot.StartsWith('D:\CodexLab\')) { throw 'JobRoot must be beneath D:\CodexLab' }
foreach ($required in @($source, $verifier, $module)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Missing L52 input: $required" }
}
if (Test-Path -LiteralPath $attemptRoot) { throw "Attempt exists; preserve it: $attemptRoot" }
$moduleItem = Get-Item -LiteralPath $module
$moduleHash = (Get-FileHash -LiteralPath $module -Algorithm SHA256).Hash.ToLowerInvariant()
if ($moduleItem.Length -ne 9064 -or $moduleHash -ne 'd45333f43f343d1f38bd304fe76214dc7064f2a7d24ce1ea06bc791867783264') {
    throw "L52 module identity mismatch: bytes=$($moduleItem.Length) sha256=$moduleHash"
}
$cudaPath = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3' }
$cudaInclude = Join-Path $cudaPath 'include'
$cudaLibrary = Join-Path $cudaPath 'lib\x64\cuda.lib'
$nvcc = Join-Path $cudaPath 'bin\nvcc.exe'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $installation) { throw 'Visual Studio C++ tools unavailable' }
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
New-Item -ItemType Directory -Path $buildRoot | Out-Null
$binary = Join-Path $buildRoot 'fg-mid-l52-rtx.exe'
$object = Join-Path $buildRoot 'fg-mid-l52-rtx.obj'
$compileLog = Join-Path $buildRoot 'compile.log'
$common = '/nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8'
$command = '"{0}" >nul && cl.exe {1} /I"{2}" /Fo"{3}" /c "{4}" >"{5}" 2>&1 && link.exe /nologo /OUT:"{6}" "{3}" "{7}" bcrypt.lib d3d12.lib dxgi.lib >>"{5}" 2>&1' -f $vcvars, $common, $cudaInclude, $object, $source, $compileLog, $binary, $cudaLibrary
& $env:ComSpec /d /s /c $command
if ($LASTEXITCODE -ne 0) { throw "L52 build failed ($LASTEXITCODE); see $compileLog" }

$cases = @(
    [ordered]@{ id='binary-depth-poison-a5'; fixture='binary-depth'; poison='a5' },
    [ordered]@{ id='binary-depth-poison-5a'; fixture='binary-depth'; poison='5a' },
    [ordered]@{ id='ramp-depth-poison-a5'; fixture='ramp-depth'; poison='a5' },
    [ordered]@{ id='ramp-depth-poison-5a'; fixture='ramp-depth'; poison='5a' }
)
$receipts = @()
$processResults = @()
foreach ($case in $cases) {
    $runRoot = Join-Path $attemptRoot $case.id
    New-Item -ItemType Directory -Path $runRoot | Out-Null
    $process = Start-Process -FilePath $binary -ArgumentList @($module, $runRoot, $case.id, $case.fixture, $case.poison) -NoNewWindow -Wait -PassThru -RedirectStandardOutput (Join-Path $runRoot 'stdout.log') -RedirectStandardError (Join-Path $runRoot 'stderr.log')
    if ($process.ExitCode -ne 0) { throw "L52 run failed: $($case.id), exit=$($process.ExitCode)" }
    $receipt = Get-Content -LiteralPath (Join-Path $runRoot 'run-receipt.json') -Raw | ConvertFrom-Json
    if ($receipt.result -ne 'pass') { throw "L52 receipt failed: $($case.id)" }
    $receipts += $receipt
    $processResults += [ordered]@{ run_id=$case.id; process_id=$process.Id; exit_code=$process.ExitCode }
}
$pids = @($receipts | ForEach-Object { [int64]$_.process_id } | Sort-Object -Unique)
if ($pids.Count -ne 4) { throw "Fresh-process gate failed: $($pids.Count)" }
foreach ($fixture in @('binary-depth','ramp-depth')) {
    $pair = @($receipts | Where-Object { $_.fixture -eq $fixture })
    if ($pair.Count -ne 2 -or $pair[0].output_root_sha256 -ne $pair[1].output_root_sha256) {
        throw "Two-poison output invariance failed: $fixture"
    }
}
$negativeRoot = Join-Path $attemptRoot 'unsupported-fixture-negative'
New-Item -ItemType Directory -Path $negativeRoot | Out-Null
$negative = Start-Process -FilePath $binary -ArgumentList @($module, $negativeRoot, 'unsupported-fixture-negative', 'unsupported', 'a5') -NoNewWindow -Wait -PassThru -RedirectStandardOutput (Join-Path $negativeRoot 'stdout.log') -RedirectStandardError (Join-Path $negativeRoot 'stderr.log')
if ($negative.ExitCode -ne 1 -or (Test-Path (Join-Path $negativeRoot 'run-receipt.json'))) {
    throw "Unsupported fixture failed to reject before execution: exit=$($negative.ExitCode)"
}
$rig = [ordered]@{
    schema='neuralstudio.fg.mid-l52-rig.v1'
    job_id=(Split-Path -Leaf $JobRoot)
    attempt_id=$AttemptId
    started_utc=$started
    ended_utc=(Get-Date).ToUniversalTime().ToString('o')
    gpu=(& nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader,nounits | Select-Object -First 1).Trim()
    cuda_toolkit=(& $nvcc --version | Select-Object -Last 1).Trim()
    visual_studio=$installation
    selected_module=[ordered]@{ relative_path='input/cu-module-000006.bin'; token=6; bytes=9064; sha256=$moduleHash; transport='NvAPI_D3D12_CreateCuModule-exact-selected-module' }
    selected_launch_token=52
    raw_live_parameter_sha256='b85b7b8f373002d0518a784cfa9e9d8511979be6dd2641106b5c5acfb4a2a01d'
    source=[ordered]@{ relative_path='src/fg_mid_l52/rtx_oracle.cpp'; bytes=(Get-Item $source).Length; sha256=(Get-FileHash $source -Algorithm SHA256).Hash.ToLowerInvariant() }
    verifier=[ordered]@{ relative_path='scripts/verify-fg-mid-l52.py'; bytes=(Get-Item $verifier).Length; sha256=(Get-FileHash $verifier -Algorithm SHA256).Hash.ToLowerInvariant() }
    binary=[ordered]@{ relative_path='build/fg-mid-l52-rtx.exe'; bytes=(Get-Item $binary).Length; sha256=(Get-FileHash $binary -Algorithm SHA256).Hash.ToLowerInvariant() }
    process_results=$processResults
    fresh_process_count=$pids.Count
    fixture_count=2
    poison_bytes=@(90,165)
    every_fixture_two_poison_invariant=$true
    unsupported_fixture_negative=[ordered]@{ process_id=$negative.Id; exit_code=$negative.ExitCode; receipt_created=$false; result='rejected-before-gpu-execution' }
    expected_output_fed_to_execution=$false
    archived_intermediates_consumed=$false
    address_fields_recorded=$false
    result='pass'
}
$utf8 = New-Object System.Text.UTF8Encoding($false)
$rigPath = Join-Path $attemptRoot 'rig-receipt.json'
[IO.File]::WriteAllText($rigPath,(($rig|ConvertTo-Json -Depth 10)+"`n"),$utf8)
$rigHash = (Get-FileHash $rigPath -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText((Join-Path $attemptRoot 'rig-receipt.sha256'),($rigHash+"  rig-receipt.json`n"),$utf8)
$backendPath = Join-Path $attemptRoot 'backend-receipt.json'
& python $verifier backend --backend rtx --root $attemptRoot --output $backendPath *> (Join-Path $attemptRoot 'backend-verification.log')
if ($LASTEXITCODE -ne 0) { throw 'L52 backend verification failed' }
$backendHash = (Get-FileHash $backendPath -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText((Join-Path $attemptRoot 'backend-receipt.sha256'),($backendHash+"  backend-receipt.json`n"),$utf8)
Write-Host "attempt_root=$attemptRoot"
Write-Host "rig_receipt_sha256=$rigHash"
Write-Host "backend_receipt_sha256=$backendHash"
Write-Host 'result=pass'
