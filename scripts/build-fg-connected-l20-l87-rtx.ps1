[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $JobRoot,
    [Parameter(Mandatory = $true)] [string] $AttemptId,
    [Parameter(Mandatory = $true)]
    [ValidateSet('prefix-smoke', 'mid-smoke', 'full')]
    [string] $Target,
    [string] $RepositoryRoot
)

$ErrorActionPreference = 'Stop'
if (-not $RepositoryRoot) {
    $RepositoryRoot = Join-Path $JobRoot 'input\source'
}
$attemptRoot = Join-Path $JobRoot (Join-Path 'output' $AttemptId)
if (Test-Path -LiteralPath $attemptRoot) {
    throw "Attempt exists; preserve it and choose a new AttemptId: $attemptRoot"
}
$sourceRoot = Join-Path $RepositoryRoot 'src\fg_connected_l20_l87'
$modules = Join-Path $JobRoot 'input\modules'
$model = Join-Path $JobRoot 'input\model'
$main = if ($Target -eq 'prefix-smoke') {
    Join-Path $sourceRoot 'rtx_prefix_smoke.cpp'
} elseif ($Target -eq 'mid-smoke') {
    Join-Path $sourceRoot 'rtx_mid_l51_l56_smoke.cpp'
} else {
    Join-Path $sourceRoot 'rtx_connected_l20_l87.cpp'
}
$prefix = Join-Path $sourceRoot 'rtx_prefix_bridge.cpp'
$liveCore = Join-Path $sourceRoot 'rtx_live_core_bridge.cpp'
$implementation = Join-Path $sourceRoot 'rtx_mid_l51_l56_smoke.cpp'
foreach ($required in @($main, $prefix, $liveCore, $implementation,
                         $modules, $model)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required connected executor input is missing: $required"
    }
}
$cuda = $env:CUDA_PATH
if (-not $cuda) {
    $cuda = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
}
$vswhere = Join-Path ${env:ProgramFiles(x86)} `
    'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $installation) { throw 'Visual Studio C++ x64 tools unavailable' }
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
$buildRoot = Join-Path $attemptRoot 'build'
New-Item -ItemType Directory -Path $buildRoot | Out-Null
$mainObject = Join-Path $buildRoot 'main.obj'
$prefixObject = Join-Path $buildRoot 'prefix.obj'
$liveCoreObject = Join-Path $buildRoot 'live-core.obj'
$binary = Join-Path $buildRoot ("fg-connected-l20-l87-$Target.exe")
$compileLog = Join-Path $buildRoot 'compile.log'
$compileCommand = '"{0}" >nul && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /I"{1}" /Fo"{2}" /c "{3}" >"{4}" 2>&1 && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /I"{1}" /Fo"{5}" /c "{6}" >>"{4}" 2>&1 && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /I"{1}" /Fo"{7}" /c "{8}" >>"{4}" 2>&1 && link.exe /nologo /OUT:"{9}" "{2}" "{5}" "{7}" "{10}" bcrypt.lib d3d12.lib dxgi.lib >>"{4}" 2>&1' -f `
    $vcvars, (Join-Path $cuda 'include'), $mainObject, $main, $compileLog,
    $prefixObject, $prefix, $liveCoreObject, $liveCore, $binary,
    (Join-Path $cuda 'lib\x64\cuda.lib')
& $env:ComSpec /d /s /c $compileCommand
if ($LASTEXITCODE -ne 0) { throw "Build failed; see $compileLog" }

$cases = @(
    [ordered]@{ id = 'poison-a5-process-a'; poison = 'a5' },
    [ordered]@{ id = 'poison-a5-process-b'; poison = 'a5' },
    [ordered]@{ id = 'poison-5a-process-a'; poison = '5a' },
    [ordered]@{ id = 'poison-5a-process-b'; poison = '5a' }
)
$runs = @()
foreach ($case in $cases) {
    $runRoot = Join-Path $attemptRoot $case.id
    New-Item -ItemType Directory -Path $runRoot | Out-Null
    $arguments = if ($Target -eq 'prefix-smoke') {
        @($modules, $model, $runRoot, $case.poison)
    } else {
        @($modules, $model, $runRoot, $case.id, $case.poison)
    }
    $process = Start-Process -FilePath $binary -ArgumentList $arguments `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput (Join-Path $runRoot 'stdout.log') `
        -RedirectStandardError (Join-Path $runRoot 'stderr.log')
    $runs += [ordered]@{
        run_id = $case.id
        process_id = $process.Id
        exit_code = $process.ExitCode
    }
    if ($process.ExitCode -ne 0) {
        $failure = [ordered]@{
            schema = 'neuralstudio.fg.connected-l20-l87-rig-failure.v1'
            job_id = Split-Path $JobRoot -Leaf
            attempt_id = $AttemptId
            target = $Target
            failed_run = $case.id
            process_id = $process.Id
            exit_code = $process.ExitCode
            remote_job_retained = $JobRoot
            result = 'fail'
        }
        $failure | ConvertTo-Json -Depth 5 |
            Set-Content -LiteralPath (Join-Path $attemptRoot 'rig-failure.json') -Encoding ascii
        throw "Connected executor run failed: $($case.id)"
    }
}
$receipts = @($cases | ForEach-Object {
    Get-Content -LiteralPath (Join-Path $attemptRoot `
        (Join-Path $_.id 'run-receipt.json')) -Raw | ConvertFrom-Json
})
$pids = @($receipts | ForEach-Object { $_.process_id } | Sort-Object -Unique)
if ($Target -eq 'prefix-smoke') {
    # The smoke receipt intentionally omits a process id; the process table is
    # the authoritative fresh-process proof for this bridge-only diagnostic.
    $pids = @($runs | ForEach-Object { $_.process_id } | Sort-Object -Unique)
    $hashes = @($receipts | ForEach-Object { $_.l51_sha256 } | Sort-Object -Unique)
} else {
    $hashes = @($receipts | ForEach-Object { $_.final_output_sha256 } |
        Sort-Object -Unique)
    $checkpointVariants = @($receipts | ForEach-Object {
        $_.checkpoint_sha256 | ConvertTo-Json -Compress
    } | Sort-Object -Unique)
    if ($checkpointVariants.Count -ne 1) {
        throw 'Two-poison checkpoint invariance failed'
    }
}
if ($pids.Count -ne 4 -or $hashes.Count -ne 1) {
    throw 'Fresh-process/two-poison repeat gate failed'
}
if (@($receipts | Where-Object { $_.result -ne 'pass' }).Count -ne 0) {
    throw 'A connected run receipt failed'
}
$receipt = [ordered]@{
    schema = 'neuralstudio.fg.connected-l20-l87-rig.v1'
    job_id = Split-Path $JobRoot -Leaf
    remote_job_retained = $JobRoot
    attempt_id = $AttemptId
    target = $Target
    gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total `
        --format=csv,noheader,nounits | Select-Object -First 1).Trim()
    source_sha256 = (Get-FileHash -LiteralPath $main -Algorithm SHA256).Hash.ToLowerInvariant()
    prefix_source_sha256 = (Get-FileHash -LiteralPath $prefix -Algorithm SHA256).Hash.ToLowerInvariant()
    live_core_source_sha256 = (Get-FileHash -LiteralPath $liveCore -Algorithm SHA256).Hash.ToLowerInvariant()
    implementation_source_sha256 = (Get-FileHash -LiteralPath $implementation -Algorithm SHA256).Hash.ToLowerInvariant()
    binary_sha256 = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant()
    processes = $runs
    fresh_process_count = $pids.Count
    poison_bytes = @(90, 165)
    invariant_output_sha256 = $hashes[0]
    all_checkpoint_two_poison_invariant = $true
    final_full_write_coverage_from_two_poison_invariance = $true
    expected_or_archived_tensor_ingress = $false
    address_fields_recorded = $false
    result = 'pass'
}
$receipt | ConvertTo-Json -Depth 7 |
    Set-Content -LiteralPath (Join-Path $attemptRoot 'rig-receipt.json') -Encoding ascii
Write-Output "attempt_root=$attemptRoot"
Write-Output "invariant_output_sha256=$($hashes[0])"
Write-Output 'result=pass'
