[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $JobRoot,
    [Parameter(Mandatory = $true)] [string] $AttemptId,
    [string] $RepositoryRoot
)

$ErrorActionPreference = 'Stop'
if (-not $RepositoryRoot) {
    $RepositoryRoot = Join-Path $JobRoot 'input\full-source'
}
$attempt = Join-Path $JobRoot (Join-Path 'output' $AttemptId)
if (Test-Path -LiteralPath $attempt) {
    throw "Attempt exists; preserve it: $attempt"
}
$modules = Join-Path $JobRoot 'input\full-modules'
$model = Join-Path $JobRoot 'input\full-model'
$earlyRoot = Join-Path $RepositoryRoot 'src\fg_connected_l3_l19'
$tailRoot = Join-Path $RepositoryRoot 'src\fg_connected_l20_l87'
$early = Join-Path $earlyRoot 'rtx_connected.cpp'
$livePrefix = Join-Path $earlyRoot 'rtx_live_prefix_bridge.cpp'
$tailSource = Join-Path $tailRoot 'rtx_mid_l51_l56_smoke.cpp'
$liveCore = Join-Path $tailRoot 'rtx_live_core_bridge.cpp'
foreach ($required in @($early, $livePrefix, $tailSource, $liveCore,
                         $modules, $model)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required connected source/input missing: $required"
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
$build = Join-Path $attempt 'build'
New-Item -ItemType Directory -Path $build | Out-Null
$generatedTail = Join-Path $build 'rtx_full_tail_live_main.cpp'
$tailText = [IO.File]::ReadAllText($tailSource)
$needle = 'int main(int argc, char** argv) {'
if (($tailText.Split(@($needle), [StringSplitOptions]::None).Count - 1) -ne 1) {
    throw 'Accepted full-tail main signature drifted'
}
$tailText = $tailText.Replace(
    $needle, 'int FgConnectedFullTailMain(int argc, char** argv) {')
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText($generatedTail, $tailText, $utf8NoBom)

$earlyObject = Join-Path $build 'early.obj'
$prefixObject = Join-Path $build 'live-prefix.obj'
$tailObject = Join-Path $build 'full-tail.obj'
$coreObject = Join-Path $build 'live-core.obj'
$binary = Join-Path $build 'fg-connected-l3-l87.exe'
$compileLog = Join-Path $build 'compile.log'
$command = '"{0}" >nul && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /DFG_CONNECTED_FULL_CHAIN /I"{1}" /I"{13}" /Fo"{2}" /c "{3}" >"{4}" 2>&1 && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /I"{1}" /I"{13}" /Fo"{5}" /c "{6}" >>"{4}" 2>&1 && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /I"{1}" /I"{13}" /Fo"{7}" /c "{8}" >>"{4}" 2>&1 && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /I"{1}" /I"{13}" /Fo"{9}" /c "{10}" >>"{4}" 2>&1 && link.exe /nologo /OUT:"{11}" "{2}" "{5}" "{7}" "{9}" "{12}" bcrypt.lib d3d12.lib dxgi.lib >>"{4}" 2>&1' -f `
    $vcvars, (Join-Path $cuda 'include'), $earlyObject, $early,
    $compileLog, $prefixObject, $livePrefix, $tailObject, $generatedTail,
    $coreObject, $liveCore, $binary, (Join-Path $cuda 'lib\x64\cuda.lib'),
    $tailRoot
& $env:ComSpec /d /s /c $command
if ($LASTEXITCODE -ne 0) { throw "Build failed; see $compileLog" }

$cases = @(
    [ordered]@{ id = 'connected-l3-l87-poison-a5-process-a'; poison = 'a5' },
    [ordered]@{ id = 'connected-l3-l87-poison-a5-process-b'; poison = 'a5' },
    [ordered]@{ id = 'connected-l3-l87-poison-5a-process-a'; poison = '5a' },
    [ordered]@{ id = 'connected-l3-l87-poison-5a-process-b'; poison = '5a' }
)
$processes = @()
foreach ($case in $cases) {
    $run = Join-Path $attempt $case.id
    New-Item -ItemType Directory -Path $run | Out-Null
    $process = Start-Process -FilePath $binary `
        -ArgumentList @($modules, $model, $run, $case.id, $case.poison) `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput (Join-Path $run 'stdout.log') `
        -RedirectStandardError (Join-Path $run 'stderr.log')
    $processes += [ordered]@{
        run_id = $case.id
        process_id = $process.Id
        exit_code = $process.ExitCode
    }
    if ($process.ExitCode -ne 0) {
        $failure = [ordered]@{
            schema = 'neuralstudio.fg.connected-l3-l87-rig-failure.v1'
            attempt_id = $AttemptId
            failed_run = $case.id
            process_id = $process.Id
            exit_code = $process.ExitCode
            remote_job_retained = $JobRoot
            result = 'fail'
        }
        $failure | ConvertTo-Json -Depth 5 |
            Set-Content -LiteralPath (Join-Path $attempt 'rig-failure.json') `
                -Encoding ascii
        throw "Connected L3-L87 run failed: $($case.id)"
    }
}

$allCheckpoints = @()
$finalHashes = @()
$runPids = @()
foreach ($case in $cases) {
    $run = Join-Path $attempt $case.id
    $earlyReceipt = Get-Content -LiteralPath `
        (Join-Path $run 'run-receipt.json') -Raw | ConvertFrom-Json
    $tailReceipt = Get-Content -LiteralPath `
        (Join-Path $run 'l20-l87\run-receipt.json') -Raw | ConvertFrom-Json
    if ($earlyReceipt.process_id -ne $tailReceipt.process_id -or
        $earlyReceipt.process_id -ne
            ($processes | Where-Object run_id -eq $case.id).process_id) {
        throw "Same-process proof failed: $($case.id)"
    }
    if ($earlyReceipt.result -ne 'pass' -or $tailReceipt.result -ne 'pass') {
        throw "Run receipt failed: $($case.id)"
    }
    foreach ($key in @('input_mismatched_bytes', 'canary_mismatched_bytes')) {
        if ($earlyReceipt.$key -ne 0) {
            throw "Early integrity gate failed: $($case.id)/$key"
        }
    }
    foreach ($key in @('input_model_mismatched_bytes',
                       'canary_mismatched_bytes',
                       'readonly_tail_mismatched_bytes')) {
        if ($tailReceipt.$key -ne 0) {
            throw "Tail integrity gate failed: $($case.id)/$key"
        }
    }
    if (-not $earlyReceipt.l20_live_handoff -or
        $earlyReceipt.allocation37_poison_halfwords -ne 0 -or
        $tailReceipt.final_output_bytes -ne 921600) {
        throw "Coverage/handoff gate failed: $($case.id)"
    }
    $combined = [ordered]@{}
    foreach ($property in $earlyReceipt.checkpoint_sha256.PSObject.Properties) {
        $combined[$property.Name] = $property.Value
    }
    foreach ($property in $tailReceipt.checkpoint_sha256.PSObject.Properties) {
        if ($combined.Contains($property.Name)) {
            throw "Duplicate checkpoint: $($property.Name)"
        }
        $combined[$property.Name] = $property.Value
    }
    $expected = 3..87 | ForEach-Object { 'launch-{0:D6}' -f $_ }
    if (@($combined.Keys).Count -ne 85 -or
        (Compare-Object @($combined.Keys) $expected)) {
        throw "Checkpoint coverage failed: $($case.id)"
    }
    $combinedReceipt = [ordered]@{
        schema = 'neuralstudio.fg.connected-l3-l87-run.v1'
        backend = 'rtx-cuda-external-memory-d3d12-nvapi'
        run_id = $case.id
        process_id = $earlyReceipt.process_id
        poison_byte = $earlyReceipt.poison_byte
        launch_range = @(3, 87)
        launch_count = 85
        checkpoint_sha256 = $combined
        allocation37_sha256 = $earlyReceipt.allocation37_sha256
        live_device_handoff_l19_l20 = $true
        host_intermediate_ingress = $false
        expected_or_archived_tensor_ingress = $false
        address_fields_recorded = $false
        final_output_bytes = $tailReceipt.final_output_bytes
        final_output_sha256 = $tailReceipt.final_output_sha256
        result = 'pass'
    }
    [IO.File]::WriteAllText(
        (Join-Path $run 'combined-receipt.json'),
        (($combinedReceipt | ConvertTo-Json -Depth 8) + "`n"), $utf8NoBom)
    $allCheckpoints += ($combined | ConvertTo-Json -Compress)
    $finalHashes += $tailReceipt.final_output_sha256
    $runPids += $earlyReceipt.process_id
}
if (@($runPids | Sort-Object -Unique).Count -ne 4 -or
    @($allCheckpoints | Sort-Object -Unique).Count -ne 1 -or
    @($finalHashes | Sort-Object -Unique).Count -ne 1) {
    throw 'Fresh-process/two-poison invariance failed'
}
$receipt = [ordered]@{
    schema = 'neuralstudio.fg.connected-l3-l87-rig.v1'
    job_id = Split-Path $JobRoot -Leaf
    remote_job_retained = $JobRoot
    attempt_id = $AttemptId
    gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total `
        --format=csv,noheader,nounits | Select-Object -First 1).Trim()
    early_source_sha256 = (Get-FileHash $early -Algorithm SHA256).Hash.ToLowerInvariant()
    live_prefix_source_sha256 = (Get-FileHash $livePrefix -Algorithm SHA256).Hash.ToLowerInvariant()
    accepted_tail_source_sha256 = (Get-FileHash $tailSource -Algorithm SHA256).Hash.ToLowerInvariant()
    generated_tail_source_sha256 = (Get-FileHash $generatedTail -Algorithm SHA256).Hash.ToLowerInvariant()
    live_core_source_sha256 = (Get-FileHash $liveCore -Algorithm SHA256).Hash.ToLowerInvariant()
    binary_sha256 = (Get-FileHash $binary -Algorithm SHA256).Hash.ToLowerInvariant()
    processes = $processes
    fresh_process_count = 4
    poison_bytes = @(90, 165)
    launch_range = @(3, 87)
    launch_count = 85
    allocation37_sha256 = (Get-Content (Join-Path $attempt `
        ($cases[0].id + '\run-receipt.json')) -Raw | ConvertFrom-Json).allocation37_sha256
    invariant_output_sha256 = $finalHashes[0]
    all_checkpoint_two_poison_invariant = $true
    full_write_coverage_from_two_poison_invariance = $true
    live_device_handoff_l19_l20 = $true
    host_intermediate_ingress = $false
    expected_or_archived_tensor_ingress = $false
    address_fields_recorded = $false
    result = 'pass'
}
[IO.File]::WriteAllText((Join-Path $attempt 'backend-receipt.json'),
    (($receipt | ConvertTo-Json -Depth 8) + "`n"), $utf8NoBom)
Write-Output "attempt_root=$attempt"
Write-Output "invariant_output_sha256=$($finalHashes[0])"
Write-Output 'result=pass'
