[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $JobRoot,
    [Parameter(Mandatory = $true)] [string] $RepositoryRoot,
    [Parameter(Mandatory = $true)] [string] $AttemptId
)

$ErrorActionPreference = 'Stop'
$attempt = Join-Path $JobRoot (Join-Path 'output\tail' $AttemptId)
if (Test-Path -LiteralPath $attempt) {
    throw "Attempt exists; preserve it: $attempt"
}
$modules = Join-Path $JobRoot 'input\full-modules-v2'
$model = Join-Path $JobRoot 'input\full-model-v2'
$prefixSource = Join-Path $RepositoryRoot `
    'src\fg_connected_l20_l87\rtx_prefix_bridge.cpp'
$coreSource = Join-Path $RepositoryRoot `
    'src\fg_connected_tail_l84_l87\rtx_core_bridge.cpp'
$tailSource = Join-Path $RepositoryRoot `
    'src\fg_connected_l20_l87\rtx_mid_l51_l56_smoke.cpp'
$prefixGenerator = Join-Path $RepositoryRoot `
    'scripts\generate-fg-connected-subrect-l3-l83-rtx-prefix.py'
$coreGenerator = Join-Path $RepositoryRoot `
    'scripts\generate-fg-connected-subrect-l3-l83-rtx-core.py'
$tailGenerator = Join-Path $RepositoryRoot `
    'scripts\generate-fg-connected-subrect-l3-l83-rtx-tail.py'
foreach ($required in @($modules, $model, $prefixSource, $coreSource,
                         $tailSource, $prefixGenerator, $coreGenerator,
                         $tailGenerator)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required tail input missing: $required"
    }
}

$build = Join-Path $attempt 'build'
$generated = Join-Path $build 'generated'
New-Item -ItemType Directory -Path $generated | Out-Null
$prefix = Join-Path $generated 'rtx_subrect_prefix.cpp'
$core = Join-Path $generated 'rtx_subrect_core.cpp'
$tail = Join-Path $generated 'rtx_subrect_tail.cpp'
& python $prefixGenerator $prefixSource $prefix
if ($LASTEXITCODE -ne 0) { throw 'Prefix source generation failed' }
& python $coreGenerator $coreSource $core
if ($LASTEXITCODE -ne 0) { throw 'Core source generation failed' }
& python $tailGenerator $tailSource $tail
if ($LASTEXITCODE -ne 0) { throw 'Tail source generation failed' }

$cuda = $env:CUDA_PATH
if (-not $cuda) {
    $cuda = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
}
$vswhere = Join-Path ${env:ProgramFiles(x86)} `
    'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $installation) { throw 'Visual Studio C++ tools unavailable' }
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
$prefixObject = Join-Path $build 'prefix.obj'
$coreObject = Join-Path $build 'core.obj'
$tailObject = Join-Path $build 'tail.obj'
$binary = Join-Path $build 'subrect-tail.exe'
$compileLog = Join-Path $build 'compile.log'
$includePrefix = Join-Path $RepositoryRoot 'src\fg_connected_l20_l87'
$includeCore = Join-Path $RepositoryRoot 'src\fg_connected_tail_l84_l87'
$includeSubrect = Join-Path $RepositoryRoot 'src\fg_connected_subrect_l3_l83'
$flags = '/nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8'
$command = '"{0}" >nul && cl.exe {1} /I"{2}" /I"{3}" /I"{4}" /I"{5}" /Fo"{6}" /c "{7}" >"{8}" 2>&1 && cl.exe {1} /I"{2}" /I"{3}" /I"{4}" /I"{5}" /Fo"{9}" /c "{10}" >>"{8}" 2>&1 && cl.exe {1} /I"{2}" /I"{3}" /I"{4}" /I"{5}" /Fo"{11}" /c "{12}" >>"{8}" 2>&1 && link.exe /nologo /OUT:"{13}" "{6}" "{9}" "{11}" "{14}" bcrypt.lib d3d12.lib dxgi.lib >>"{8}" 2>&1' -f `
    $vcvars, $flags, (Join-Path $cuda 'include'), $includePrefix,
    $includeCore, $includeSubrect, $prefixObject, $prefix, $compileLog,
    $coreObject, $core, $tailObject, $tail, $binary,
    (Join-Path $cuda 'lib\x64\cuda.lib')
& $env:ComSpec /d /s /c $command
if ($LASTEXITCODE -ne 0) { throw "Build failed; see $compileLog" }

$cases = @(
    [ordered]@{id='subrect-tail-poison-a5-process-a';poison='a5'},
    [ordered]@{id='subrect-tail-poison-a5-process-b';poison='a5'},
    [ordered]@{id='subrect-tail-poison-5a-process-a';poison='5a'},
    [ordered]@{id='subrect-tail-poison-5a-process-b';poison='5a'}
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
    if ($process.ExitCode -ne 0) {
        throw "Tail run failed: $($case.id), exit $($process.ExitCode)"
    }
    $receipt = Get-Content -LiteralPath (Join-Path $run 'run-receipt.json') `
        -Raw | ConvertFrom-Json
    if ($receipt.result -ne 'pass') { throw "Tail receipt failed: $($case.id)" }
    $processes += [ordered]@{
        run_id = $case.id
        process_id = $process.Id
        poison_byte = $receipt.poison_byte
        checkpoint_sha256 = $receipt.checkpoint_sha256
        output_full_sha256 = $receipt.output_full_sha256
        output_region_sha256 = $receipt.output_region_sha256
    }
}
$checkpointMaps = @($processes | ForEach-Object {
    $_.checkpoint_sha256 | ConvertTo-Json -Compress -Depth 5
})
if (@($processes.process_id | Sort-Object -Unique).Count -ne 4 -or
    @($checkpointMaps | Sort-Object -Unique).Count -ne 1 -or
    @($processes.output_region_sha256 | Sort-Object -Unique).Count -ne 1) {
    throw 'Fresh-process/two-poison region invariance failed'
}
$receipt = [ordered]@{
    schema = 'neuralstudio.fg.connected-subrect-l18-l83-rtx-backend.v1'
    job_id = Split-Path $JobRoot -Leaf
    attempt_id = $AttemptId
    remote_job_retained = $JobRoot
    gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total `
        --format=csv,noheader,nounits | Select-Object -First 1).Trim()
    cuda_toolkit = (& (Join-Path $cuda 'bin\nvcc.exe') --version |
        Select-Object -Last 1).Trim()
    source_sha256 = [ordered]@{
        prefix = (Get-FileHash $prefix -Algorithm SHA256).Hash.ToLowerInvariant()
        core = (Get-FileHash $core -Algorithm SHA256).Hash.ToLowerInvariant()
        tail = (Get-FileHash $tail -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    binary_sha256 = (Get-FileHash $binary -Algorithm SHA256).Hash.ToLowerInvariant()
    compile_log_sha256 =
        (Get-FileHash $compileLog -Algorithm SHA256).Hash.ToLowerInvariant()
    processes = @($processes | ForEach-Object {
        [ordered]@{run_id=$_.run_id;process_id=$_.process_id;
          poison_byte=$_.poison_byte;output_full_sha256=$_.output_full_sha256;
          output_region_sha256=$_.output_region_sha256}
    })
    fresh_process_count = 4
    poison_bytes = @(90,165)
    launch_range = @(18,83)
    launch_count = 66
    checkpoint_sha256 = $processes[0].checkpoint_sha256
    output_region_sha256 = $processes[0].output_region_sha256
    full_surface_poison_invariant = $false
    region_scoped_poison_invariant = $true
    input_model_immutability = $true
    canary_integrity = $true
    live_device_handoffs = $true
    expected_or_archived_tensor_ingress = $false
    address_fields_recorded = $false
    result = 'pass'
}
$utf8NoBom = New-Object Text.UTF8Encoding($false)
[IO.File]::WriteAllText((Join-Path $attempt 'backend-receipt.json'),
    (($receipt | ConvertTo-Json -Depth 8) + "`n"), $utf8NoBom)
Write-Output "attempt_root=$attempt"
Write-Output "output_region_sha256=$($receipt.output_region_sha256)"
Write-Output 'result=pass'
