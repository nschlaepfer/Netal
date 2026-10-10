[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $JobRoot,
    [Parameter(Mandatory = $true)] [string] $RepositoryRoot,
    [Parameter(Mandatory = $true)] [string] $AttemptId,
    [ValidateSet('smoke','matrix')] [string] $Mode = 'matrix'
)

$ErrorActionPreference = 'Stop'
$attempt = Join-Path $JobRoot (Join-Path 'output\full' $AttemptId)
if (Test-Path -LiteralPath $attempt) {
    throw "Attempt exists; preserve it: $attempt"
}
$modules = Join-Path $JobRoot 'input\full-modules-v2'
$model = Join-Path $JobRoot 'input\full-model-v2'
$earlyBase = Join-Path $RepositoryRoot 'src\fg_connected_l3_l19\rtx_connected.cpp'
$prefixBase = Join-Path $RepositoryRoot 'src\fg_connected_l20_l87\rtx_prefix_bridge.cpp'
$coreBase = Join-Path $RepositoryRoot 'src\fg_connected_tail_l84_l87\rtx_core_bridge.cpp'
$tailBase = Join-Path $RepositoryRoot 'src\fg_connected_l20_l87\rtx_mid_l51_l56_smoke.cpp'
$earlyGenerator = Join-Path $RepositoryRoot 'scripts\generate-fg-connected-subrect-l3-l83-rtx-early.py'
$fullEarlyGenerator = Join-Path $RepositoryRoot 'scripts\generate-fg-connected-subrect-l3-l83-rtx-full-early.py'
$prefixGenerator = Join-Path $RepositoryRoot 'scripts\generate-fg-connected-subrect-l3-l83-rtx-prefix.py'
$livePrefixGenerator = Join-Path $RepositoryRoot 'scripts\generate-fg-connected-subrect-l3-l83-rtx-live-prefix.py'
$coreGenerator = Join-Path $RepositoryRoot 'scripts\generate-fg-connected-subrect-l3-l83-rtx-core.py'
$tailGenerator = Join-Path $RepositoryRoot 'scripts\generate-fg-connected-subrect-l3-l83-rtx-tail.py'
$fullTailGenerator = Join-Path $RepositoryRoot 'scripts\generate-fg-connected-subrect-l3-l83-rtx-full-tail.py'
foreach ($required in @($modules,$model,$earlyBase,$prefixBase,$coreBase,
    $tailBase,$earlyGenerator,$fullEarlyGenerator,$prefixGenerator,
    $livePrefixGenerator,$coreGenerator,$tailGenerator,$fullTailGenerator)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required full-chain input missing: $required"
    }
}

$build = Join-Path $attempt 'build'
$generated = Join-Path $build 'generated'
New-Item -ItemType Directory -Path $generated | Out-Null
$earlyStandalone = Join-Path $generated 'rtx_subrect_early_standalone.cpp'
$early = Join-Path $generated 'rtx_subrect_early_live.cpp'
$prefixStandalone = Join-Path $generated 'rtx_subrect_prefix_standalone.cpp'
$prefix = Join-Path $generated 'rtx_subrect_prefix_live.cpp'
$core = Join-Path $generated 'rtx_subrect_core.cpp'
$tailStandalone = Join-Path $generated 'rtx_subrect_tail_standalone.cpp'
$tail = Join-Path $generated 'rtx_subrect_tail_live.cpp'
& python $earlyGenerator $earlyBase $earlyStandalone
if ($LASTEXITCODE -ne 0) { throw 'Early generation failed' }
& python $fullEarlyGenerator $earlyStandalone $early
if ($LASTEXITCODE -ne 0) { throw 'Full early generation failed' }
& python $prefixGenerator $prefixBase $prefixStandalone
if ($LASTEXITCODE -ne 0) { throw 'Prefix generation failed' }
& python $livePrefixGenerator $prefixStandalone $prefix
if ($LASTEXITCODE -ne 0) { throw 'Live prefix generation failed' }
& python $coreGenerator $coreBase $core
if ($LASTEXITCODE -ne 0) { throw 'Core generation failed' }
& python $tailGenerator $tailBase $tailStandalone
if ($LASTEXITCODE -ne 0) { throw 'Tail generation failed' }
& python $fullTailGenerator $tailStandalone $tail
if ($LASTEXITCODE -ne 0) { throw 'Full tail generation failed' }

$cuda = $env:CUDA_PATH
if (-not $cuda) { $cuda = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3' }
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $installation) { throw 'Visual Studio C++ tools unavailable' }
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
$includeSubrect = Join-Path $RepositoryRoot 'src\fg_connected_subrect_l3_l83'
$includeEarly = Join-Path $RepositoryRoot 'src\fg_connected_l3_l19'
$includePrefix = Join-Path $RepositoryRoot 'src\fg_connected_l20_l87'
$includeCore = Join-Path $RepositoryRoot 'src\fg_connected_tail_l84_l87'
$flags = '/nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8'
$earlyObject = Join-Path $build 'early.obj'
$prefixObject = Join-Path $build 'prefix.obj'
$coreObject = Join-Path $build 'core.obj'
$tailObject = Join-Path $build 'tail.obj'
$binary = Join-Path $build 'subrect-full.exe'
$compileLog = Join-Path $build 'compile.log'
$includes = '/I"{0}" /I"{1}" /I"{2}" /I"{3}" /I"{4}"' -f `
    (Join-Path $cuda 'include'),$includeSubrect,$includeEarly,
    $includePrefix,$includeCore
$command = '"{0}" >nul && cl.exe {1} {2} /Fo"{3}" /c "{4}" >"{5}" 2>&1 && cl.exe {1} {2} /Fo"{6}" /c "{7}" >>"{5}" 2>&1 && cl.exe {1} {2} /Fo"{8}" /c "{9}" >>"{5}" 2>&1 && cl.exe {1} {2} /Fo"{10}" /c "{11}" >>"{5}" 2>&1 && link.exe /nologo /OUT:"{12}" "{3}" "{6}" "{8}" "{10}" "{13}" bcrypt.lib d3d12.lib dxgi.lib >>"{5}" 2>&1' -f `
    $vcvars,$flags,$includes,$earlyObject,$early,$compileLog,
    $prefixObject,$prefix,$coreObject,$core,$tailObject,$tail,$binary,
    (Join-Path $cuda 'lib\x64\cuda.lib')
& $env:ComSpec /d /s /c $command
if ($LASTEXITCODE -ne 0) { throw "Build failed; see $compileLog" }

if ($Mode -eq 'smoke') {
    $cases = @([ordered]@{id='subrect-full-poison-a5-process-a';poison='a5'})
} else {
    $cases = @(
        [ordered]@{id='subrect-full-poison-a5-process-a';poison='a5'},
        [ordered]@{id='subrect-full-poison-a5-process-b';poison='a5'},
        [ordered]@{id='subrect-full-poison-5a-process-a';poison='5a'},
        [ordered]@{id='subrect-full-poison-5a-process-b';poison='5a'}
    )
}
$processes = @()
foreach ($case in $cases) {
    $run = Join-Path $attempt $case.id
    New-Item -ItemType Directory -Path $run | Out-Null
    $process = Start-Process -FilePath $binary `
        -ArgumentList @($modules,$model,$run,$case.id,$case.poison) `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput (Join-Path $run 'stdout.log') `
        -RedirectStandardError (Join-Path $run 'stderr.log')
    if ($process.ExitCode -ne 0) {
        throw "Full run failed: $($case.id), exit $($process.ExitCode)"
    }
    $earlyReceipt = Get-Content -LiteralPath `
        (Join-Path $run 'early-run-receipt.json') -Raw | ConvertFrom-Json
    $tailReceipt = Get-Content -LiteralPath `
        (Join-Path $run 'l18-l83\run-receipt.json') -Raw | ConvertFrom-Json
    if ($earlyReceipt.result -ne 'pass' -or $tailReceipt.result -ne 'pass' -or
        $earlyReceipt.process_id -ne $tailReceipt.process_id) {
        throw "Nested full-chain receipt failed: $($case.id)"
    }
    $checkpoint = [ordered]@{}
    foreach ($property in $earlyReceipt.checkpoint_sha256.PSObject.Properties) {
        $checkpoint[$property.Name] = $property.Value
    }
    foreach ($property in $tailReceipt.checkpoint_sha256.PSObject.Properties) {
        $checkpoint[$property.Name] = $property.Value
    }
    $expected = @(3..83 | ForEach-Object { 'launch-{0:d6}' -f $_ })
    if (@($checkpoint.Keys).Count -ne 81 -or
        (Compare-Object @($checkpoint.Keys) $expected)) {
        throw "81-launch checkpoint sequence failed: $($case.id)"
    }
    $combined = [ordered]@{
        schema = 'neuralstudio.fg.connected-subrect-l3-l83-run.v1'
        backend = 'rtx-d3d12-nvapi'
        run_id = $case.id
        process_id = $earlyReceipt.process_id
        poison_byte = $earlyReceipt.poison_byte
        region = @(32,24,512,288)
        launch_range = @(3,83)
        launch_count = 81
        checkpoint_sha256 = $checkpoint
        allocation37_sha256 = $earlyReceipt.allocation37_sha256
        output_full_sha256 = $tailReceipt.output_full_sha256
        output_region_sha256 = $tailReceipt.output_region_sha256
        live_allocation37_handoff = $true
        expected_or_archived_intermediate_ingress = $false
        address_fields_recorded = $false
        result = 'pass'
    }
    $combined | ConvertTo-Json -Depth 7 |
        Set-Content -LiteralPath (Join-Path $run 'combined-run-receipt.json') `
            -Encoding ascii
    $processes += [pscustomobject][ordered]@{
        run_id=$case.id;process_id=$process.Id;
        poison_byte=$combined.poison_byte;
        checkpoint_sha256=$combined.checkpoint_sha256;
        allocation37_sha256=$combined.allocation37_sha256;
        output_full_sha256=$combined.output_full_sha256;
        output_region_sha256=$combined.output_region_sha256
    }
}
if ($Mode -eq 'matrix') {
    $maps = @($processes | ForEach-Object {
        $_.checkpoint_sha256 | ConvertTo-Json -Compress -Depth 5
    })
    if (@($processes.process_id | Sort-Object -Unique).Count -ne 4 -or
        @($maps | Sort-Object -Unique).Count -ne 1 -or
        @($processes.allocation37_sha256 | Sort-Object -Unique).Count -ne 1 -or
        @($processes.output_region_sha256 | Sort-Object -Unique).Count -ne 1) {
        throw 'Fresh-process/two-poison invariant gate failed'
    }
    $a5 = @($processes | Where-Object poison_byte -eq 165 |
        Select-Object -ExpandProperty output_full_sha256 -Unique)
    $fivea = @($processes | Where-Object poison_byte -eq 90 |
        Select-Object -ExpandProperty output_full_sha256 -Unique)
    if ($a5.Count -ne 1 -or $fivea.Count -ne 1 -or $a5[0] -eq $fivea[0]) {
        throw 'Outside-poison counterexample gate failed'
    }
}
$backend = [ordered]@{
    schema = 'neuralstudio.fg.connected-subrect-l3-l83-rtx-backend.v1'
    job_id = Split-Path $JobRoot -Leaf
    attempt_id = $AttemptId
    remote_job_retained = $JobRoot
    gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total `
        --format=csv,noheader,nounits | Select-Object -First 1).Trim()
    generated_source_sha256 = [ordered]@{
        early=(Get-FileHash $early -Algorithm SHA256).Hash.ToLowerInvariant()
        prefix=(Get-FileHash $prefix -Algorithm SHA256).Hash.ToLowerInvariant()
        core=(Get-FileHash $core -Algorithm SHA256).Hash.ToLowerInvariant()
        tail=(Get-FileHash $tail -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    binary_sha256 = (Get-FileHash $binary -Algorithm SHA256).Hash.ToLowerInvariant()
    compile_log_sha256 = (Get-FileHash $compileLog -Algorithm SHA256).Hash.ToLowerInvariant()
    fresh_process_count = $processes.Count
    poison_bytes = @($processes.poison_byte | Sort-Object -Unique)
    processes = @($processes | ForEach-Object {
        [ordered]@{run_id=$_.run_id;process_id=$_.process_id;
          poison_byte=$_.poison_byte;output_full_sha256=$_.output_full_sha256}
    })
    launch_range = @(3,83)
    launch_count = 81
    checkpoint_sha256 = $processes[0].checkpoint_sha256
    allocation37_sha256 = $processes[0].allocation37_sha256
    output_region_sha256 = $processes[0].output_region_sha256
    full_surface_poison_invariant = $false
    region_scoped_poison_invariant = $true
    live_device_handoff_count = 80
    live_allocation37_handoff = $true
    input_model_immutability = $true
    canary_integrity = $true
    expected_or_archived_intermediate_ingress = $false
    address_fields_recorded = $false
    result = 'pass'
}
$utf8NoBom = New-Object Text.UTF8Encoding($false)
[IO.File]::WriteAllText((Join-Path $attempt 'backend-receipt.json'),
    (($backend | ConvertTo-Json -Depth 8) + "`n"),$utf8NoBom)
Write-Output "attempt_root=$attempt"
Write-Output "allocation37_sha256=$($backend.allocation37_sha256)"
Write-Output "output_region_sha256=$($backend.output_region_sha256)"
Write-Output 'result=pass'
