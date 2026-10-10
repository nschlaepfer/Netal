[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $JobRoot,
    [Parameter(Mandatory = $true)] [string] $RepositoryRoot,
    [Parameter(Mandatory = $true)] [string] $AttemptId,
    [ValidateSet('smoke','matrix')] [string] $Mode = 'matrix'
)

$ErrorActionPreference = 'Stop'
$attempt = Join-Path $JobRoot (Join-Path 'output\early' $AttemptId)
if (Test-Path -LiteralPath $attempt) {
    throw "Attempt exists; preserve it: $attempt"
}
$modules = Join-Path $JobRoot 'input\full-modules-v2'
$model = Join-Path $JobRoot 'input\full-model-v2'
$base = Join-Path $RepositoryRoot 'src\fg_connected_l3_l19\rtx_connected.cpp'
$generator = Join-Path $RepositoryRoot `
    'scripts\generate-fg-connected-subrect-l3-l83-rtx-early.py'
foreach ($required in @($modules, $model, $base, $generator)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required early input missing: $required"
    }
}

$build = Join-Path $attempt 'build'
New-Item -ItemType Directory -Path $build | Out-Null
$source = Join-Path $build 'rtx_subrect_early.cpp'
& python $generator $base $source
if ($LASTEXITCODE -ne 0) { throw 'Early source generation failed' }

$vswhere = Join-Path ${env:ProgramFiles(x86)} `
    'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $installation) { throw 'Visual Studio C++ tools unavailable' }
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
$object = Join-Path $build 'rtx_subrect_early.obj'
$binary = Join-Path $build 'rtx_subrect_early.exe'
$compileLog = Join-Path $build 'compile.log'
$includeSubrect = Join-Path $RepositoryRoot 'src\fg_connected_subrect_l3_l83'
$includeBase = Join-Path $RepositoryRoot 'src\fg_connected_l3_l19'
$flags = '/nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8'
$command = '"{0}" >nul && cl.exe {1} /I"{2}" /I"{3}" /Fo"{4}" /c "{5}" >"{6}" 2>&1 && link.exe /nologo /OUT:"{7}" "{4}" bcrypt.lib d3d12.lib dxgi.lib >>"{6}" 2>&1' -f `
    $vcvars, $flags, $includeSubrect, $includeBase, $object, $source,
    $compileLog, $binary
& $env:ComSpec /d /s /c $command
if ($LASTEXITCODE -ne 0) { throw "Build failed; see $compileLog" }

if ($Mode -eq 'smoke') {
    $cases = @([ordered]@{id='subrect-early-poison-a5-process-a';poison='a5'})
} else {
    $cases = @(
        [ordered]@{id='subrect-early-poison-a5-process-a';poison='a5'},
        [ordered]@{id='subrect-early-poison-a5-process-b';poison='a5'},
        [ordered]@{id='subrect-early-poison-5a-process-a';poison='5a'},
        [ordered]@{id='subrect-early-poison-5a-process-b';poison='5a'}
    )
}
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
        throw "Early run failed: $($case.id), exit $($process.ExitCode)"
    }
    $receipt = Get-Content -LiteralPath `
        (Join-Path $run 'early-run-receipt.json') -Raw | ConvertFrom-Json
    if ($receipt.result -ne 'pass') { throw "Early receipt failed: $($case.id)" }
    $processes += [ordered]@{
        run_id = $case.id
        process_id = $process.Id
        poison_byte = $receipt.poison_byte
        allocation37_sha256 = $receipt.allocation37_sha256
        checkpoint_sha256 = $receipt.checkpoint_sha256
    }
}
if ($Mode -eq 'matrix') {
    $checkpointMaps = @($processes | ForEach-Object {
        $_.checkpoint_sha256 | ConvertTo-Json -Compress -Depth 4
    })
    if (@($processes.process_id | Sort-Object -Unique).Count -ne 4 -or
        @($checkpointMaps | Sort-Object -Unique).Count -ne 1 -or
        @($processes.allocation37_sha256 | Sort-Object -Unique).Count -ne 1) {
        throw 'Fresh-process/two-poison invariance failed'
    }
}
$backend = [ordered]@{
    schema = 'neuralstudio.fg.connected-subrect-l3-l17-rtx-backend.v1'
    job_id = Split-Path $JobRoot -Leaf
    attempt_id = $AttemptId
    remote_job_retained = $JobRoot
    gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total `
        --format=csv,noheader,nounits | Select-Object -First 1).Trim()
    source_sha256 = (Get-FileHash $source -Algorithm SHA256).Hash.ToLowerInvariant()
    binary_sha256 = (Get-FileHash $binary -Algorithm SHA256).Hash.ToLowerInvariant()
    compile_log_sha256 = `
        (Get-FileHash $compileLog -Algorithm SHA256).Hash.ToLowerInvariant()
    fresh_process_count = $processes.Count
    poison_bytes = @($processes.poison_byte | Sort-Object -Unique)
    processes = @($processes | ForEach-Object {
        [ordered]@{run_id=$_.run_id;process_id=$_.process_id;
          poison_byte=$_.poison_byte;
          allocation37_sha256=$_.allocation37_sha256}
    })
    launch_range = @(3,17)
    launch_count = 15
    checkpoint_sha256 = $processes[0].checkpoint_sha256
    allocation37_sha256 = $processes[0].allocation37_sha256
    origin = @(32,24)
    active_extent = @(512,288)
    module23_mode = 1
    omitted_launch_families = @('module22','module24','module27')
    expected_or_archived_intermediate_ingress = $false
    address_fields_recorded = $false
    result = 'pass'
}
$utf8NoBom = New-Object Text.UTF8Encoding($false)
[IO.File]::WriteAllText((Join-Path $attempt 'backend-receipt.json'),
    (($backend | ConvertTo-Json -Depth 7) + "`n"), $utf8NoBom)
Write-Output "attempt_root=$attempt"
Write-Output "allocation37_sha256=$($backend.allocation37_sha256)"
Write-Output 'result=pass'
