[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $JobRoot,
    [Parameter(Mandatory = $true)] [string] $RepositoryRoot,
    [Parameter(Mandatory = $true)] [string] $AttemptId
)

$ErrorActionPreference = 'Stop'
$attempt = Join-Path $JobRoot (Join-Path 'output\module19' $AttemptId)
if (Test-Path -LiteralPath $attempt) {
    throw "Attempt exists; preserve it: $attempt"
}
$source = Join-Path $RepositoryRoot `
    'src\fg_connected_subrect_l3_l83\rtx_module19_mode1.cpp'
$module = Join-Path $JobRoot 'input\modules\cu-module-000019.bin'
foreach ($required in @($source, $module)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required input missing: $required"
    }
}
$moduleItem = Get-Item -LiteralPath $module
$moduleHash = (Get-FileHash -LiteralPath $module -Algorithm SHA256).Hash
if ($moduleItem.Length -ne 57792 -or
    $moduleHash -ne
        '143C6712C99DB4A5792DF697D68178AFEEA3657F4F0EEB12F905C90ABF38A1B0') {
    throw 'Selected module19 identity mismatch'
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
if (-not $installation) { throw 'Visual Studio C++ tools unavailable' }
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
$build = Join-Path $attempt 'build'
New-Item -ItemType Directory -Path $build | Out-Null
$object = Join-Path $build 'module19-mode1.obj'
$binary = Join-Path $build 'module19-mode1.exe'
$compileLog = Join-Path $build 'compile.log'
$command = '"{0}" >nul && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /I"{1}" /Fo"{2}" /c "{3}" >"{4}" 2>&1 && link.exe /nologo /OUT:"{5}" "{2}" "{6}" bcrypt.lib d3d12.lib dxgi.lib >>"{4}" 2>&1' -f `
    $vcvars, (Join-Path $cuda 'include'), $object, $source, $compileLog,
    $binary, (Join-Path $cuda 'lib\x64\cuda.lib')
& $env:ComSpec /d /s /c $command
if ($LASTEXITCODE -ne 0) { throw "Build failed; see $compileLog" }

$cases = @()
foreach ($fixture in @('mode1-a', 'mode1-b')) {
    foreach ($poison in @('a5', '5a')) {
        foreach ($process in @('a', 'b')) {
            $cases += [ordered]@{
                id = "$fixture-poison-$poison-process-$process"
                fixture = $fixture
                poison = $poison
            }
        }
    }
}
$processes = @()
foreach ($case in $cases) {
    $run = Join-Path $attempt $case.id
    New-Item -ItemType Directory -Path $run | Out-Null
    $process = Start-Process -FilePath $binary `
        -ArgumentList @($module, $run, $case.id, $case.fixture,
                        $case.poison, '1') `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput (Join-Path $run 'stdout.log') `
        -RedirectStandardError (Join-Path $run 'stderr.log')
    if ($process.ExitCode -ne 0) {
        throw "Run failed: $($case.id), exit $($process.ExitCode)"
    }
    $receipt = Get-Content -LiteralPath (Join-Path $run 'run-receipt.json') `
        -Raw | ConvertFrom-Json
    if ($receipt.result -ne 'pass') {
        throw "Receipt failed: $($case.id)"
    }
    $processes += [ordered]@{
        run_id = $case.id
        process_id = $process.Id
        exit_code = $process.ExitCode
        fixture = $case.fixture
        poison_byte = $receipt.poison_byte
        output_full_sha256 = $receipt.output_full_sha256
        output_region_sha256 = $receipt.output_region_sha256
    }
}
if (@($processes.process_id | Sort-Object -Unique).Count -ne 8) {
    throw 'Fresh process identity gate failed'
}
foreach ($fixture in @('mode1-a', 'mode1-b')) {
    $rows = @($processes | Where-Object fixture -eq $fixture)
    if (@($rows.output_region_sha256 | Sort-Object -Unique).Count -ne 1) {
        throw "Region invariance failed: $fixture"
    }
    foreach ($poison in @(90, 165)) {
        $repeat = @($rows | Where-Object poison_byte -eq $poison)
        if ($repeat.Count -ne 2 -or
            @($repeat.output_full_sha256 | Sort-Object -Unique).Count -ne 1) {
            throw "Process repeat failed: $fixture/$poison"
        }
    }
}

$negative = Join-Path $attempt 'reject-mode0'
New-Item -ItemType Directory -Path $negative | Out-Null
$negativeProcess = Start-Process -FilePath $binary `
    -ArgumentList @($module, $negative, 'reject-mode0', 'mode1-a', 'a5', '0') `
    -NoNewWindow -Wait -PassThru `
    -RedirectStandardOutput (Join-Path $negative 'stdout.log') `
    -RedirectStandardError (Join-Path $negative 'stderr.log')
if ($negativeProcess.ExitCode -ne 4 -or
    (Test-Path -LiteralPath (Join-Path $negative 'run-receipt.json'))) {
    throw 'Mode rejection gate failed'
}

$backend = [ordered]@{
    schema = 'neuralstudio.fg.subrect-module19-mode1-rtx-backend.v1'
    job_id = Split-Path $JobRoot -Leaf
    attempt_id = $AttemptId
    remote_job_retained = $JobRoot
    gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total `
        --format=csv,noheader,nounits | Select-Object -First 1).Trim()
    cuda_toolkit = (& (Join-Path $cuda 'bin\nvcc.exe') --version |
        Select-Object -Last 1).Trim()
    source_sha256 =
        (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
    module_sha256 = $moduleHash.ToLowerInvariant()
    binary_sha256 =
        (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant()
    compile_log_sha256 =
        (Get-FileHash -LiteralPath $compileLog -Algorithm SHA256).Hash.ToLowerInvariant()
    processes = $processes
    fresh_process_count = 8
    fixture_count = 2
    poison_bytes = @(90, 165)
    two_process_repeat_per_fixture_poison = $true
    two_poison_region_invariance = $true
    region_extent = @(32, 24, 512, 288)
    output_region_bytes = 589824
    mode_0_rejected = $true
    typed_d3d12_resources = $true
    input_immutability = $true
    canary_integrity = $true
    complete_declared_write_coverage = $true
    expected_or_captured_output_ingress = $false
    archived_intermediate_ingress = $false
    address_fields_recorded = $false
    result = 'pass'
}
$utf8NoBom = New-Object Text.UTF8Encoding($false)
[IO.File]::WriteAllText(
    (Join-Path $attempt 'backend-receipt.json'),
    (($backend | ConvertTo-Json -Depth 8) + "`n"), $utf8NoBom)
Write-Output "attempt_root=$attempt"
Write-Output 'result=pass'
