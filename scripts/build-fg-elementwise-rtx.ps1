[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$JobRoot,
    [string]$AttemptId,
    [string]$RepositoryRoot
)

$ErrorActionPreference = 'Stop'
$expectedModuleHash = '1970B33C56B1580790A7D56377E4BFD5C1ED8522CD3D5E28B07192B481067D26'
$expectedParameterHash = '0BB4C1C5E582EA1A7386318D294DF1511FDC2E81287B7943F23F9A531A1690BC'
$expectedModuleBytes = 2152
$expectedParameterBytes = 52
$started = (Get-Date).ToUniversalTime().ToString('o')
if (-not $RepositoryRoot) {
    $RepositoryRoot = Split-Path -Parent $PSScriptRoot
}
if (-not $AttemptId) {
    $AttemptId = 'attempt-' + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
}
$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
$JobRoot = [IO.Path]::GetFullPath($JobRoot)
$source = Join-Path $RepositoryRoot 'src\fg_elementwise\rtx_oracle.cpp'
$verifier = Join-Path $RepositoryRoot 'scripts\verify-fg-elementwise.py'
$buildScript = Join-Path $RepositoryRoot 'scripts\build-fg-elementwise-rtx.ps1'
$moduleA = Join-Path $JobRoot 'inputs\run-a\cu-module-000037.bin'
$moduleB = Join-Path $JobRoot 'inputs\run-b\cu-module-000037.bin'
$parametersA = Join-Path $JobRoot 'inputs\run-a\launch-000033-kernel-0000.params.bin'
$parametersB = Join-Path $JobRoot 'inputs\run-b\launch-000033-kernel-0000.params.bin'
$attemptRoot = Join-Path $JobRoot (Join-Path 'output\fg-elementwise-add-v1' $AttemptId)
$buildRoot = Join-Path $attemptRoot 'build'

foreach ($required in @($source, $verifier, $buildScript, $moduleA, $moduleB, $parametersA, $parametersB)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required FG elementwise input is missing: $required"
    }
}
if (Test-Path -LiteralPath $attemptRoot) {
    throw "Attempt directory already exists; preserve it and choose a new AttemptId: $attemptRoot"
}
foreach ($module in @($moduleA, $moduleB)) {
    $item = Get-Item -LiteralPath $module
    $hash = (Get-FileHash -LiteralPath $module -Algorithm SHA256).Hash
    if ($item.Length -ne $expectedModuleBytes -or $hash -ne $expectedModuleHash) {
        throw "Selected elementwise module identity mismatch: bytes=$($item.Length) sha256=$hash"
    }
}
foreach ($parameters in @($parametersA, $parametersB)) {
    $item = Get-Item -LiteralPath $parameters
    $hash = (Get-FileHash -LiteralPath $parameters -Algorithm SHA256).Hash
    if ($item.Length -ne $expectedParameterBytes -or $hash -ne $expectedParameterHash) {
        throw "Selected elementwise parameter identity mismatch: bytes=$($item.Length) sha256=$hash"
    }
}

$cudaPath = $env:CUDA_PATH
if (-not $cudaPath) {
    $cudaPath = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
}
$cudaInclude = Join-Path $cudaPath 'include'
$cudaLibrary = Join-Path $cudaPath 'lib\x64\cuda.lib'
foreach ($required in @((Join-Path $cudaInclude 'cuda.h'), $cudaLibrary)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "CUDA Driver development input is missing: $required"
    }
}
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
    throw 'Visual Studio Installer vswhere.exe was not found.'
}
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
$binary = Join-Path $buildRoot 'fg-elementwise-rtx.exe'
$object = Join-Path $buildRoot 'fg-elementwise-rtx.obj'
$compileLog = Join-Path $buildRoot 'compile.log'
$common = '/nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8'
$compileCommand = '"{0}" >nul && cl.exe {1} /I"{2}" /Fo"{3}" /c "{4}" >"{5}" 2>&1 && link.exe /nologo /OUT:"{6}" "{3}" "{7}" bcrypt.lib >>"{5}" 2>&1' -f `
    $vcvars, $common, $cudaInclude, $object, $source, $compileLog, $binary, $cudaLibrary
& $env:ComSpec /d /s /c $compileCommand
if ($LASTEXITCODE -ne 0) {
    throw "FG elementwise CUDA Driver build failed with exit code $LASTEXITCODE; see $compileLog"
}

$cases = @(
    [ordered]@{ id = 'poison-a5-process-a'; poison = 'a5' },
    [ordered]@{ id = 'poison-a5-process-b'; poison = 'a5' },
    [ordered]@{ id = 'poison-5a-process-a'; poison = '5a' },
    [ordered]@{ id = 'poison-5a-process-b'; poison = '5a' }
)
$runReceipts = @()
foreach ($case in $cases) {
    $runRoot = Join-Path $attemptRoot $case.id
    New-Item -ItemType Directory -Path $runRoot | Out-Null
    $stdout = Join-Path $runRoot 'stdout.log'
    $stderr = Join-Path $runRoot 'stderr.log'
    $process = Start-Process -FilePath $binary `
        -ArgumentList @($moduleA, $runRoot, $case.id, $case.poison) `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    if ($process.ExitCode -ne 0) {
        throw "FG elementwise RTX run $($case.id) failed with exit code $($process.ExitCode); see $stderr"
    }
    $receiptPath = Join-Path $runRoot 'run-receipt.json'
    if (-not (Test-Path -LiteralPath $receiptPath -PathType Leaf)) {
        throw "FG elementwise RTX run did not produce a receipt: $receiptPath"
    }
    $receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
    if ($receipt.result -ne 'pass') {
        throw "FG elementwise RTX run receipt did not pass: $($case.id)"
    }
    $runReceipts += $receipt
}
$processIds = @($runReceipts | ForEach-Object { [int64]$_.process_id } | Sort-Object -Unique)
if ($processIds.Count -ne 4) {
    throw "Fresh-process gate failed: unique process count=$($processIds.Count)"
}

$gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader,nounits | Select-Object -First 1).Trim()
$cudaVersion = (& (Join-Path $cudaPath 'bin\nvcc.exe') --version | Select-Object -Last 1).Trim()
$sourceArtifacts = @(
    [ordered]@{ path = $source; relative_path = 'src/fg_elementwise/rtx_oracle.cpp' },
    [ordered]@{ path = $verifier; relative_path = 'scripts/verify-fg-elementwise.py' },
    [ordered]@{ path = $buildScript; relative_path = 'scripts/build-fg-elementwise-rtx.ps1' }
) | ForEach-Object {
    [ordered]@{
        relative_path = $_.relative_path
        bytes = (Get-Item -LiteralPath $_.path).Length
        sha256 = (Get-FileHash -LiteralPath $_.path -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$rigReceipt = [ordered]@{
    schema = 'neuralstudio.fg.elementwise-add-rig.v1'
    job_id = Split-Path -Leaf $JobRoot
    attempt_id = $AttemptId
    started_utc = $started
    ended_utc = (Get-Date).ToUniversalTime().ToString('o')
    gpu = $gpu
    cuda_toolkit = $cudaVersion
    visual_studio = $installation
    source_capture_job = 'fg2x-intake-20260904T082417Z'
    selected_module = [ordered]@{
        job_relative_paths = @(
            'inputs/run-a/cu-module-000037.bin',
            'inputs/run-b/cu-module-000037.bin'
        )
        token = 37
        bytes = $expectedModuleBytes
        sha256 = $expectedModuleHash.ToLowerInvariant()
        two_source_processes_equal = $true
        embedded_code = 'sm_89-ptx-no-elf'
        transport = 'cuModuleLoadDataEx-exact-selected-module'
    }
    live_parameter_capture = [ordered]@{
        job_relative_paths = @(
            'inputs/run-a/launch-000033-kernel-0000.params.bin',
            'inputs/run-b/launch-000033-kernel-0000.params.bin'
        )
        bytes = $expectedParameterBytes
        captured_raw_sha256 = $expectedParameterHash.ToLowerInvariant()
        two_source_processes_equal = $true
        address_free_normalized_sha256 = '60fc0e7d1b56104174f99a50457306f3308cd9c099873ba6e6083503f88b9d80'
        scalar_bytes_sha256 = '0f2b3cefe270ff83b105bc583212a894fdaf8671acab257be993fcedab912822'
        pointer_roles = @('input_a', 'input_b', 'output')
        scalars_u32 = @(10, 6, 256, 10, 6, 10, 6)
    }
    launch = [ordered]@{
        token = 33
        evaluate_1_phase_order = 22
        global_order = 31
        function = 'k_element_wise'
        grid = @(3, 2, 1)
        block = @(4, 4, 8)
        shape_nhwc = @(1, 6, 10, 256)
    }
    source_artifacts = $sourceArtifacts
    binary = [ordered]@{
        relative_path = 'build/fg-elementwise-rtx.exe'
        bytes = (Get-Item -LiteralPath $binary).Length
        sha256 = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    run_ids = @($cases | ForEach-Object { $_.id })
    fresh_process_count = $processIds.Count
    address_fields_recorded = $false
    result = 'pass'
}
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$rigReceiptPath = Join-Path $attemptRoot 'rig-receipt.json'
[IO.File]::WriteAllText(
    $rigReceiptPath,
    (($rigReceipt | ConvertTo-Json -Depth 10) + "`n"),
    $utf8NoBom
)
$rigReceiptHash = (Get-FileHash -LiteralPath $rigReceiptPath -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText(
    (Join-Path $attemptRoot 'rig-receipt.sha256'),
    ($rigReceiptHash + "  rig-receipt.json`n"),
    $utf8NoBom
)

$backendReceipt = Join-Path $attemptRoot 'backend-receipt.json'
& python $verifier backend --backend rtx --root $attemptRoot --output $backendReceipt
if ($LASTEXITCODE -ne 0) {
    throw "FG elementwise RTX backend verification failed with exit code $LASTEXITCODE"
}
$backendHash = (Get-FileHash -LiteralPath $backendReceipt -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host "attempt_root=$attemptRoot"
Write-Host "rig_receipt_sha256=$rigReceiptHash"
Write-Host "backend_receipt_sha256=$backendHash"
Write-Host 'result=pass'
