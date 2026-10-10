[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$JobRoot,
    [string]$AttemptId,
    [string]$RepositoryRoot
)

$ErrorActionPreference = 'Stop'
$started = (Get-Date).ToUniversalTime().ToString('o')
if (-not $RepositoryRoot) {
    $RepositoryRoot = Split-Path -Parent $PSScriptRoot
}
if (-not $AttemptId) {
    $AttemptId = 'attempt-' + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
}
$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
$JobRoot = [IO.Path]::GetFullPath($JobRoot)
$source = Join-Path $RepositoryRoot 'src\fg_elementwise\rtx_family_oracle.cpp'
$verifier = Join-Path $RepositoryRoot 'scripts\verify-fg-elementwise-family.py'
$buildScript = Join-Path $RepositoryRoot 'scripts\build-fg-elementwise-family-rtx.ps1'
$inputRoot = Join-Path $JobRoot 'inputs'
$moduleDirectory = Join-Path $inputRoot 'run-a'
$attemptRoot = Join-Path $JobRoot (Join-Path 'output\fg-elementwise-family-v2' $AttemptId)
$buildRoot = Join-Path $attemptRoot 'build'

$modules = @(
    [ordered]@{ token = 37; bytes = 2152; sha256 = '1970B33C56B1580790A7D56377E4BFD5C1ED8522CD3D5E28B07192B481067D26' },
    [ordered]@{ token = 41; bytes = 2152; sha256 = 'A717783407A1CEA5E3D64B4B23EDB451F2B791D8CB7980B374433B9F6DED17E1' },
    [ordered]@{ token = 46; bytes = 2152; sha256 = 'C733968C9CDCA9BA170A9A453D2A831537F467DE10E089325AABF8E8ED1CD58F' },
    [ordered]@{ token = 50; bytes = 2144; sha256 = 'DC36C503F9FCA1CB42944CAB551C268B2D738912927F4AAB9DB4D53AE66766B6' }
)
$launches = @(
    [ordered]@{ token = 33; module = 37; raw = '0BB4C1C5E582EA1A7386318D294DF1511FDC2E81287B7943F23F9A531A1690BC'; normalized = '60fc0e7d1b56104174f99a50457306f3308cd9c099873ba6e6083503f88b9d80'; scalars = @(10, 6, 256, 10, 6, 10, 6) },
    [ordered]@{ token = 37; module = 41; raw = '6746A095B6748B7257237E90FD20E1A0D8B15A21813FD90F19C3BCD87B3D4296'; normalized = 'a143c92c761c72c3b2e384fd13bd94d0b4d6311c3c8ba38de0d45206558f80d2'; scalars = @(20, 12, 128, 20, 12, 20, 12) },
    [ordered]@{ token = 41; module = 41; raw = 'CE20833A64A17BC2A9E38AAC82FF4A36753BC837B7268F0B5E4368AAB0E7035B'; normalized = 'f018d7df61d7d25e61b0648680adfcbfce6ef93d7aec05f8a7172f92b450d8cf'; scalars = @(40, 24, 128, 40, 24, 40, 24) },
    [ordered]@{ token = 45; module = 46; raw = '7C0BCDB4678FA19591FB171585F20C226E23C0097B498CAED9E099DE67400A45'; normalized = '4e4460b122f091bf427f41d1ffa6822eca5d7cc5344a4c750ed3c57e5a8616b2'; scalars = @(80, 48, 64, 80, 48, 80, 48) },
    [ordered]@{ token = 49; module = 50; raw = 'A680B8D6C4AA3EA28E7962297986D8DC0020CC70679DCDD3B8E2CF1F402CE82D'; normalized = 'bebf0d883fac729020854143da492c3d22894b49451d9973b27757d000d96bb1'; scalars = @(160, 96, 32, 160, 96, 160, 96) }
)

foreach ($required in @($source, $verifier, $buildScript)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required elementwise-family source is missing: $required"
    }
}
if (Test-Path -LiteralPath $attemptRoot) {
    throw "Attempt directory already exists; preserve it and choose a new AttemptId: $attemptRoot"
}
foreach ($sourceRun in @('run-a', 'run-b')) {
    foreach ($module in $modules) {
        $path = Join-Path $inputRoot (Join-Path $sourceRun ('cu-module-{0:D6}.bin' -f $module.token))
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Selected module is missing: $path"
        }
        $item = Get-Item -LiteralPath $path
        $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
        if ($item.Length -ne $module.bytes -or $hash -ne $module.sha256) {
            throw "Selected module identity mismatch: $path bytes=$($item.Length) sha256=$hash"
        }
    }
    foreach ($launch in $launches) {
        $path = Join-Path $inputRoot (Join-Path $sourceRun ('launch-{0:D6}-kernel-0000.params.bin' -f $launch.token))
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Selected launch parameter block is missing: $path"
        }
        $item = Get-Item -LiteralPath $path
        $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
        if ($item.Length -ne 52 -or $hash -ne $launch.raw) {
            throw "Selected parameter identity mismatch: $path bytes=$($item.Length) sha256=$hash"
        }
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
$binary = Join-Path $buildRoot 'fg-elementwise-family-rtx.exe'
$object = Join-Path $buildRoot 'fg-elementwise-family-rtx.obj'
$compileLog = Join-Path $buildRoot 'compile.log'
$common = '/nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8'
$compileCommand = '"{0}" >nul && cl.exe {1} /I"{2}" /Fo"{3}" /c "{4}" >"{5}" 2>&1 && link.exe /nologo /NODEFAULTLIB:LIBCMT /OUT:"{6}" "{3}" "{7}" bcrypt.lib >>"{5}" 2>&1' -f `
    $vcvars, $common, $cudaInclude, $object, $source, $compileLog, $binary, $cudaLibrary
& $env:ComSpec /d /s /c $compileCommand
if ($LASTEXITCODE -ne 0) {
    throw "FG elementwise-family build failed with exit code $LASTEXITCODE; see $compileLog"
}

$processCases = @(
    [ordered]@{ id = 'poison-a5-process-a'; poison = 'a5' },
    [ordered]@{ id = 'poison-a5-process-b'; poison = 'a5' },
    [ordered]@{ id = 'poison-5a-process-a'; poison = '5a' },
    [ordered]@{ id = 'poison-5a-process-b'; poison = '5a' }
)
$runReceipts = @()
foreach ($case in $processCases) {
    $runRoot = Join-Path $attemptRoot $case.id
    New-Item -ItemType Directory -Path $runRoot | Out-Null
    $process = Start-Process -FilePath $binary `
        -ArgumentList @($moduleDirectory, $runRoot, $case.id, $case.poison) `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput (Join-Path $runRoot 'stdout.log') `
        -RedirectStandardError (Join-Path $runRoot 'stderr.log')
    if ($process.ExitCode -ne 0) {
        throw "FG elementwise-family run $($case.id) failed with exit code $($process.ExitCode)"
    }
    $receiptPath = Join-Path $runRoot 'run-receipt.json'
    if (-not (Test-Path -LiteralPath $receiptPath -PathType Leaf)) {
        throw "FG elementwise-family run receipt is missing: $receiptPath"
    }
    $receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
    if ($receipt.result -ne 'pass') {
        throw "FG elementwise-family run receipt did not pass: $($case.id)"
    }
    $runReceipts += $receipt
}
$processIds = @($runReceipts | ForEach-Object { [int64]$_.process_id } | Sort-Object -Unique)
if ($processIds.Count -ne 4) {
    throw "Fresh-process gate failed: unique process count=$($processIds.Count)"
}

$sourceArtifacts = @(
    [ordered]@{ path = $source; relative_path = 'src/fg_elementwise/rtx_family_oracle.cpp' },
    [ordered]@{ path = $verifier; relative_path = 'scripts/verify-fg-elementwise-family.py' },
    [ordered]@{ path = $buildScript; relative_path = 'scripts/build-fg-elementwise-family-rtx.ps1' }
) | ForEach-Object {
    [ordered]@{
        relative_path = $_.relative_path
        bytes = (Get-Item -LiteralPath $_.path).Length
        sha256 = (Get-FileHash -LiteralPath $_.path -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$moduleContracts = @($modules | ForEach-Object {
    [ordered]@{
        token = $_.token
        bytes = $_.bytes
        sha256 = $_.sha256.ToLowerInvariant()
        source_paths = @(
            ('inputs/run-a/cu-module-{0:D6}.bin' -f $_.token),
            ('inputs/run-b/cu-module-{0:D6}.bin' -f $_.token)
        )
        two_source_processes_equal = $true
    }
})
$launchContracts = @($launches | ForEach-Object {
    [ordered]@{
        token = $_.token
        module_token = $_.module
        captured_raw_sha256 = $_.raw.ToLowerInvariant()
        address_free_normalized_sha256 = $_.normalized
        pointer_roles = @('input_a', 'input_b', 'output')
        scalars_u32 = $_.scalars
        two_source_processes_equal = $true
    }
})
$gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader,nounits | Select-Object -First 1).Trim()
$cudaVersion = (& (Join-Path $cudaPath 'bin\nvcc.exe') --version | Select-Object -Last 1).Trim()
$rigReceipt = [ordered]@{
    schema = 'neuralstudio.fg.elementwise-family-rig.v2'
    job_id = Split-Path -Leaf $JobRoot
    attempt_id = $AttemptId
    started_utc = $started
    ended_utc = (Get-Date).ToUniversalTime().ToString('o')
    gpu = $gpu
    cuda_toolkit = $cudaVersion
    visual_studio = $installation
    source_capture_job = 'fg2x-intake-20260904T082417Z'
    selected_modules = $moduleContracts
    selected_launch_parameters = $launchContracts
    source_artifacts = $sourceArtifacts
    binary = [ordered]@{
        relative_path = 'build/fg-elementwise-family-rtx.exe'
        bytes = (Get-Item -LiteralPath $binary).Length
        sha256 = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    run_ids = @($processCases | ForEach-Object { $_.id })
    fresh_process_count = $processIds.Count
    address_fields_recorded = $false
    result = 'pass'
}
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$rigReceiptPath = Join-Path $attemptRoot 'rig-receipt.json'
[IO.File]::WriteAllText(
    $rigReceiptPath,
    (($rigReceipt | ConvertTo-Json -Depth 12) + "`n"),
    $utf8NoBom
)
$rigHash = (Get-FileHash -LiteralPath $rigReceiptPath -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText(
    (Join-Path $attemptRoot 'rig-receipt.sha256'),
    ($rigHash + "  rig-receipt.json`n"),
    $utf8NoBom
)

$backendReceipt = Join-Path $attemptRoot 'backend-receipt.json'
& python $verifier backend --backend rtx --root $attemptRoot --output $backendReceipt
if ($LASTEXITCODE -ne 0) {
    throw "FG elementwise-family backend verification failed with exit code $LASTEXITCODE"
}
$backendHash = (Get-FileHash -LiteralPath $backendReceipt -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host "attempt_root=$attemptRoot"
Write-Host "rig_receipt_sha256=$rigHash"
Write-Host "backend_receipt_sha256=$backendHash"
Write-Host 'result=pass'
