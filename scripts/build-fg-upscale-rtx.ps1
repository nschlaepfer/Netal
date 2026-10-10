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
if (-not $JobRoot.StartsWith('D:\CodexLab\', [StringComparison]::OrdinalIgnoreCase)) {
    throw "JobRoot must be a scoped directory below D:\CodexLab: $JobRoot"
}

$source = Join-Path $RepositoryRoot 'src\fg_upscale\rtx_oracle.cpp'
$verifier = Join-Path $RepositoryRoot 'scripts\verify-fg-upscale.py'
$moduleSpecs = @(
    [ordered]@{ token = 36; name = 'cu-module-000036.bin'; bytes = 2296; sha256 = 'D6962E058C46F6DE9C6C5F8C75DAFDE64A34F094A49F1719AF16C505F3B530FD' },
    [ordered]@{ token = 40; name = 'cu-module-000040.bin'; bytes = 2296; sha256 = '070C7F4987251C28BCAD097AC8A00E7B1E2644DAE4267AE37B676E280478A2B8' },
    [ordered]@{ token = 45; name = 'cu-module-000045.bin'; bytes = 2296; sha256 = '547A05AB275EDCB0D413E4B7F89BADA85F08035FA0517CD5267960FE695498FC' },
    [ordered]@{ token = 49; name = 'cu-module-000049.bin'; bytes = 2296; sha256 = '2B91BF740602F074ACBAA5EFBA1CD1115C5EE52653A96D087C0D52231DE2A74A' }
)
$modulePaths = @($moduleSpecs | ForEach-Object { Join-Path $JobRoot (Join-Path 'input' $_.name) })
$attemptRoot = Join-Path $JobRoot (Join-Path 'output\fg-upscale-v1' $AttemptId)
$buildRoot = Join-Path $attemptRoot 'build'

foreach ($required in @($source, $verifier) + $modulePaths) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required FG upscale input is missing: $required"
    }
}
if (Test-Path -LiteralPath $attemptRoot) {
    throw "Attempt directory already exists; preserve it and choose a new AttemptId: $attemptRoot"
}
for ($index = 0; $index -lt $moduleSpecs.Count; ++$index) {
    $item = Get-Item -LiteralPath $modulePaths[$index]
    $hash = (Get-FileHash -LiteralPath $modulePaths[$index] -Algorithm SHA256).Hash
    if ($item.Length -ne $moduleSpecs[$index].bytes -or $hash -ne $moduleSpecs[$index].sha256) {
        throw "Selected module identity mismatch for token $($moduleSpecs[$index].token): bytes=$($item.Length) sha256=$hash"
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
$binary = Join-Path $buildRoot 'fg-upscale-rtx.exe'
$object = Join-Path $buildRoot 'fg-upscale-rtx.obj'
$compileLog = Join-Path $buildRoot 'compile.log'
$common = '/nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8'
$compileCommand = '"{0}" >nul && cl.exe {1} /I"{2}" /Fo"{3}" /c "{4}" >"{5}" 2>&1 && link.exe /nologo /OUT:"{6}" "{3}" "{7}" bcrypt.lib >>"{5}" 2>&1' -f `
    $vcvars, $common, $cudaInclude, $object, $source, $compileLog, $binary, $cudaLibrary
& $env:ComSpec /d /s /c $compileCommand
if ($LASTEXITCODE -ne 0) {
    throw "FG upscale CUDA Driver build failed with exit code $LASTEXITCODE; see $compileLog"
}

$cases = @(
    [ordered]@{ id = 'poison-a5-process-a'; poison = 'a5' },
    [ordered]@{ id = 'poison-a5-process-b'; poison = 'a5' },
    [ordered]@{ id = 'poison-5a-process-a'; poison = '5a' },
    [ordered]@{ id = 'poison-5a-process-b'; poison = '5a' }
)
$runReceipts = @()
$processResults = @()
foreach ($case in $cases) {
    $runRoot = Join-Path $attemptRoot $case.id
    New-Item -ItemType Directory -Path $runRoot | Out-Null
    $stdout = Join-Path $runRoot 'stdout.log'
    $stderr = Join-Path $runRoot 'stderr.log'
    $arguments = @($modulePaths) + @($runRoot, $case.id, $case.poison)
    $process = Start-Process -FilePath $binary `
        -ArgumentList $arguments `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $processResults += [ordered]@{
        run_id = $case.id
        exit_code = $process.ExitCode
        stdout_relative_path = "$($case.id)/stdout.log"
        stderr_relative_path = "$($case.id)/stderr.log"
    }
    if ($process.ExitCode -ne 0) {
        throw "FG upscale RTX run $($case.id) failed with exit code $($process.ExitCode); see $stderr"
    }
    $receiptPath = Join-Path $runRoot 'run-receipt.json'
    if (-not (Test-Path -LiteralPath $receiptPath -PathType Leaf)) {
        throw "FG upscale RTX run did not produce a receipt: $receiptPath"
    }
    $receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
    if ($receipt.result -ne 'pass') {
        throw "FG upscale RTX run receipt did not pass: $($case.id)"
    }
    $runReceipts += $receipt
}
$processIds = @($runReceipts | ForEach-Object { [int64]$_.process_id } | Sort-Object -Unique)
if ($processIds.Count -ne 4) {
    throw "Fresh-process gate failed: unique process count=$($processIds.Count)"
}

$gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader,nounits | Select-Object -First 1).Trim()
$cudaVersion = (& (Join-Path $cudaPath 'bin\nvcc.exe') --version | Select-Object -Last 1).Trim()
$sourceArtifacts = @($source, $verifier, $MyInvocation.MyCommand.Path) | ForEach-Object {
    [ordered]@{
        relative_path = [IO.Path]::GetFileName($_)
        bytes = (Get-Item -LiteralPath $_).Length
        sha256 = (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$selectedModules = @()
for ($index = 0; $index -lt $moduleSpecs.Count; ++$index) {
    $selectedModules += [ordered]@{
        job_relative_path = 'input/' + $moduleSpecs[$index].name
        token = $moduleSpecs[$index].token
        bytes = $moduleSpecs[$index].bytes
        sha256 = $moduleSpecs[$index].sha256.ToLowerInvariant()
        embedded_code = 'sm_89-ptx-no-elf'
        transport = 'cuModuleLoadDataEx-exact-selected-module'
    }
}
$rigReceipt = [ordered]@{
    schema = 'neuralstudio.fg.upscale-rig.v1'
    job_id = Split-Path -Leaf $JobRoot
    attempt_id = $AttemptId
    started_utc = $started
    ended_utc = (Get-Date).ToUniversalTime().ToString('o')
    gpu = $gpu
    cuda_toolkit = $cudaVersion
    visual_studio = $installation
    invocation = [ordered]@{
        script_relative_path = 'scripts/build-fg-upscale-rtx.ps1'
        job_root_leaf = Split-Path -Leaf $JobRoot
        repository_root_leaf = Split-Path -Leaf $RepositoryRoot
        attempt_id = $AttemptId
    }
    compile_exit_code = 0
    process_results = $processResults
    command_exit_code = 0
    selected_modules = $selectedModules
    selected_launch_tokens = @(32, 36, 40, 44, 48)
    source_artifacts = $sourceArtifacts
    binary = [ordered]@{
        relative_path = 'build/fg-upscale-rtx.exe'
        bytes = (Get-Item -LiteralPath $binary).Length
        sha256 = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    run_ids = @($cases | ForEach-Object { $_.id })
    fresh_process_count = $processIds.Count
    external_expected_tensors_consumed = $false
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
    throw "FG upscale RTX backend verification failed with exit code $LASTEXITCODE"
}
$backendHash = (Get-FileHash -LiteralPath $backendReceipt -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText(
    (Join-Path $attemptRoot 'backend-receipt.sha256'),
    ($backendHash + "  backend-receipt.json`n"),
    $utf8NoBom
)
Write-Host "attempt_root=$attemptRoot"
Write-Host "rig_receipt_sha256=$rigReceiptHash"
Write-Host "backend_receipt_sha256=$backendHash"
Write-Host 'result=pass'
