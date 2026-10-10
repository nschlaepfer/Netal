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
    $AttemptId = 'rtx-' + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
}
$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
$JobRoot = [IO.Path]::GetFullPath($JobRoot)
if (-not $JobRoot.StartsWith('D:\CodexLab\', [StringComparison]::OrdinalIgnoreCase)) {
    throw "JobRoot must be a scoped directory below D:\CodexLab: $JobRoot"
}

$source = Join-Path $RepositoryRoot 'src\fg_custom_upsample\rtx_oracle.cpp'
$verifier = Join-Path $RepositoryRoot 'scripts\verify-fg-custom-upsample.py'
$scriptPath = $MyInvocation.MyCommand.Path
$module = Join-Path $JobRoot 'inputs\selected-module.bin'
$attemptRoot = Join-Path $JobRoot (Join-Path 'outputs' $AttemptId)
$buildRoot = Join-Path $attemptRoot 'build'

foreach ($required in @($source, $verifier, $scriptPath, $module)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required custom-upsample input is missing: $required"
    }
}
if (Test-Path -LiteralPath $attemptRoot) {
    throw "Attempt directory already exists; preserve it and choose a new AttemptId: $attemptRoot"
}
$moduleItem = Get-Item -LiteralPath $module
$moduleHash = (Get-FileHash -LiteralPath $module -Algorithm SHA256).Hash.ToLowerInvariant()
if ($moduleItem.Length -ne 2112 -or $moduleHash -ne '1d19bfddd2affc5587ec19829259b070404e44c28a6f0c570a4b7769fd0dbba6') {
    throw "Selected module identity mismatch: bytes=$($moduleItem.Length) sha256=$moduleHash"
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
$installationVersion = & $vswhere -latest -products '*' `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationVersion
if (-not $installation) {
    throw 'Visual Studio C++ x64 build tools were not found.'
}
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path -LiteralPath $vcvars -PathType Leaf)) {
    throw "vcvars64.bat was not found: $vcvars"
}

New-Item -ItemType Directory -Path $buildRoot | Out-Null
$binary = Join-Path $buildRoot 'fg-custom-upsample-rtx.exe'
$object = Join-Path $buildRoot 'fg-custom-upsample-rtx.obj'
$compileLog = Join-Path $buildRoot 'compile.log'
$common = '/nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8'
$compileCommand = '"{0}" >nul && cl.exe {1} /I"{2}" /Fo"{3}" /c "{4}" >"{5}" 2>&1 && link.exe /nologo /OUT:"{6}" "{3}" "{7}" bcrypt.lib >>"{5}" 2>&1' -f `
    $vcvars, $common, $cudaInclude, $object, $source, $compileLog, $binary, $cudaLibrary
& $env:ComSpec /d /s /c $compileCommand
if ($LASTEXITCODE -ne 0) {
    throw "Custom-upsample CUDA Driver build failed with exit code $LASTEXITCODE; see $compileLog"
}

$cases = @(
    [ordered]@{ id = 'poison-a5-process-a'; poison = 'a5' },
    [ordered]@{ id = 'poison-a5-process-b'; poison = 'a5' },
    [ordered]@{ id = 'poison-5a-process-a'; poison = '5a' },
    [ordered]@{ id = 'poison-5a-process-b'; poison = '5a' }
)
$runReceipts = @()
$processExitCodes = @()
foreach ($case in $cases) {
    $runRoot = Join-Path $attemptRoot $case.id
    New-Item -ItemType Directory -Path $runRoot | Out-Null
    $stdout = Join-Path $runRoot 'stdout.log'
    $stderr = Join-Path $runRoot 'stderr.log'
    $arguments = @($module, $runRoot, $case.id, $case.poison)
    $process = Start-Process -FilePath $binary `
        -ArgumentList $arguments `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $processExitCodes += $process.ExitCode
    if ($process.ExitCode -ne 0) {
        throw "Custom-upsample RTX run $($case.id) failed with exit code $($process.ExitCode); see $stderr"
    }
    $receiptPath = Join-Path $runRoot 'run-receipt.json'
    if (-not (Test-Path -LiteralPath $receiptPath -PathType Leaf)) {
        throw "Custom-upsample RTX run did not produce a receipt: $receiptPath"
    }
    $receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
    if ($receipt.result -ne 'pass') {
        throw "Custom-upsample RTX run receipt did not pass: $($case.id)"
    }
    $runReceipts += $receipt
}
$processIds = @($runReceipts | ForEach-Object { [int64]$_.process_id } | Sort-Object -Unique)
if ($processIds.Count -ne 4) {
    throw "Fresh-process gate failed: unique process count=$($processIds.Count)"
}

$backendReceipt = Join-Path $attemptRoot 'backend-receipt.json'
& python $verifier backend --backend rtx --root $attemptRoot --output $backendReceipt
if ($LASTEXITCODE -ne 0) {
    throw "Custom-upsample RTX backend verification failed with exit code $LASTEXITCODE"
}

$gpuRow = (& nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader,nounits | Select-Object -First 1).Trim()
$gpuFields = @($gpuRow -split ',' | ForEach-Object { $_.Trim() })
if ($gpuFields.Count -ne 4) {
    throw "Unexpected nvidia-smi hardware row: $gpuRow"
}
$cudaVersion = (& (Join-Path $cudaPath 'bin\nvcc.exe') --version | Select-Object -Last 1).Trim()
$drive = Get-PSDrive -Name D
$sourceArtifacts = @($source, $verifier, $scriptPath) | ForEach-Object {
    [ordered]@{
        name = [IO.Path]::GetFileName($_)
        bytes = (Get-Item -LiteralPath $_).Length
        sha256 = (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$draft = [ordered]@{
    schema = 'neuralstudio.fg.custom-upsample-rig.v1'
    job_id = Split-Path -Leaf $JobRoot
    attempt_id = $AttemptId
    started_utc = $started
    ended_utc = (Get-Date).ToUniversalTime().ToString('o')
    hardware = [ordered]@{
        host = $env:COMPUTERNAME
        gpu = $gpuFields[0]
        driver = $gpuFields[1]
        vram_mib = [int]$gpuFields[2]
        compute_capability = $gpuFields[3]
        cuda_toolkit = $cudaVersion
        visual_studio = $installationVersion
        d_free_gb = [math]::Round($drive.Free / 1GB, 1)
    }
    selected_module = [ordered]@{
        token = 60
        bytes = 2112
        sha256 = $moduleHash
        transport = 'cuModuleLoadDataEx-exact-selected-module'
    }
    source_artifacts = $sourceArtifacts
    binary = [ordered]@{
        name = [IO.Path]::GetFileName($binary)
        bytes = (Get-Item -LiteralPath $binary).Length
        sha256 = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    compile_exit_code = 0
    process_exit_codes = @($processExitCodes)
    fresh_process_count = $processIds.Count
    command_exit_code = 0
    external_expected_tensors_consumed = $false
    address_fields_recorded = $false
    result = 'pass'
}
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$draftPath = Join-Path $attemptRoot 'rig-receipt-draft.json'
[IO.File]::WriteAllText(
    $draftPath,
    (($draft | ConvertTo-Json -Depth 10) + "`n"),
    $utf8NoBom
)
$rigReceipt = Join-Path $attemptRoot 'rig-receipt.json'
& python $verifier root-rig --input $draftPath --output $rigReceipt
if ($LASTEXITCODE -ne 0) {
    throw "Custom-upsample rig receipt rooting failed with exit code $LASTEXITCODE"
}

Write-Host "attempt_root=$attemptRoot"
Write-Host "backend_receipt_sha256=$((Get-FileHash -LiteralPath $backendReceipt -Algorithm SHA256).Hash.ToLowerInvariant())"
Write-Host "rig_receipt_sha256=$((Get-FileHash -LiteralPath $rigReceipt -Algorithm SHA256).Hash.ToLowerInvariant())"
Write-Host 'result=pass'
