param(
    [Parameter(Mandatory = $true)] [string] $JobRoot,
    [Parameter(Mandatory = $true)] [string] $SourceRoot,
    [Parameter(Mandatory = $true)] [string] $AttemptId
)

$ErrorActionPreference = 'Stop'
$started = (Get-Date).ToUniversalTime().ToString('o')
$modulePath = Join-Path $JobRoot 'input\cu-module-000020.bin'
$sourcePath = Join-Path $SourceRoot 'rtx_nvapi_oracle.cpp'
$attemptRoot = Join-Path $JobRoot (Join-Path 'output\fg-compositor-l87-v1' $AttemptId)
$buildRoot = Join-Path $attemptRoot 'build'
foreach ($required in @($modulePath, $sourcePath, (Join-Path $SourceRoot 'compositor_l87_contract.h'))) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required compositor input is missing: $required"
    }
}
if (Test-Path -LiteralPath $attemptRoot) {
    throw "Attempt already exists; preserve it and choose a new AttemptId: $attemptRoot"
}
$module = Get-Item -LiteralPath $modulePath
$moduleHash = (Get-FileHash -LiteralPath $modulePath -Algorithm SHA256).Hash
if ($module.Length -ne 26024 -or $moduleHash -ne '6CC983A76DECAED1A8C7F207A95531BECE5A1AD092C4DB3A0159B590C331C24B') {
    throw "Selected compositor module identity mismatch: bytes=$($module.Length) sha256=$moduleHash"
}

$cudaPath = $env:CUDA_PATH
if (-not $cudaPath) { $cudaPath = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3' }
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $installation) { throw 'Visual Studio C++ x64 build tools were not found.' }
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
New-Item -ItemType Directory -Path $buildRoot | Out-Null
$binary = Join-Path $buildRoot 'fg-compositor-l87-rtx.exe'
$object = Join-Path $buildRoot 'fg-compositor-l87-rtx.obj'
$compileLog = Join-Path $buildRoot 'compile.log'
$compileCommand = '"{0}" >nul && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /I"{1}" /Fo"{2}" /c "{3}" >"{4}" 2>&1 && link.exe /nologo /OUT:"{5}" "{2}" bcrypt.lib d3d12.lib dxgi.lib >>"{4}" 2>&1' -f `
    $vcvars, $SourceRoot, $object, $sourcePath, $compileLog, $binary
& $env:ComSpec /d /s /c $compileCommand
if ($LASTEXITCODE -ne 0) { throw "Compositor build failed; see $compileLog" }

$cases = @(
    [ordered]@{ id = 'poison-a5-process-a'; poison = 'a5'; mode = 'full' },
    [ordered]@{ id = 'poison-a5-process-b'; poison = 'a5'; mode = 'full' },
    [ordered]@{ id = 'poison-5a-process-a'; poison = '5a'; mode = 'full' },
    [ordered]@{ id = 'poison-5a-process-b'; poison = '5a'; mode = 'full' },
    [ordered]@{ id = 'sparse-mask-branch'; poison = '3c'; mode = 'sparse' }
)
$processes = @()
$receipts = @()
foreach ($case in $cases) {
    $runRoot = Join-Path $attemptRoot $case.id
    New-Item -ItemType Directory -Path $runRoot | Out-Null
    $stdout = Join-Path $runRoot 'stdout.log'
    $stderr = Join-Path $runRoot 'stderr.log'
    $process = Start-Process -FilePath $binary `
        -ArgumentList @($modulePath, $runRoot, $case.id, $case.poison, $case.mode) `
        -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    if ($process.ExitCode -ne 0) {
        throw "Compositor run $($case.id) failed with exit code $($process.ExitCode); see $stderr"
    }
    $receipt = Get-Content -LiteralPath (Join-Path $runRoot 'run-receipt.json') -Raw | ConvertFrom-Json
    if ($receipt.result -ne 'pass') { throw "Compositor run receipt failed: $($case.id)" }
    $receipts += $receipt
    $processes += [ordered]@{ run_id = $case.id; process_id = $process.Id; exit_code = $process.ExitCode }
}
$primary = @($receipts | Where-Object { $_.mask_mode -eq 'full' })
$primaryPids = @($primary | ForEach-Object { [int64]$_.process_id } | Sort-Object -Unique)
$primaryOutputs = @($primary | ForEach-Object { $_.output_sha256 } | Sort-Object -Unique)
if ($primaryPids.Count -ne 4) { throw "Fresh-process gate failed: $($primaryPids.Count)" }
if ($primaryOutputs.Count -ne 1) { throw "Two-poison output invariance failed" }
if (@($primary | Where-Object { [int64]$_.poison_pixels_after -ne 0 }).Count -ne 0) {
    throw 'Primary write-coverage gate failed'
}

$gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader,nounits | Select-Object -First 1).Trim()
$sourceHashes = Get-FileHash -Algorithm SHA256 -LiteralPath $sourcePath,(Join-Path $SourceRoot 'compositor_l87_contract.h'),$MyInvocation.MyCommand.Path |
    ForEach-Object { [ordered]@{ file = Split-Path $_.Path -Leaf; sha256 = $_.Hash.ToLowerInvariant() } }
$receipt = [ordered]@{
    schema = 'neuralstudio.fg.compositor-l87-rig.v1'
    job_id = Split-Path $JobRoot -Leaf
    remote_job_retained = $JobRoot
    attempt_id = $AttemptId
    started_utc = $started
    finished_utc = (Get-Date).ToUniversalTime().ToString('o')
    gpu = $gpu
    cuda_toolkit = (& (Join-Path $cudaPath 'bin\nvcc.exe') --version | Select-Object -Last 1).Trim()
    module = [ordered]@{ bytes = $module.Length; sha256 = $moduleHash.ToLowerInvariant() }
    sources = @($sourceHashes)
    processes = @($processes)
    fresh_primary_processes = $primaryPids.Count
    primary_output_sha256 = $primaryOutputs[0]
    primary_output_bytes = 921600
    two_poison_output_invariance = $true
    primary_zero_untouched_pixels = $true
    sparse_mask_branch_gated = $true
    result = 'pass'
}
$receiptPath = Join-Path $attemptRoot 'rig-receipt.json'
$receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receiptPath -Encoding ascii
(Get-FileHash -LiteralPath $receiptPath -Algorithm SHA256).Hash.ToLowerInvariant() |
    Set-Content -LiteralPath (Join-Path $attemptRoot 'rig-receipt.sha256') -Encoding ascii
Write-Output "attempt_root=$attemptRoot"
Write-Output "output_sha256=$($primaryOutputs[0])"
Write-Output 'result=pass'
