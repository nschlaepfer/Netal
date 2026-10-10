param(
    [Parameter(Mandatory = $true)] [string] $JobRoot,
    [Parameter(Mandatory = $true)] [string] $SourceRoot,
    [Parameter(Mandatory = $true)] [string] $AttemptId
)

$ErrorActionPreference = 'Stop'
$started = (Get-Date).ToUniversalTime().ToString('o')
$modulePath = Join-Path $JobRoot 'input\cu-module-000003.bin'
$sourcePath = Join-Path $SourceRoot 'rtx_oracle.cpp'
$contractPath = Join-Path $SourceRoot 'fg_mid_l55_contract.h'
$inputsPath = Join-Path $SourceRoot 'fg_mid_l55_inputs.h'
$attemptRoot = Join-Path $JobRoot (Join-Path 'output\fg-mid-l55-v1' $AttemptId)
$buildRoot = Join-Path $attemptRoot 'build'
foreach ($required in @($modulePath, $sourcePath, $contractPath, $inputsPath)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required L55 input is missing: $required"
    }
}
if (Test-Path -LiteralPath $attemptRoot) {
    throw "Attempt already exists; preserve it and choose a new AttemptId: $attemptRoot"
}
$module = Get-Item -LiteralPath $modulePath
$moduleHash = (Get-FileHash -LiteralPath $modulePath -Algorithm SHA256).Hash
if ($module.Length -ne 43880 -or $moduleHash -ne '66A9E6BDA87EC8B63C5E10752E84D07435DC4392BBE42EC142E78B0D0B624E37') {
    throw "Selected L55 module identity mismatch: bytes=$($module.Length) sha256=$moduleHash"
}
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $installation) { throw 'Visual Studio C++ x64 build tools were not found.' }
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
New-Item -ItemType Directory -Path $buildRoot | Out-Null
$binary = Join-Path $buildRoot 'fg-mid-l55-rtx.exe'
$object = Join-Path $buildRoot 'fg-mid-l55-rtx.obj'
$compileLog = Join-Path $buildRoot 'compile.log'
$compileCommand = '"{0}" >nul && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /I"{1}" /Fo"{2}" /c "{3}" >"{4}" 2>&1 && link.exe /nologo /OUT:"{5}" "{2}" bcrypt.lib d3d12.lib dxgi.lib >>"{4}" 2>&1' -f `
    $vcvars, $SourceRoot, $object, $sourcePath, $compileLog, $binary
& $env:ComSpec /d /s /c $compileCommand
if ($LASTEXITCODE -ne 0) { throw "L55 build failed; see $compileLog" }

$cases = @(
    [ordered]@{ id = 'poison-a5-process-a'; poison = 'a5' },
    [ordered]@{ id = 'poison-a5-process-b'; poison = 'a5' },
    [ordered]@{ id = 'poison-5a-process-a'; poison = '5a' },
    [ordered]@{ id = 'poison-5a-process-b'; poison = '5a' }
)
$processes = @()
$receipts = @()
foreach ($case in $cases) {
    $runRoot = Join-Path $attemptRoot $case.id
    New-Item -ItemType Directory -Path $runRoot | Out-Null
    $stdout = Join-Path $runRoot 'stdout.log'
    $stderr = Join-Path $runRoot 'stderr.log'
    $process = Start-Process -FilePath $binary `
        -ArgumentList @($modulePath, $runRoot, $case.id, $case.poison) `
        -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    if ($process.ExitCode -ne 0) {
        throw "L55 run $($case.id) failed with exit code $($process.ExitCode); see $stderr"
    }
    $receipt = Get-Content -LiteralPath (Join-Path $runRoot 'run-receipt.json') -Raw | ConvertFrom-Json
    if ($receipt.result -ne 'pass') { throw "L55 run receipt failed: $($case.id)" }
    $receipts += $receipt
    $processes += [ordered]@{ run_id = $case.id; process_id = $process.Id; exit_code = $process.ExitCode }
}
$pids = @($receipts | ForEach-Object { [int64]$_.process_id } | Sort-Object -Unique)
$outputs = @($receipts | ForEach-Object { $_.combined_output_sha256 } | Sort-Object -Unique)
if ($pids.Count -ne 4) { throw "Fresh-process gate failed: $($pids.Count)" }
if ($outputs.Count -ne 1) { throw 'Two-poison output invariance failed' }
$cudaPath = $env:CUDA_PATH
if (-not $cudaPath) { $cudaPath = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3' }
$gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader,nounits | Select-Object -First 1).Trim()
$sourceHashes = Get-FileHash -Algorithm SHA256 -LiteralPath $sourcePath,$contractPath,$inputsPath,$MyInvocation.MyCommand.Path |
    ForEach-Object { [ordered]@{ file = Split-Path $_.Path -Leaf; sha256 = $_.Hash.ToLowerInvariant() } }
$receipt = [ordered]@{
    schema = 'neuralstudio.fg.mid-l55-rig.v1'
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
    fresh_process_count = $pids.Count
    poison_bytes = @(90, 165)
    combined_output_sha256 = $outputs[0]
    two_poison_output_invariance = $true
    full_write_coverage_from_two_poison_invariance = $true
    result = 'pass'
}
$receiptPath = Join-Path $attemptRoot 'rig-receipt.json'
$receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receiptPath -Encoding ascii
(Get-FileHash -LiteralPath $receiptPath -Algorithm SHA256).Hash.ToLowerInvariant() |
    Set-Content -LiteralPath (Join-Path $attemptRoot 'rig-receipt.sha256') -Encoding ascii
Write-Output "attempt_root=$attemptRoot"
Write-Output "combined_output_sha256=$($outputs[0])"
Write-Output 'result=pass'
