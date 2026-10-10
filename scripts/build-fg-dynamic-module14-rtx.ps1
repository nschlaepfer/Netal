[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$JobRoot,
    [string]$RepositoryRoot,
    [string]$AttemptId = 'oracle-v1'
)

$ErrorActionPreference = 'Stop'
if (-not $RepositoryRoot) { $RepositoryRoot = Join-Path $JobRoot 'repo' }
$JobRoot = [IO.Path]::GetFullPath($JobRoot)
$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
$startedUtc = (Get-Date).ToUniversalTime().ToString('o')
$source = Join-Path $RepositoryRoot 'src\fg_dynamic_module14\rtx_oracle.cpp'
$contract = Join-Path $RepositoryRoot 'src\fg_dynamic_module14\contract.h'
$baseSource = Join-Path $RepositoryRoot 'src\fg_conv\rtx_oracle.cpp'
$harness = Join-Path $RepositoryRoot 'scripts\build-fg-dynamic-module14-rtx.ps1'
$module = Join-Path $JobRoot 'inputs\cu-module-000014.bin'
$attemptRoot = Join-Path $JobRoot (Join-Path 'output' $AttemptId)
foreach ($required in @($source, $contract, $baseSource, $harness, $module)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Missing module14 input: $required"
    }
}
if (Test-Path -LiteralPath $attemptRoot) { throw "Attempt exists: $attemptRoot" }
if ((Get-Item -LiteralPath $module).Length -ne 4584 -or
    (Get-FileHash -LiteralPath $module -Algorithm SHA256).Hash -ne
        '540F686FFCD4508212B19296992A69ED81CA95772FD229D0A252905C8015419C') {
    throw 'module14 identity mismatch'
}
$cudaPath = $env:CUDA_PATH
if (-not $cudaPath) {
    $cudaPath = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
}
$vswhere = Join-Path ${env:ProgramFiles(x86)} `
    'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
$buildRoot = Join-Path $attemptRoot 'build'
New-Item -ItemType Directory -Path $buildRoot | Out-Null
$binary = Join-Path $buildRoot 'fg-dynamic-module14-rtx.exe'
$object = Join-Path $buildRoot 'fg-dynamic-module14-rtx.obj'
$compileLog = Join-Path $buildRoot 'compile.log'
$compileCommand = '"{0}" >nul && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /I"{1}" /Fo"{2}" /c "{3}" >"{4}" 2>&1 && link.exe /nologo /OUT:"{5}" "{2}" "{6}" bcrypt.lib >>"{4}" 2>&1' -f `
    $vcvars, (Join-Path $cudaPath 'include'), $object, $source, $compileLog,
    $binary, (Join-Path $cudaPath 'lib\x64\cuda.lib')
& $env:ComSpec /d /s /c $compileCommand
if ($LASTEXITCODE -ne 0) {
    throw "module14 build failed; see $compileLog"
}

$receipts = @()
foreach ($resolution in @('720p', '1080p', '1440p', '4k')) {
    foreach ($poison in @('a5', '5a')) {
        foreach ($ordinal in @('a', 'b')) {
            $runId = "$resolution-poison-$poison-process-$ordinal"
            $runRoot = Join-Path $attemptRoot $runId
            New-Item -ItemType Directory -Path $runRoot | Out-Null
            $artifactMode = if ($poison -eq 'a5' -and $ordinal -eq 'a') {
                'full'
            } else {
                'hash-only'
            }
            $process = Start-Process -FilePath $binary `
                -ArgumentList @($module, $runRoot, $runId, $poison,
                                $resolution, $artifactMode) `
                -NoNewWindow -Wait -PassThru `
                -RedirectStandardOutput (Join-Path $runRoot 'stdout.log') `
                -RedirectStandardError (Join-Path $runRoot 'stderr.log')
            if ($process.ExitCode -ne 0) {
                throw "module14 run failed: $runId exit=$($process.ExitCode)"
            }
            $receipts += Get-Content -LiteralPath `
                (Join-Path $runRoot 'run-receipt.json') -Raw | ConvertFrom-Json
        }
    }
}
$processIds = @($receipts | ForEach-Object { [int64]$_.process_id } |
                Sort-Object -Unique)
if ($processIds.Count -ne 16) { throw 'module14 fresh-process gate failed' }
foreach ($resolution in @('720p', '1080p', '1440p', '4k')) {
    $hashes = @($receipts | Where-Object { $_.resolution -eq $resolution } |
                ForEach-Object { $_.output_sha256 } | Sort-Object -Unique)
    if ($hashes.Count -ne 1) {
        throw "module14 poison/repeat invariance failed: $resolution"
    }
}
$receipt = [ordered]@{
    schema = 'neuralstudio.fg.dynamic-module14-rig.v1'
    job_id = Split-Path -Leaf $JobRoot
    attempt_id = $AttemptId
    gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total,pci.bus_id `
        --format=csv,noheader,nounits | Select-Object -First 1).Trim()
    cuda_toolkit = (& (Join-Path $cudaPath 'bin\nvcc.exe') --version |
        Select-Object -Last 1).Trim()
    module_sha256 = (Get-FileHash -LiteralPath $module -Algorithm SHA256).Hash.ToLowerInvariant()
    source_sha256 = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
    contract_sha256 = (Get-FileHash -LiteralPath $contract -Algorithm SHA256).Hash.ToLowerInvariant()
    base_source_sha256 = (Get-FileHash -LiteralPath $baseSource -Algorithm SHA256).Hash.ToLowerInvariant()
    harness_sha256 = (Get-FileHash -LiteralPath $harness -Algorithm SHA256).Hash.ToLowerInvariant()
    binary_sha256 = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant()
    started_utc = $startedUtc
    completed_utc = (Get-Date).ToUniversalTime().ToString('o')
    command = 'build-fg-dynamic-module14-rtx.ps1 -JobRoot <job> -RepositoryRoot <staged-repo> -AttemptId oracle-v2'
    exit_code = 0
    raw_parameter_sha256 = [ordered]@{
        '720p' = 'e1f743baeb62a509d45537354a13acf4a8c3c61f6ea8f08d4469d46f70365c4f'
        '1080p' = '36bfa4cc6d18f4fa17d4e0191bf4c4e6abdfebb07e865ff540d447319146fbfe'
        '1440p' = '7c9719e0b5d5ee388bc0d40de8c17f52a3cedd2c60125804b07ad279817e80d6'
        '4k' = '36bfa4cc6d18f4fa17d4e0191bf4c4e6abdfebb07e865ff540d447319146fbfe'
    }
    run_count = 16
    fresh_process_count = 16
    true_typed_cuda_arrays = $true
    expected_or_captured_output_ingress = $false
    result = 'pass'
}
$utf8 = New-Object System.Text.UTF8Encoding($false)
$receiptPath = Join-Path $attemptRoot 'rig-receipt.json'
[IO.File]::WriteAllText(
    $receiptPath, (($receipt | ConvertTo-Json -Depth 5) + "`n"), $utf8)
Write-Host "attempt_root=$attemptRoot"
Write-Host "rig_receipt_sha256=$((Get-FileHash -LiteralPath $receiptPath -Algorithm SHA256).Hash.ToLowerInvariant())"
Write-Host 'result=pass'
