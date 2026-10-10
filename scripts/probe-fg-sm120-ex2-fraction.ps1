[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$JobRoot,
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [Parameter(Mandatory = $true)][string]$AttemptId
)

$ErrorActionPreference = 'Stop'
$JobRoot = [IO.Path]::GetFullPath($JobRoot).TrimEnd('\')
$prefix = [IO.Path]::GetFullPath('D:\CodexLab').TrimEnd('\') + '\'
if (-not $JobRoot.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'EX2 job must stay beneath D:\CodexLab'
}
$source = Join-Path $RepositoryRoot 'src\fg_math_ex2\rtx_fraction_probe.cu'
$attemptRoot = Join-Path $JobRoot "output\$AttemptId"
if (Test-Path -LiteralPath $attemptRoot) {
    throw "EX2 attempt already exists: $attemptRoot"
}
New-Item -ItemType Directory -Path (Join-Path $attemptRoot 'build') -Force |
    Out-Null

$cudaPath = if ($env:CUDA_PATH) {
    $env:CUDA_PATH
} else {
    'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
}
$nvcc = Join-Path $cudaPath 'bin\nvcc.exe'
$vswhere = Join-Path ${env:ProgramFiles(x86)} `
    'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
$binary = Join-Path $attemptRoot 'build\fg-sm120-ex2-fraction-probe.exe'
$compileLog = Join-Path $attemptRoot 'build\compile.log'
$command = '"{0}" >nul && "{1}" -std=c++20 -arch=sm_120 -O2 -Xcompiler="/W4 /WX /wd4211 /EHsc /MD /permissive-" "{2}" bcrypt.lib -o "{3}" >"{4}" 2>&1' -f `
    $vcvars, $nvcc, $source, $binary, $compileLog
& $env:ComSpec /d /s /c $command
if ($LASTEXITCODE -ne 0) {
    throw "EX2 fraction probe build failed; see $compileLog"
}

$runs = @()
foreach ($ordinal in @('a', 'b')) {
    $runId = "process-$ordinal"
    $runRoot = Join-Path $attemptRoot $runId
    $process = Start-Process -FilePath $binary `
        -ArgumentList @($runRoot, $runId) -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput (Join-Path $attemptRoot "$runId.stdout.log") `
        -RedirectStandardError (Join-Path $attemptRoot "$runId.stderr.log")
    if ($process.ExitCode -ne 0) {
        throw "EX2 fraction probe failed: $runId"
    }
    $runs += (Get-Content -LiteralPath `
        (Join-Path $runRoot 'run-receipt.json') -Raw | ConvertFrom-Json)
}
if (@($runs.process_id | Sort-Object -Unique).Count -ne 2) {
    throw 'EX2 fraction fresh-process gate failed'
}
foreach ($field in @('fraction_sha256', 'identity_sha256',
                     'identity_mismatches', 'first_identity_mismatch')) {
    if ($runs[0].$field -ne $runs[1].$field) {
        throw "EX2 fraction repeat mismatch: $field"
    }
}

$receipt = [ordered]@{
    schema = 'neuralstudio.fg.sm120-ex2-fraction-rig.v1'
    job_id = Split-Path -Leaf $JobRoot
    attempt_id = $AttemptId
    gpu = (& nvidia-smi `
        --query-gpu=name,driver_version,memory.total,compute_cap `
        --format=csv,noheader,nounits | Select-Object -First 1).Trim()
    cuda_toolkit = (& $nvcc --version | Select-Object -Last 1).Trim()
    source_sha256 = (Get-FileHash -LiteralPath $source `
        -Algorithm SHA256).Hash.ToLowerInvariant()
    binary_sha256 = (Get-FileHash -LiteralPath $binary `
        -Algorithm SHA256).Hash.ToLowerInvariant()
    process_ids = @($runs.process_id)
    fraction = [ordered]@{
        records = $runs[0].fraction_records
        bytes = 33554436
        sha256 = $runs[0].fraction_sha256
    }
    normalization_identity = [ordered]@{
        records = $runs[0].identity_records
        bytes = 16777216
        sha256 = $runs[0].identity_sha256
        mismatches = $runs[0].identity_mismatches
        first_mismatch = $runs[0].first_identity_mismatch
    }
    repeat_equal = $true
    expected_or_frame_bytes_consumed = $false
    address_fields_recorded = $false
    result = 'pass'
}
$utf8 = New-Object System.Text.UTF8Encoding($false)
$receiptPath = Join-Path $attemptRoot 'rig-receipt.json'
[IO.File]::WriteAllText(
    $receiptPath,
    (($receipt | ConvertTo-Json -Depth 8) + "`n"),
    $utf8)
Write-Host "attempt_root=$attemptRoot"
Write-Host "fraction_sha256=$($runs[0].fraction_sha256)"
Write-Host "identity_mismatches=$($runs[0].identity_mismatches)"
Write-Host 'result=pass'
