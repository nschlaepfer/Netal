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
    throw 'EX2 quantization job must stay beneath D:\CodexLab'
}
$source = Join-Path $RepositoryRoot `
    'src\fg_math_ex2\rtx_unit_quantization_probe.cu'
$attemptRoot = Join-Path $JobRoot "output\$AttemptId"
if (Test-Path -LiteralPath $attemptRoot) {
    throw "EX2 quantization attempt already exists: $attemptRoot"
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
$binary = Join-Path $attemptRoot 'build\fg-sm120-ex2-unit-quantization.exe'
$compileLog = Join-Path $attemptRoot 'build\compile.log'
$command = '"{0}" >nul && "{1}" -std=c++20 -arch=sm_120 -O2 -Xcompiler="/W4 /WX /wd4211 /EHsc /MD /permissive-" "{2}" -o "{3}" >"{4}" 2>&1' -f `
    $vcvars, $nvcc, $source, $binary, $compileLog
& $env:ComSpec /d /s /c $command
if ($LASTEXITCODE -ne 0) {
    throw "EX2 quantization probe build failed; see $compileLog"
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
        throw "EX2 quantization probe failed: $runId"
    }
    $summaryPath = Join-Path $runRoot 'summary.json'
    $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
    $runs += [ordered]@{
        summary = $summary
        summary_sha256 = (Get-FileHash -LiteralPath $summaryPath `
            -Algorithm SHA256).Hash.ToLowerInvariant()
        table_sha256 = (Get-FileHash -LiteralPath `
            (Join-Path $runRoot 'unit-minus1-to-zero-u32.bin') `
            -Algorithm SHA256).Hash.ToLowerInvariant()
        block_counts_sha256 = (Get-FileHash -LiteralPath `
            (Join-Path $runRoot 'quantization-block-counts.bin') `
            -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
if (@($runs.summary.process_id | Sort-Object -Unique).Count -ne 2) {
    throw 'EX2 quantization fresh-process gate failed'
}
foreach ($field in @('table_sha256', 'block_counts_sha256')) {
    if ($runs[0].$field -ne $runs[1].$field) {
        throw "EX2 quantization repeat mismatch: $field"
    }
}
foreach ($field in @('floor_mismatches', 'nearest_mismatches',
                     'ceil_mismatches')) {
    if ($runs[0].summary.$field -ne $runs[1].summary.$field) {
        throw "EX2 quantization summary mismatch: $field"
    }
}

$receipt = [ordered]@{
    schema = 'neuralstudio.fg.sm120-ex2-unit-quantization-rig.v1'
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
    process_ids = @($runs.summary.process_id)
    unit_table = [ordered]@{
        records = $runs[0].summary.unit_table_records
        bytes = 67108868
        sha256 = $runs[0].table_sha256
    }
    audit = [ordered]@{
        records = $runs[0].summary.audit_records
        first_exponent = $runs[0].summary.first_exponent
        last_exponent = $runs[0].summary.last_exponent
        floor_mismatches = $runs[0].summary.floor_mismatches
        nearest_mismatches = $runs[0].summary.nearest_mismatches
        ceil_mismatches = $runs[0].summary.ceil_mismatches
        block_counts_sha256 = $runs[0].block_counts_sha256
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
Write-Host "floor_mismatches=$($runs[0].summary.floor_mismatches)"
Write-Host "nearest_mismatches=$($runs[0].summary.nearest_mismatches)"
Write-Host "ceil_mismatches=$($runs[0].summary.ceil_mismatches)"
Write-Host 'result=pass'
