[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$JobRoot,
    [string]$AttemptId = 'semantics-v1'
)

$ErrorActionPreference = 'Stop'
$JobRoot = [IO.Path]::GetFullPath($JobRoot)
$source = Join-Path $JobRoot 'src\rtx_semantics_probe.cu'
$attemptRoot = Join-Path $JobRoot (Join-Path 'output' $AttemptId)
$buildRoot = Join-Path $attemptRoot 'build'
if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
    throw "Missing semantics probe: $source"
}
if (Test-Path -LiteralPath $attemptRoot) {
    throw "Attempt exists: $attemptRoot"
}
New-Item -ItemType Directory -Path $buildRoot | Out-Null

$cudaPath = $env:CUDA_PATH
if (-not $cudaPath) {
    $cudaPath = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
}
$nvcc = Join-Path $cudaPath 'bin\nvcc.exe'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
$binary = Join-Path $buildRoot 'fg-mid-l54-semantics-probe.exe'
$compileLog = Join-Path $buildRoot 'compile.log'
$compileCommand = '"{0}" >nul && "{1}" -std=c++20 -O2 -arch=sm_120 --expt-relaxed-constexpr -lineinfo -Xcompiler "/W4" -o "{2}" "{3}" >"{4}" 2>&1' -f `
    $vcvars, $nvcc, $binary, $source, $compileLog
& $env:ComSpec /d /s /c $compileCommand
if ($LASTEXITCODE -ne 0) {
    throw "Semantics probe build failed; see $compileLog"
}

$runs = @()
foreach ($ordinal in @('a', 'b')) {
    $runRoot = Join-Path $attemptRoot "process-$ordinal"
    $stdout = Join-Path $attemptRoot "process-$ordinal.stdout.log"
    $stderr = Join-Path $attemptRoot "process-$ordinal.stderr.log"
    $process = Start-Process -FilePath $binary -ArgumentList @($runRoot) `
        -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr
    if ($process.ExitCode -ne 0) {
        throw "Semantics probe process $ordinal failed: $($process.ExitCode)"
    }
    $runs += [ordered]@{
        ordinal = $ordinal
        process_id = $process.Id
        point_sha256 = (Get-FileHash -LiteralPath (Join-Path $runRoot 'point-selected-u32.bin') -Algorithm SHA256).Hash.ToLowerInvariant()
        pair_sha256 = (Get-FileHash -LiteralPath (Join-Path $runRoot 'pair-semantics.bin') -Algorithm SHA256).Hash.ToLowerInvariant()
        dense_ex2_sha256 = (Get-FileHash -LiteralPath (Join-Path $runRoot 'dense-ex2-minus2-to0-u32.bin') -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
foreach ($field in @('point_sha256', 'pair_sha256', 'dense_ex2_sha256')) {
    if (@($runs | ForEach-Object { $_[$field] } | Sort-Object -Unique).Count -ne 1) {
        throw "Fresh-process mismatch: $field"
    }
}
if (@($runs | ForEach-Object { $_.process_id } | Sort-Object -Unique).Count -ne 2) {
    throw 'Fresh-process identity gate failed'
}

$receipt = [ordered]@{
    schema = 'neuralstudio.fg.mid-l54-semantics-rig.v1'
    job_id = Split-Path -Leaf $JobRoot
    attempt_id = $AttemptId
    gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total,pci.bus_id --format=csv,noheader,nounits | Select-Object -First 1).Trim()
    cuda_toolkit = (& $nvcc --version | Select-Object -Last 1).Trim()
    source_sha256 = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
    binary_sha256 = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant()
    process_count = 2
    process_ids = @($runs | ForEach-Object { $_.process_id })
    outputs = [ordered]@{
        point_selected = [ordered]@{ records = 4665600; bytes = 18662400; sha256 = $runs[0].point_sha256 }
        pair_semantics = [ordered]@{ records = 289; bytes = 8092; sha256 = $runs[0].pair_sha256 }
        dense_ex2 = [ordered]@{ records = 4194305; bytes = 16777220; sha256 = $runs[0].dense_ex2_sha256 }
    }
    expected_frame_bytes_read = $false
    archived_intermediates_read = $false
    address_fields_recorded = $false
    repeat_equal = $true
    result = 'pass'
}
$utf8 = New-Object System.Text.UTF8Encoding($false)
$receiptPath = Join-Path $attemptRoot 'rig-receipt.json'
[IO.File]::WriteAllText($receiptPath, (($receipt | ConvertTo-Json -Depth 6) + "`n"), $utf8)
$receiptHash = (Get-FileHash -LiteralPath $receiptPath -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host "attempt_root=$attemptRoot"
Write-Host "rig_receipt_sha256=$receiptHash"
Write-Host 'result=pass'
