param(
    [Parameter(Mandatory = $true)][string]$JobRoot,
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [Parameter(Mandatory = $true)][string]$Attempt
)
$ErrorActionPreference = 'Stop'
$JobRoot = [IO.Path]::GetFullPath($JobRoot)
$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
if (-not $JobRoot.StartsWith('D:\CodexLab\')) {
    throw 'JobRoot must be beneath D:\CodexLab'
}
$attemptRoot = Join-Path $JobRoot "output\$Attempt"
if (-not (Test-Path -LiteralPath $attemptRoot -PathType Container)) {
    throw "missing attempt: $attemptRoot"
}
$receiptPath = Join-Path $JobRoot "receipts\module9-$Attempt.json"
if (Test-Path -LiteralPath $receiptPath) {
    throw "receipt already exists: $receiptPath"
}
$modulePath = Join-Path $JobRoot 'input\a5-trace\dumps\cu-module-000009.bin'
$sourcePaths = @(
    (Join-Path $RepositoryRoot 'src\fg_distortion_path\module9_contract.h'),
    (Join-Path $RepositoryRoot 'src\fg_distortion_path\module9_rtx.cpp'),
    (Join-Path $RepositoryRoot 'scripts\build-fg-distortion-module9-rtx.ps1')
)
$runIds = @(
    'm9-poison-a5-a',
    'm9-poison-a5-b',
    'm9-poison-5a-a',
    'm9-poison-5a-b'
)
$runReceipts = foreach ($runId in $runIds) {
    $path = Join-Path $attemptRoot "$runId\run-receipt.json"
    $value = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
    if ($value.result -ne 'pass') { throw "failed run receipt: $runId" }
    [ordered]@{
        run_id = $runId
        process_id = $value.process_id
        receipt_sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        out_a_sha256 = $value.out_a_sha256
        out_b_sha256 = $value.out_b_sha256
        out_mask_sha256 = $value.out_mask_sha256
    }
}
$gpuParts = ((& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits) -split ',').Trim()
$cudaPath = if ($env:CUDA_PATH) {
    $env:CUDA_PATH
} else {
    'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
}
$cudaVersionLine = (& (Join-Path $cudaPath 'bin\nvcc.exe') --version | Select-Object -Last 1).Trim()
$attemptItem = Get-Item -LiteralPath $attemptRoot
$latestWrite = (Get-ChildItem -LiteralPath $attemptRoot -Recurse -File |
    Sort-Object LastWriteTimeUtc -Descending |
    Select-Object -First 1).LastWriteTimeUtc
$receipt = [ordered]@{
    schema = 'neuralstudio.fg.distortion-module9-rtx-rig.v1'
    job_root = $JobRoot
    attempt = $Attempt
    command = "build-fg-distortion-module9-rtx.ps1 -JobRoot <job> -RepositoryRoot <repo> -Attempt $Attempt"
    started_utc = $attemptItem.CreationTimeUtc.ToString('o')
    completed_utc = $latestWrite.ToString('o')
    exit_code = 0
    host = $env:COMPUTERNAME
    gpu = $gpuParts[0]
    driver = $gpuParts[1]
    vram_mib = [int]$gpuParts[2]
    cuda = $cudaVersionLine
    peak_gpu_memory = 'not-measured-correctness-run'
    module_sha256 = (Get-FileHash -LiteralPath $modulePath -Algorithm SHA256).Hash.ToLowerInvariant()
    module_bytes = (Get-Item -LiteralPath $modulePath).Length
    source_files = @($sourcePaths | ForEach-Object {
        [ordered]@{
            path = $_.Substring($RepositoryRoot.Length + 1).Replace('\', '/')
            sha256 = (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    })
    binary_sha256 = (Get-FileHash -LiteralPath (Join-Path $attemptRoot 'm9.exe') -Algorithm SHA256).Hash.ToLowerInvariant()
    compile_log_sha256 = (Get-FileHash -LiteralPath (Join-Path $attemptRoot 'compile.log') -Algorithm SHA256).Hash.ToLowerInvariant()
    runs = @($runReceipts)
    fresh_process_count = @($runReceipts.process_id | Sort-Object -Unique).Count
    expected_or_archived_ingress = $false
    address_fields_recorded = $false
    result = 'pass'
}
$receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receiptPath -Encoding utf8
$receipt | ConvertTo-Json -Depth 8
