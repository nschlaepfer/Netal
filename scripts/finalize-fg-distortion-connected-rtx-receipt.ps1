param(
    [Parameter(Mandatory = $true)][string]$JobRoot,
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [Parameter(Mandatory = $true)][string]$Module10Attempt,
    [Parameter(Mandatory = $true)][string]$ConnectedAttempt
)
$ErrorActionPreference = 'Stop'
$JobRoot = [IO.Path]::GetFullPath($JobRoot)
$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
if (-not $JobRoot.StartsWith('D:\CodexLab\')) { throw 'job must be under D:\CodexLab' }
$receiptPath = Join-Path $JobRoot 'receipts\distortion-connected-final.json'
if (Test-Path -LiteralPath $receiptPath) { throw "receipt exists: $receiptPath" }
$sourcePaths = @(
    'src\fg_distortion_path\module10_contract.h',
    'src\fg_distortion_path\module10_rtx.cpp',
    'src\fg_distortion_path\connected_contract.h',
    'src\fg_distortion_path\connected_rtx.cpp',
    'src\fg_distortion_path\module9_contract.h',
    'scripts\build-fg-distortion-module10-rtx.ps1',
    'scripts\build-fg-distortion-connected-rtx.ps1'
)
$attempts = foreach ($attemptName in @($Module10Attempt, $ConnectedAttempt)) {
    $attemptRoot = Join-Path $JobRoot "output\$attemptName"
    if (-not (Test-Path -LiteralPath $attemptRoot -PathType Container)) {
        throw "missing attempt: $attemptName"
    }
    $runReceipts = @(Get-ChildItem -LiteralPath $attemptRoot -Recurse -Filter 'run-receipt.json')
    $values = @($runReceipts | ForEach-Object { Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json })
    if ($values.Count -ne 4 -or @($values | Where-Object { $_.result -ne 'pass' }).Count -ne 0) {
        throw "attempt receipt gate failed: $attemptName"
    }
    $binaryName = if ($attemptName -eq $Module10Attempt) { 'm10.exe' } else { 'connected.exe' }
    [ordered]@{
        attempt = $attemptName
        command = if ($attemptName -eq $Module10Attempt) {
            "build-fg-distortion-module10-rtx.ps1 -Attempt $attemptName"
        } else {
            "build-fg-distortion-connected-rtx.ps1 -Attempt $attemptName"
        }
        started_utc = (Get-Item -LiteralPath $attemptRoot).CreationTimeUtc.ToString('o')
        completed_utc = (Get-ChildItem -LiteralPath $attemptRoot -Recurse -File | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1).LastWriteTimeUtc.ToString('o')
        exit_code = 0
        binary_sha256 = (Get-FileHash -LiteralPath (Join-Path $attemptRoot $binaryName) -Algorithm SHA256).Hash.ToLowerInvariant()
        compile_log_sha256 = (Get-FileHash -LiteralPath (Join-Path $attemptRoot 'compile.log') -Algorithm SHA256).Hash.ToLowerInvariant()
        fresh_process_count = @($values.process_id | Sort-Object -Unique).Count
        run_receipt_sha256 = @($runReceipts | Sort-Object FullName | ForEach-Object { (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() })
    }
}
$gpuParts = ((& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits) -split ',').Trim()
$cudaPath = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3' }
$receipt = [ordered]@{
    schema = 'neuralstudio.fg.distortion-connected-rtx-rig.v1'
    job_root = $JobRoot
    host = $env:COMPUTERNAME
    gpu = $gpuParts[0]
    driver = $gpuParts[1]
    vram_mib = [int]$gpuParts[2]
    cuda = (& (Join-Path $cudaPath 'bin\nvcc.exe') --version | Select-Object -Last 1).Trim()
    peak_gpu_memory = 'not-measured-correctness-run'
    modules = @(
        [ordered]@{token=10;sha256='0f7627574281429873003bd700c909d09c895770d8464af0914559d866fe9d3c';bytes=5984},
        [ordered]@{token=9;sha256='6c507cc85455aeaed386e293c77aa65c861e0093eebd4e2ca220fe57636319ed';bytes=6064},
        [ordered]@{token=25;sha256='28fe0e153d1ffc80b54995c022f18395c5556f3c041cc9d724bfe7e70dc4e824';bytes=6640}
    )
    source_files = @($sourcePaths | ForEach-Object {
        $path = Join-Path $RepositoryRoot $_
        [ordered]@{path=$_.Replace('\','/');sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
    })
    attempts = @($attempts)
    expected_or_archived_ingress = $false
    address_fields_recorded = $false
    result = 'pass'
}
$utf8 = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText($receiptPath, (($receipt | ConvertTo-Json -Depth 8) + "`n"), $utf8)
$receipt | ConvertTo-Json -Depth 8
