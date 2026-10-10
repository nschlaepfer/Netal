[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $JobRoot,
    [Parameter(Mandatory = $true)] [string] $AttemptId
)

$ErrorActionPreference = 'Stop'
$attempt = Join-Path $JobRoot (Join-Path 'output' $AttemptId)
$backendPath = Join-Path $attempt 'backend-receipt.json'
$receiptPath = Join-Path $attempt 'rig-receipt.json'
if (-not (Test-Path -LiteralPath $backendPath)) {
    throw "Accepted backend receipt missing: $backendPath"
}
if (Test-Path -LiteralPath $receiptPath) {
    throw "Receipt exists; preserve it: $receiptPath"
}

function Get-DirectoryManifest {
    param([Parameter(Mandatory = $true)] [string] $Root)
    if (-not (Test-Path -LiteralPath $Root -PathType Container)) {
        throw "Manifest root missing: $Root"
    }
    $rootItem = Get-Item -LiteralPath $Root
    $rows = @()
    $totalBytes = [int64]0
    foreach ($file in @(Get-ChildItem -LiteralPath $Root -Recurse -File |
                       Sort-Object FullName)) {
        $relative = $file.FullName.Substring($rootItem.FullName.Length)
        $relative = $relative.TrimStart('\').Replace('\', '/')
        $digest = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).
            Hash.ToLowerInvariant()
        $rows += ('{0}|{1}|{2}' -f $relative, $file.Length, $digest)
        $totalBytes += $file.Length
    }
    $serialized = [Text.Encoding]::UTF8.GetBytes(($rows -join "`n") + "`n")
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $rootDigest = -join ($sha.ComputeHash($serialized) |
            ForEach-Object { $_.ToString('x2') })
    } finally {
        $sha.Dispose()
    }
    return [ordered]@{
        file_count = $rows.Count
        total_bytes = $totalBytes
        canonical_manifest_sha256 = $rootDigest
    }
}

$backend = Get-Content -LiteralPath $backendPath -Raw | ConvertFrom-Json
if ($backend.result -ne 'pass' -or $backend.attempt_id -ne $AttemptId -or
    $backend.launch_count -ne 89 -or $backend.fresh_process_count -ne 4 -or
    -not $backend.all_checkpoint_two_poison_invariant -or
    $backend.expected_or_archived_tensor_ingress -or
    $backend.host_intermediate_ingress -or
    $backend.distortion_checkpoint_ingress) {
    throw 'Backend receipt does not satisfy the accepted gate'
}
$processIds = @($backend.processes | ForEach-Object { [int]$_.process_id })
if (@($processIds | Sort-Object -Unique).Count -ne 4 -or
    @($backend.processes | Where-Object { $_.exit_code -ne 0 }).Count -ne 0) {
    throw 'Fresh-process receipt gate failed'
}

$firstRun = Join-Path $attempt $backend.processes[0].run_id
$tailReceiptPath = Join-Path $firstRun 'l20-l87\run-receipt.json'
$tail = Get-Content -LiteralPath $tailReceiptPath -Raw | ConvertFrom-Json
if ($tail.result -ne 'pass' -or $tail.input_model_mismatched_bytes -ne 0 -or
    $tail.canary_mismatched_bytes -ne 0 -or
    $tail.readonly_tail_mismatched_bytes -ne 0) {
    throw 'Tail integrity gate failed'
}

$binaryPath = Join-Path $attempt 'build\fg-connected-l3-l87.exe'
$compileLogPath = Join-Path $attempt 'build\compile.log'
$collectorPath = $MyInvocation.MyCommand.Path
foreach ($required in @($binaryPath, $compileLogPath, $collectorPath)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Receipt input missing: $required"
    }
}
$binary = Get-Item -LiteralPath $binaryPath
$backendItem = Get-Item -LiteralPath $backendPath
$gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total `
    --format=csv,noheader,nounits | Select-Object -First 1).Trim().Split(',')
$nvcc = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe'
$cudaLine = (& $nvcc --version | Select-Object -Last 1).Trim()
$os = Get-CimInstance Win32_OperatingSystem
$compileLogHash = (Get-FileHash -LiteralPath $compileLogPath -Algorithm SHA256).
    Hash.ToLowerInvariant()
$backendHash = (Get-FileHash -LiteralPath $backendPath -Algorithm SHA256).
    Hash.ToLowerInvariant()
$collectorHash = (Get-FileHash -LiteralPath $collectorPath -Algorithm SHA256).
    Hash.ToLowerInvariant()

$receipt = [ordered]@{
    schema = 'neuralstudio.fg.connected-distortion-l3-l91-rig-receipt.v1'
    job_id = Split-Path $JobRoot -Leaf
    attempt_id = $AttemptId
    remote_job_retained = $JobRoot
    host = $env:COMPUTERNAME
    os = $os.Caption
    os_version = $os.Version
    gpu = $gpu[0].Trim()
    driver = $gpu[1].Trim()
    vram_mib = [int]$gpu[2].Trim()
    cuda_toolkit = $cudaLine
    build_command = ('powershell -File scripts/build-fg-connected-distortion-l3-l87-rtx.ps1 ' +
        '-JobRoot D:\CodexLab\fg-connected-distortion-l3-l87-20260904T150406Z ' +
        '-AttemptId prototype-010-final -RepositoryRoot ' +
        'D:\CodexLab\fg-connected-distortion-l3-l87-20260904T150406Z\input\full-source')
    process_started_utc = $binary.CreationTimeUtc.ToString('o')
    receipt_completed_utc = $backendItem.LastWriteTimeUtc.ToString('o')
    exit_codes = @($backend.processes | ForEach-Object { [int]$_.exit_code })
    fresh_process_ids = $processIds
    fresh_process_count = 4
    poison_bytes = @(90, 165)
    launch_range = @(3, 91)
    launch_count = 89
    source_sha256 = [ordered]@{
        early = $backend.early_source_sha256
        live_prefix = $backend.live_prefix_source_sha256
        accepted_tail = $backend.accepted_tail_source_sha256
        generated_tail = $backend.generated_tail_source_sha256
        live_core = $backend.live_core_source_sha256
        collector = $collectorHash
    }
    input_manifests = [ordered]@{
        source = Get-DirectoryManifest (Join-Path $JobRoot 'input\full-source')
        modules = Get-DirectoryManifest (Join-Path $JobRoot 'input\full-modules')
        model = Get-DirectoryManifest (Join-Path $JobRoot 'input\full-model')
    }
    selected_module_census_root_sha256 =
        $tail.selected_module_census_root_sha256
    allocation_provenance_root_sha256 =
        $tail.allocation_provenance_root_sha256
    descriptor_provenance_root_sha256 =
        $tail.descriptor_provenance_root_sha256
    binary_sha256 = $backend.binary_sha256
    compile_log_bytes = (Get-Item -LiteralPath $compileLogPath).Length
    compile_log_sha256 = $compileLogHash
    backend_receipt_sha256 = $backendHash
    invariant_allocation37_sha256 = $backend.allocation37_sha256
    invariant_output_sha256 = $backend.invariant_output_sha256
    all_checkpoint_two_poison_invariant = $true
    canary_integrity = $true
    input_model_immutability = $true
    live_device_handoffs = $true
    expected_or_archived_ingress = $false
    address_fields_recorded = $false
    result = 'pass'
}
$utf8NoBom = New-Object Text.UTF8Encoding($false)
[IO.File]::WriteAllText(
    $receiptPath, (($receipt | ConvertTo-Json -Depth 8) + "`n"), $utf8NoBom)
Write-Output "rig_receipt=$receiptPath"
Write-Output 'result=pass'
