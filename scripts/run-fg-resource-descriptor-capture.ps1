[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$JobRoot,
    [Parameter(Mandatory)]
    [string]$RunnerDirectory,
    [Parameter(Mandatory)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$')]
    [string]$RunId,
    [Parameter(Mandatory)]
    [ValidateRange(0, 255)]
    [int]$OutputPoisonByte,
    [Parameter(Mandatory)]
    [ValidatePattern('^[0-9A-Fa-f]{64}$')]
    [string]$ExpectedBaselineOutputSha256
)

$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent $PSScriptRoot
$runnerScript = Join-Path $PSScriptRoot 'run-fg-traced-job.ps1'
$JobRoot = [IO.Path]::GetFullPath($JobRoot).TrimEnd('\')
$RunnerDirectory = [IO.Path]::GetFullPath($RunnerDirectory).TrimEnd('\')
$jobPrefix = [IO.Path]::GetFullPath('D:\CodexLab').TrimEnd('\') + '\'
if (-not $JobRoot.StartsWith($jobPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "FG descriptor jobs must stay beneath D:\CodexLab: $JobRoot"
}
if (-not $RunnerDirectory.StartsWith($JobRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw "FG descriptor runner must stay inside the job: $RunnerDirectory"
}
if (-not (Test-Path -LiteralPath $runnerScript -PathType Leaf)) {
    throw "Traced runner script is missing: $runnerScript"
}

$descriptorDirectory = Join-Path $JobRoot "descriptors\$RunId"
$semanticPath = Join-Path $descriptorDirectory 'descriptor-semantic.jsonl'
$volatilePath = Join-Path $descriptorDirectory 'descriptor-volatile.jsonl'
$descriptorReceiptPath = Join-Path $JobRoot "receipts\$RunId-resource-descriptor-process.json"
foreach ($fresh in @($descriptorDirectory, $descriptorReceiptPath)) {
    if (Test-Path -LiteralPath $fresh) {
        throw "Descriptor capture target must be fresh: $fresh"
    }
}
New-Item -ItemType Directory -Path $descriptorDirectory | Out-Null

$environment = [ordered]@{
    NEURALSTUDIO_FG_D3D12_DESCRIPTOR_TRACE = $semanticPath
    NEURALSTUDIO_FG_D3D12_DESCRIPTOR_VOLATILE_TRACE = $volatilePath
}
$previousEnvironment = @{}
foreach ($name in $environment.Keys) {
    $previousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
    [Environment]::SetEnvironmentVariable($name, $environment[$name], 'Process')
}

$started = (Get-Date).ToUniversalTime()
$exitCode = 0
$errorText = $null
try {
    & $runnerScript `
        -JobRoot $JobRoot `
        -RunnerDirectory $RunnerDirectory `
        -RunId $RunId `
        -OutputPoisonByte $OutputPoisonByte `
        -CaptureRuntimeAllocations `
        -ExpectedBaselineOutputSha256 $ExpectedBaselineOutputSha256
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "FG descriptor host exited $exitCode"
    }
} catch {
    if ($exitCode -eq 0) { $exitCode = 1 }
    $errorText = $_.Exception.Message
} finally {
    foreach ($name in $environment.Keys) {
        [Environment]::SetEnvironmentVariable(
            $name, $previousEnvironment[$name], 'Process')
    }
}
$ended = (Get-Date).ToUniversalTime()

$processReceiptPath = Join-Path $JobRoot "receipts\$RunId-process.json"
$hostReceiptPath = Join-Path $JobRoot "output\$RunId\fg-run-receipt.json"
$tracePath = Join-Path $JobRoot "output\$RunId-trace\nvapi-trace.jsonl"
$volatileTracePath = Join-Path $JobRoot "output\$RunId-trace\nvapi-volatile.jsonl"
$requiredOutputs = @(
    $processReceiptPath,
    $hostReceiptPath,
    $tracePath,
    $volatileTracePath,
    $semanticPath,
    $volatilePath
)
if ($exitCode -eq 0) {
    foreach ($required in $requiredOutputs) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            $exitCode = 1
            $errorText = "Required descriptor output is missing: $required"
            break
        }
    }
}

$hostReceipt = $null
$descriptorSummary = $null
if (Test-Path -LiteralPath $hostReceiptPath -PathType Leaf) {
    $hostReceipt = Get-Content -LiteralPath $hostReceiptPath -Raw | ConvertFrom-Json
}
if (Test-Path -LiteralPath $semanticPath -PathType Leaf) {
    $semanticLines = @(Get-Content -LiteralPath $semanticPath)
    if ($semanticLines.Count -gt 0) {
        $descriptorSummary = $semanticLines[-1] | ConvertFrom-Json
    }
}
if ($exitCode -eq 0) {
    if (-not $hostReceipt.accepted -or
        $hostReceipt.frames[1].output_interpolated_sha256 -ne
            $ExpectedBaselineOutputSha256.ToLowerInvariant()) {
        $exitCode = 1
        $errorText = 'Host receipt did not preserve the accepted generated-frame baseline.'
    } elseif ($hostReceipt.frames[1].output_untouched_pixels -ne 0) {
        $exitCode = 1
        $errorText = 'Descriptor capture left output poison bytes untouched.'
    } elseif ($descriptorSummary.event -ne 'descriptor_capture_summary' -or
              -not $descriptorSummary.capture_healthy -or
              -not $descriptorSummary.hooks_restored -or
              $descriptorSummary.unresolved_successful_request_count -ne 0) {
        $exitCode = 1
        $errorText = 'Descriptor capture summary is missing, unhealthy, or unresolved.'
    }
}

function FileReceipt([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $null }
    return [ordered]@{
        path = $Path.Substring($JobRoot.Length + 1).Replace('\', '/')
        bytes = (Get-Item -LiteralPath $Path).Length
        sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

$receipt = [ordered]@{
    schema = 'neuralstudio.fg.resource-descriptor-process.v1'
    job_id = Split-Path -Leaf $JobRoot
    run_id = $RunId
    started_utc = $started.ToString('o')
    ended_utc = $ended.ToString('o')
    exit_code = $exitCode
    error = $errorText
    host = $env:COMPUTERNAME
    gpu = (nvidia-smi --query-gpu=name --format=csv,noheader)
    driver = (nvidia-smi --query-gpu=driver_version --format=csv,noheader)
    poison_byte = $OutputPoisonByte
    expected_baseline_output_sha256 = $ExpectedBaselineOutputSha256.ToLowerInvariant()
    process_id = if ($hostReceipt) { $hostReceipt.process_id } else { $null }
    run_nonce = if ($hostReceipt) { $hostReceipt.run_nonce } else { $null }
    queue_nonce = if ($hostReceipt) { $hostReceipt.queue_nonce } else { $null }
    output_sha256 = if ($hostReceipt) { $hostReceipt.frames[1].output_interpolated_sha256 } else { $null }
    output_untouched_pixels = if ($hostReceipt) { $hostReceipt.frames[1].output_untouched_pixels } else { $null }
    host_accepted = if ($hostReceipt) { [bool]$hostReceipt.accepted } else { $false }
    descriptor_summary = $descriptorSummary
    files = [ordered]@{
        traced_process_receipt = FileReceipt $processReceiptPath
        host_receipt = FileReceipt $hostReceiptPath
        nvapi_trace = FileReceipt $tracePath
        nvapi_volatile_trace = FileReceipt $volatileTracePath
        descriptor_semantic_trace = FileReceipt $semanticPath
        descriptor_volatile_trace = FileReceipt $volatilePath
    }
    source_identity = [ordered]@{
        host_executable = FileReceipt (Join-Path $RunnerDirectory 'neuralstudio-fg-standalone.exe')
        runtime = FileReceipt (Join-Path $RunnerDirectory 'nvngx_dlssg.dll')
        thunk = FileReceipt (Join-Path $RunnerDirectory 'nvngx.dll')
        nvapi_proxy = FileReceipt (Join-Path $RunnerDirectory 'nvapi64.dll')
        real_nvapi = FileReceipt (Join-Path $RunnerDirectory 'nvapi64.real.dll')
        descriptor_tracer_source = FileReceipt (Join-Path $repository 'src\fg_trace\d3d12_descriptor_trace.cpp')
        forward_tracer_source = FileReceipt (Join-Path $repository 'src\fg_trace\nvapi_forward_trace.cpp')
        host_source = FileReceipt (Join-Path $repository 'src\fg_host\main.cpp')
    }
    accepted = $exitCode -eq 0
    claim = 'capture_provenance_only_no_model_recovery_or_cross_platform_parity'
}
$receipt | ConvertTo-Json -Depth 12 |
    Set-Content -LiteralPath $descriptorReceiptPath -Encoding UTF8
$receipt | ConvertTo-Json -Compress -Depth 12
exit $exitCode
