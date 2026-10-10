[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$JobRoot,
    [string]$RunnerDirectory,
    [Parameter(Mandatory)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$')]
    [string]$RunId,
    [ValidateRange(0, 255)]
    [int]$OutputPoisonByte = 165,
    [switch]$DisableInterpolationOutput,
    [switch]$OutputReal,
    [switch]$Hudless,
    [switch]$UiAlpha,
    [switch]$UiColorAlpha,
    [switch]$BidirectionalDistortion,
    [switch]$MenuDetection,
    [switch]$ResetEveryFrame,
    [switch]$Hdr,
    [switch]$CaptureRuntimeAllocations,
    [switch]$CaptureRuntimeUploadBuffers,
    [ValidateRange(1, 1000000)]
    [int]$ExpectedRuntimeUploadBufferCount = 67,
    [UInt64]$ExpectedRuntimeUploadBytes = 4868576,
    [switch]$CaptureRuntimeDefaultBuffers,
    [ValidatePattern('^[0-9]+(?:,[0-9]+)*$')]
    [string]$RuntimeDefaultBufferTokens = '63',
    [ValidateSet('post_create', 'post_evaluate_0', 'post_evaluate_1')]
    [string]$RuntimeDefaultBufferCapturePoint = 'post_create',
    [ValidateRange(1, 1000000)]
    [int]$ExpectedRuntimeDefaultBufferCount = 1,
    [UInt64]$ExpectedRuntimeDefaultBufferBytes = 56320,
    [ValidatePattern('^[0-9A-Fa-f]{64}$')]
    [string]$ExpectedBaselineOutputSha256,
    [ValidateRange(0, 3840)]
    [int]$SubrectX = 0,
    [ValidateRange(0, 2160)]
    [int]$SubrectY = 0,
    [ValidateRange(0, 3840)]
    [int]$SubrectWidth = 0,
    [ValidateRange(0, 2160)]
    [int]$SubrectHeight = 0,
    [ValidateRange(1, 3)]
    [int]$MultiFrameCount = 1,
    [ValidateRange(1, 3)]
    [int]$MultiFrameIndex = 1,
    [ValidateRange(128, 3840)]
    [int]$Width = 640,
    [ValidateRange(72, 2160)]
    [int]$Height = 360,
    [ValidateRange(2, 16)]
    [int]$RenderedFrames = 2
)

$ErrorActionPreference = 'Stop'
$JobRoot = [IO.Path]::GetFullPath($JobRoot).TrimEnd('\')
$jobPrefix = [IO.Path]::GetFullPath('D:\CodexLab').TrimEnd('\') + '\'
if (-not $JobRoot.StartsWith($jobPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "FG jobs must stay beneath D:\CodexLab: $JobRoot"
}
if ($MultiFrameIndex -gt $MultiFrameCount) {
    throw 'MultiFrameIndex cannot exceed MultiFrameCount.'
}
if (($UiAlpha -or $UiColorAlpha) -and -not $Hudless) {
    throw 'UI resources require -Hudless.'
}
if ($UiAlpha -and $UiColorAlpha) {
    throw 'Choose either -UiAlpha or -UiColorAlpha.'
}
if ($CaptureRuntimeAllocations -and -not $ExpectedBaselineOutputSha256) {
    throw '-CaptureRuntimeAllocations requires -ExpectedBaselineOutputSha256.'
}
if (-not $CaptureRuntimeAllocations -and $ExpectedBaselineOutputSha256) {
    throw '-ExpectedBaselineOutputSha256 is only valid with -CaptureRuntimeAllocations.'
}
if ($CaptureRuntimeUploadBuffers -and -not $CaptureRuntimeAllocations) {
    throw '-CaptureRuntimeUploadBuffers requires -CaptureRuntimeAllocations.'
}
if ($CaptureRuntimeUploadBuffers -and $ExpectedRuntimeUploadBytes -eq 0) {
    throw '-ExpectedRuntimeUploadBytes must be positive.'
}
if ($CaptureRuntimeDefaultBuffers -and -not $CaptureRuntimeAllocations) {
    throw '-CaptureRuntimeDefaultBuffers requires -CaptureRuntimeAllocations.'
}
if ($CaptureRuntimeDefaultBuffers -and $CaptureRuntimeUploadBuffers) {
    throw 'Runtime upload and DEFAULT readback captures must run separately.'
}
if ($CaptureRuntimeDefaultBuffers -and $ExpectedRuntimeDefaultBufferBytes -eq 0) {
    throw '-ExpectedRuntimeDefaultBufferBytes must be positive.'
}
if ($CaptureRuntimeDefaultBuffers -and
    $RuntimeDefaultBufferCapturePoint -eq 'post_evaluate_1' -and
    $RenderedFrames -lt 3) {
    throw 'post_evaluate_1 DEFAULT capture requires RenderedFrames >= 3.'
}
if (($SubrectWidth -eq 0) -xor ($SubrectHeight -eq 0)) {
    throw 'SubrectWidth and SubrectHeight must both be zero or both be nonzero.'
}
if ($SubrectWidth -gt 0 -and
    ($SubrectX + $SubrectWidth -gt $Width -or
     $SubrectY + $SubrectHeight -gt $Height)) {
    throw 'Requested subrect exceeds the configured frame dimensions.'
}

$runner = if ($RunnerDirectory) {
    [IO.Path]::GetFullPath($RunnerDirectory)
} else {
    Join-Path $JobRoot 'build\fg_traced_runner'
}
if (-not $runner.StartsWith($JobRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw "Traced runner must stay inside the FG job: $runner"
}
$exe = Join-Path $runner 'neuralstudio-fg-standalone.exe'
$runtime = Join-Path $runner 'nvngx_dlssg.dll'
$thunk = Join-Path $runner 'nvngx.dll'
$proxy = Join-Path $runner 'nvapi64.dll'
$realNvapi = Join-Path $runner 'nvapi64.real.dll'
foreach ($required in @($exe, $runtime, $thunk, $proxy, $realNvapi)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required traced-runner file is missing: $required"
    }
}

$outputDirectory = Join-Path $JobRoot "output\$RunId"
$traceDirectory = Join-Path $JobRoot "output\$RunId-trace"
$dumpDirectory = Join-Path $traceDirectory 'dumps'
$tracePath = Join-Path $traceDirectory 'nvapi-trace.jsonl'
$volatileTracePath = Join-Path $traceDirectory 'nvapi-volatile.jsonl'
$logPath = Join-Path $JobRoot "logs\$RunId.log"
$processReceiptPath = Join-Path $JobRoot "receipts\$RunId-process.json"
foreach ($fresh in @($outputDirectory, $traceDirectory, $processReceiptPath)) {
    if (Test-Path -LiteralPath $fresh) {
        throw "Run target must be fresh: $fresh"
    }
}
New-Item -ItemType Directory -Path $dumpDirectory -Force | Out-Null

$realNvapiHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $realNvapi).Hash.ToLowerInvariant()
$environment = [ordered]@{
    NEURALSTUDIO_FG_REAL_NVAPI64 = $realNvapi
    NEURALSTUDIO_FG_REAL_NVAPI64_SHA256 = $realNvapiHash
    NEURALSTUDIO_FG_NVAPI_TRACE = $tracePath
    NEURALSTUDIO_FG_NVAPI_DUMP_DIR = $dumpDirectory
    NEURALSTUDIO_FG_NVAPI_VOLATILE_TRACE = $volatileTracePath
}
$previousEnvironment = @{}
foreach ($name in $environment.Keys) {
    $previousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
    [Environment]::SetEnvironmentVariable($name, $environment[$name], 'Process')
}

$arguments = @(
    '--arm-execution',
    '--runtime', $runtime,
    '--thunk', $thunk,
    '--nvapi-proxy', $proxy,
    '--output-dir', $outputDirectory,
    '--width', [string]$Width,
    '--height', [string]$Height,
    '--rendered-frames', [string]$RenderedFrames,
    '--output-poison-byte', [string]$OutputPoisonByte,
    '--multi-frame-count', [string]$MultiFrameCount,
    '--multi-frame-index', [string]$MultiFrameIndex
)
if ($DisableInterpolationOutput) { $arguments += '--disable-interpolation-output' }
if ($OutputReal) { $arguments += '--output-real' }
if ($Hudless) { $arguments += '--hudless' }
if ($UiAlpha) { $arguments += '--ui-alpha' }
if ($UiColorAlpha) { $arguments += '--ui-color-alpha' }
if ($BidirectionalDistortion) { $arguments += '--bidirectional-distortion' }
if ($MenuDetection) { $arguments += '--menu-detection' }
if ($ResetEveryFrame) { $arguments += '--reset-every-frame' }
if ($Hdr) { $arguments += '--hdr' }
if ($CaptureRuntimeAllocations) {
    $arguments += @(
        '--capture-runtime-allocations',
        '--expected-baseline-output-sha256',
        $ExpectedBaselineOutputSha256.ToLowerInvariant()
    )
}
if ($CaptureRuntimeUploadBuffers) {
    $arguments += @(
        '--capture-runtime-upload-buffers',
        '--expected-runtime-upload-buffer-count',
        [string]$ExpectedRuntimeUploadBufferCount,
        '--expected-runtime-upload-bytes',
        [string]$ExpectedRuntimeUploadBytes
    )
}
if ($CaptureRuntimeDefaultBuffers) {
    $arguments += @(
        '--capture-runtime-default-buffers',
        '--runtime-default-buffer-tokens',
        $RuntimeDefaultBufferTokens,
        '--runtime-default-buffer-capture-point',
        $RuntimeDefaultBufferCapturePoint,
        '--expected-runtime-default-buffer-count',
        [string]$ExpectedRuntimeDefaultBufferCount,
        '--expected-runtime-default-buffer-bytes',
        [string]$ExpectedRuntimeDefaultBufferBytes
    )
}
if ($SubrectWidth -gt 0) {
    $arguments += @(
        '--subrect-x', [string]$SubrectX,
        '--subrect-y', [string]$SubrectY,
        '--subrect-width', [string]$SubrectWidth,
        '--subrect-height', [string]$SubrectHeight
    )
}

$started = (Get-Date).ToUniversalTime()
$exitCode = 0
$errorText = $null
Push-Location $runner
try {
    & $exe @arguments *>&1 | Set-Content -LiteralPath $logPath -Encoding UTF8
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "FG host exited $exitCode"
    }
    if (-not (Test-Path -LiteralPath $tracePath -PathType Leaf)) {
        throw 'NvAPI forwarding proxy produced no trace.'
    }
} catch {
    if ($exitCode -eq 0) { $exitCode = 1 }
    $errorText = $_.Exception.Message
} finally {
    Pop-Location
    foreach ($name in $environment.Keys) {
        [Environment]::SetEnvironmentVariable(
            $name, $previousEnvironment[$name], 'Process')
    }
}
$ended = (Get-Date).ToUniversalTime()

$hostReceiptPath = Join-Path $outputDirectory 'fg-run-receipt.json'
$dumpFiles = @(
    Get-ChildItem -LiteralPath $dumpDirectory -File -ErrorAction SilentlyContinue |
        Sort-Object Name |
        ForEach-Object {
            [ordered]@{
                name = $_.Name
                bytes = $_.Length
                sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
            }
        }
)
$receipt = [ordered]@{
    schema = 'neuralstudio.dlss_fg.traced_process.v1'
    job_id = Split-Path -Leaf $JobRoot
    run_id = $RunId
    command = @($exe) + $arguments
    working_directory = $runner
    started_utc = $started.ToString('o')
    ended_utc = $ended.ToString('o')
    exit_code = $exitCode
    host = $env:COMPUTERNAME
    gpu = (nvidia-smi --query-gpu=name --format=csv,noheader)
    driver = (nvidia-smi --query-gpu=driver_version --format=csv,noheader)
    output_poison_byte = $OutputPoisonByte
    rendered_frames = $RenderedFrames
    capture_runtime_allocations = [bool]$CaptureRuntimeAllocations
    capture_runtime_upload_buffers = [bool]$CaptureRuntimeUploadBuffers
    expected_runtime_upload_buffer_count = $ExpectedRuntimeUploadBufferCount
    expected_runtime_upload_bytes = $ExpectedRuntimeUploadBytes
    capture_runtime_default_buffers = [bool]$CaptureRuntimeDefaultBuffers
    runtime_default_buffer_tokens = $RuntimeDefaultBufferTokens
    runtime_default_buffer_capture_point = $RuntimeDefaultBufferCapturePoint
    expected_runtime_default_buffer_count = $ExpectedRuntimeDefaultBufferCount
    expected_runtime_default_buffer_bytes = $ExpectedRuntimeDefaultBufferBytes
    executable_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.ToLowerInvariant()
    runtime_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $runtime).Hash.ToLowerInvariant()
    thunk_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $thunk).Hash.ToLowerInvariant()
    proxy_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $proxy).Hash.ToLowerInvariant()
    real_nvapi_sha256 = $realNvapiHash
    host_receipt_sha256 = $(
        if (Test-Path -LiteralPath $hostReceiptPath -PathType Leaf) {
            (Get-FileHash -Algorithm SHA256 -LiteralPath $hostReceiptPath).Hash.ToLowerInvariant()
        } else { $null }
    )
    trace_sha256 = $(
        if (Test-Path -LiteralPath $tracePath -PathType Leaf) {
            (Get-FileHash -Algorithm SHA256 -LiteralPath $tracePath).Hash.ToLowerInvariant()
        } else { $null }
    )
    trace_lines = $(
        if (Test-Path -LiteralPath $tracePath -PathType Leaf) {
            @(Get-Content -LiteralPath $tracePath).Count
        } else { 0 }
    )
    volatile_trace_sha256 = $(
        if (Test-Path -LiteralPath $volatileTracePath -PathType Leaf) {
            (Get-FileHash -Algorithm SHA256 -LiteralPath $volatileTracePath).Hash.ToLowerInvariant()
        } else { $null }
    )
    volatile_trace_lines = $(
        if (Test-Path -LiteralPath $volatileTracePath -PathType Leaf) {
            @(Get-Content -LiteralPath $volatileTracePath).Count
        } else { 0 }
    )
    dump_files = $dumpFiles
    error = $errorText
}
$receipt | ConvertTo-Json -Depth 8 |
    Set-Content -LiteralPath $processReceiptPath -Encoding UTF8
$receipt | ConvertTo-Json -Compress -Depth 8
exit $exitCode
