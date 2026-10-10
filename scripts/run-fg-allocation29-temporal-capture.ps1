[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$JobRoot
)

$ErrorActionPreference = 'Stop'
$JobRoot = [IO.Path]::GetFullPath($JobRoot).TrimEnd('\')
$prefix = [IO.Path]::GetFullPath('D:\CodexLab').TrimEnd('\') + '\'
if (-not $JobRoot.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Temporal capture job must stay beneath D:\CodexLab'
}

$sourceRoot = Join-Path $JobRoot 'source'
$inputRoot = Join-Path $JobRoot 'input'
$hostBuild = Join-Path $JobRoot 'build\fg_host-temporal-v1'
$runner = Join-Path $JobRoot 'build\fg_traced_runner'
$buildScript = Join-Path $sourceRoot 'scripts\build-fg-standalone.ps1'
$runScript = Join-Path $sourceRoot 'scripts\run-fg-traced-job.ps1'
foreach ($required in @(
    $buildScript,
    $runScript,
    (Join-Path $sourceRoot 'src\fg_host\main.cpp'),
    (Join-Path $sourceRoot 'src\fg_host\runtime_allocation_capture.cpp'),
    (Join-Path $inputRoot 'sdk\include\nvsdk_ngx.h'),
    (Join-Path $inputRoot 'runtime\nvngx_dlssg.dll'),
    (Join-Path $runner 'nvapi64.dll'),
    (Join-Path $runner 'nvapi64.real.dll')
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Missing temporal capture input: $required"
    }
}
if (Test-Path -LiteralPath $hostBuild) {
    throw "Host build already exists: $hostBuild"
}

& $buildScript `
    -NgxInclude (Join-Path $inputRoot 'sdk\include') `
    -RuntimePath (Join-Path $inputRoot 'runtime\nvngx_dlssg.dll') `
    -BuildDirectory $hostBuild *> (Join-Path $JobRoot 'logs\build-host.log')
foreach ($name in @('neuralstudio-fg-standalone.exe', 'nvngx.dll',
                     'nvngx_dlssg.dll')) {
    Copy-Item -LiteralPath (Join-Path $hostBuild $name) `
        -Destination (Join-Path $runner $name) -Force
}

$baseline = '07a87435b2e7d79540dcec737b9abb6354e68e8595b4df29ca6f8d92cce54383'
$runs = @()
foreach ($poison in @(165, 90)) {
    foreach ($processOrdinal in @('a', 'b')) {
        $runId = "allocation29-post-generated-poison-$('{0:x2}' -f $poison)-process-$processOrdinal"
        $descriptorRoot = Join-Path $JobRoot "descriptors\$runId"
        if (Test-Path -LiteralPath $descriptorRoot) {
            throw "Descriptor capture target already exists: $descriptorRoot"
        }
        New-Item -ItemType Directory -Path $descriptorRoot | Out-Null
        $descriptorSemantic = Join-Path $descriptorRoot 'descriptor-semantic.jsonl'
        $descriptorVolatile = Join-Path $descriptorRoot 'descriptor-volatile.jsonl'
        $previousDescriptorSemantic = [Environment]::GetEnvironmentVariable(
            'NEURALSTUDIO_FG_D3D12_DESCRIPTOR_TRACE', 'Process')
        $previousDescriptorVolatile = [Environment]::GetEnvironmentVariable(
            'NEURALSTUDIO_FG_D3D12_DESCRIPTOR_VOLATILE_TRACE', 'Process')
        try {
            [Environment]::SetEnvironmentVariable(
                'NEURALSTUDIO_FG_D3D12_DESCRIPTOR_TRACE',
                $descriptorSemantic,
                'Process')
            [Environment]::SetEnvironmentVariable(
                'NEURALSTUDIO_FG_D3D12_DESCRIPTOR_VOLATILE_TRACE',
                $descriptorVolatile,
                'Process')
            & $runScript -JobRoot $JobRoot -RunnerDirectory $runner `
                -RunId $runId -OutputPoisonByte $poison -RenderedFrames 3 `
                -DisableInterpolationOutput -CaptureRuntimeAllocations `
                -ExpectedBaselineOutputSha256 $baseline `
                -CaptureRuntimeDefaultBuffers -RuntimeDefaultBufferTokens '29' `
                -RuntimeDefaultBufferCapturePoint 'post_evaluate_1' `
                -ExpectedRuntimeDefaultBufferCount 1 `
                -ExpectedRuntimeDefaultBufferBytes 115200
            if ($LASTEXITCODE -ne 0) {
                throw "Temporal traced run failed: $runId"
            }
        } finally {
            [Environment]::SetEnvironmentVariable(
                'NEURALSTUDIO_FG_D3D12_DESCRIPTOR_TRACE',
                $previousDescriptorSemantic,
                'Process')
            [Environment]::SetEnvironmentVariable(
                'NEURALSTUDIO_FG_D3D12_DESCRIPTOR_VOLATILE_TRACE',
                $previousDescriptorVolatile,
                'Process')
        }

        $runRoot = Join-Path $JobRoot "output\$runId"
        $dump = Join-Path $runRoot `
            'runtime-default-buffers-post-evaluate-1\default-token-000029.bin'
        $manifest = Join-Path $runRoot `
            'runtime-default-buffers-post-evaluate-1.manifest.json'
        $hostReceiptPath = Join-Path $runRoot 'fg-run-receipt.json'
        $processReceiptPath = Join-Path $JobRoot "receipts\$runId-process.json"
        foreach ($required in @($descriptorSemantic, $descriptorVolatile)) {
            if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
                throw "Temporal descriptor capture is missing: $required"
            }
        }
        if ((Get-Item -LiteralPath $dump).Length -ne 115200) {
            throw "allocation29 temporal dump size drift: $runId"
        }
        $hostReceipt = Get-Content -LiteralPath $hostReceiptPath -Raw |
            ConvertFrom-Json
        if (@($hostReceipt.frames).Count -ne 3) {
            throw "temporal run did not produce exactly three frames: $runId"
        }
        if (-not $hostReceipt.frames[1].generated_output_accepted -or
            -not $hostReceipt.frames[2].generated_output_accepted) {
            throw "temporal generated-frame acceptance failed: $runId"
        }
        if ($hostReceipt.frames[1].output_interpolated_sha256 -ne $baseline) {
            throw "first generated frame baseline drift: $runId"
        }
        $runs += [ordered]@{
            run_id = $runId
            poison_byte = $poison
            process_ordinal = $processOrdinal
            rendered_frames = 3
            capture_point = 'post_evaluate_1_fence_pre_evaluate_2'
            dump_bytes = 115200
            dump_sha256 = (Get-FileHash -LiteralPath $dump -Algorithm SHA256).Hash.ToLowerInvariant()
            manifest_sha256 = (Get-FileHash -LiteralPath $manifest -Algorithm SHA256).Hash.ToLowerInvariant()
            host_receipt_sha256 = (Get-FileHash -LiteralPath $hostReceiptPath -Algorithm SHA256).Hash.ToLowerInvariant()
            process_receipt_sha256 = (Get-FileHash -LiteralPath $processReceiptPath -Algorithm SHA256).Hash.ToLowerInvariant()
            descriptor_semantic_sha256 = (Get-FileHash -LiteralPath $descriptorSemantic -Algorithm SHA256).Hash.ToLowerInvariant()
            descriptor_volatile_sha256 = (Get-FileHash -LiteralPath $descriptorVolatile -Algorithm SHA256).Hash.ToLowerInvariant()
            first_generated_output_sha256 = $hostReceipt.frames[1].output_interpolated_sha256
            second_generated_output_sha256 = $hostReceipt.frames[2].output_interpolated_sha256
            first_generated_untouched_pixels = $hostReceipt.frames[1].output_untouched_pixels
            second_generated_untouched_pixels = $hostReceipt.frames[2].output_untouched_pixels
        }
    }
}

$receipt = [ordered]@{
    schema = 'neuralstudio.fg.allocation29-temporal-capture.v1'
    job_id = Split-Path -Leaf $JobRoot
    runtime_sha256 = (Get-FileHash -LiteralPath `
        (Join-Path $inputRoot 'runtime\nvngx_dlssg.dll') `
        -Algorithm SHA256).Hash.ToLowerInvariant()
    host_source_sha256 = (Get-FileHash -LiteralPath `
        (Join-Path $sourceRoot 'src\fg_host\main.cpp') `
        -Algorithm SHA256).Hash.ToLowerInvariant()
    allocation_capture_source_sha256 = (Get-FileHash -LiteralPath `
        (Join-Path $sourceRoot 'src\fg_host\runtime_allocation_capture.cpp') `
        -Algorithm SHA256).Hash.ToLowerInvariant()
    orchestration_source_sha256 = (Get-FileHash -LiteralPath `
        (Join-Path $sourceRoot 'scripts\run-fg-allocation29-temporal-capture.ps1') `
        -Algorithm SHA256).Hash.ToLowerInvariant()
    host_binary_sha256 = (Get-FileHash -LiteralPath `
        (Join-Path $hostBuild 'neuralstudio-fg-standalone.exe') `
        -Algorithm SHA256).Hash.ToLowerInvariant()
    runs = $runs
    fresh_processes = 4
    poison_values = @(165, 90)
    capture_between_generated_evaluations = $true
    d3d12_descriptor_capture = $true
    expected_frame_bytes_fed_to_capture = $false
    captured_bytes_fed_back_to_execution = $false
    address_fields_recorded = $false
    result = 'pass'
}
$utf8 = New-Object System.Text.UTF8Encoding($false)
$receiptPath = Join-Path $JobRoot 'allocation29-temporal-capture-receipt.json'
[IO.File]::WriteAllText(
    $receiptPath,
    (($receipt | ConvertTo-Json -Depth 7) + "`n"),
    $utf8)
Write-Host "receipt=$receiptPath"
Write-Host "receipt_sha256=$((Get-FileHash -LiteralPath $receiptPath -Algorithm SHA256).Hash.ToLowerInvariant())"
Write-Host 'result=pass'
