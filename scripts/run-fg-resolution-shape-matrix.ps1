[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$JobRoot
)

$ErrorActionPreference = 'Stop'
$JobRoot = [IO.Path]::GetFullPath($JobRoot).TrimEnd('\')
$prefix = [IO.Path]::GetFullPath('D:\CodexLab').TrimEnd('\') + '\'
if (-not $JobRoot.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Resolution-shape job must stay beneath D:\CodexLab'
}

$sourceRoot = Join-Path $JobRoot 'source'
$runner = Join-Path $JobRoot 'build\fg_traced_runner'
$runScript = Join-Path $sourceRoot 'scripts\run-fg-traced-job.ps1'
foreach ($required in @(
    $runScript,
    (Join-Path $runner 'neuralstudio-fg-standalone.exe'),
    (Join-Path $runner 'nvngx.dll'),
    (Join-Path $runner 'nvngx_dlssg.dll'),
    (Join-Path $runner 'nvapi64.dll'),
    (Join-Path $runner 'nvapi64.real.dll')
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Missing resolution-shape input: $required"
    }
}

$resolutions = @(
    [ordered]@{ name = '720p'; width = 1280; height = 720 },
    [ordered]@{ name = '1080p'; width = 1920; height = 1080 },
    [ordered]@{ name = '1440p'; width = 2560; height = 1440 },
    [ordered]@{ name = '4k'; width = 3840; height = 2160 }
)
$utf8 = New-Object System.Text.UTF8Encoding($false)
$runs = @()
foreach ($resolution in $resolutions) {
    $resolutionRuns = @()
    $resolutionResetLaunches = $null
    $resolutionGeneratedLaunches = $null
    foreach ($poison in @(165, 90)) {
        foreach ($processOrdinal in @('a', 'b')) {
            $runId = "shape-$($resolution.name)-poison-$('{0:x2}' -f $poison)-process-$processOrdinal"
            & $runScript -JobRoot $JobRoot -RunnerDirectory $runner `
                -RunId $runId -OutputPoisonByte $poison `
                -Width $resolution.width -Height $resolution.height `
                -DisableInterpolationOutput | Out-Null
            if ($LASTEXITCODE -ne 0) {
                throw "Resolution-shape traced run failed: $runId"
            }

            $runRoot = Join-Path $JobRoot "output\$runId"
            $hostReceiptPath = Join-Path $runRoot 'fg-run-receipt.json'
            $processReceiptPath = Join-Path $JobRoot "receipts\$runId-process.json"
            $tracePath = Join-Path $JobRoot "output\$runId-trace\nvapi-trace.jsonl"
            foreach ($required in @($hostReceiptPath, $processReceiptPath, $tracePath)) {
                if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
                    throw "Resolution-shape output is missing: $required"
                }
            }
            $hostReceipt = Get-Content -LiteralPath $hostReceiptPath -Raw |
                ConvertFrom-Json
            if (-not $hostReceipt.accepted -or @($hostReceipt.frames).Count -ne 2) {
                throw "Resolution-shape host rejected or frame count drifted: $runId"
            }
            $generated = $hostReceipt.frames[1]
            $expectedBytes = [UInt64]$resolution.width * [UInt64]$resolution.height * 4
            $generatedOutputPath = Join-Path $runRoot 'output-interpolated-1.rgba8'
            if (-not $generated.generated_output_accepted -or
                $generated.output_untouched_pixels -ne 0 -or
                -not (Test-Path -LiteralPath $generatedOutputPath -PathType Leaf) -or
                [UInt64](Get-Item -LiteralPath $generatedOutputPath).Length -ne $expectedBytes) {
                throw "Resolution-shape generated output coverage failed: $runId"
            }

            $phase = '__bootstrap__'
            $shape = @()
            foreach ($line in Get-Content -LiteralPath $tracePath) {
                $event = $line | ConvertFrom-Json
                if ($event.event -eq 'phase_marker') {
                    $phase = [string]$event.phase
                } elseif ($event.event -eq 'kernel_launch') {
                    $shape += [ordered]@{
                        phase = $phase
                        launch_token = [string]$event.launch_token
                        kernel_index = [int]$event.kernel_index
                        module_token = [string]$event.module_token
                        function_token = [string]$event.function_token
                        function_name = [string]$event.function_name
                        grid = @($event.grid | ForEach-Object { [int]$_ })
                        block = @($event.block | ForEach-Object { [int]$_ })
                        dynamic_shared_bytes = [int]$event.dynamic_shared_bytes
                        parameter_bytes = [int]$event.parameter_bytes
                    }
                }
            }
            $resetLaunches = @($shape | Where-Object phase -eq 'evaluate_0').Count
            $generatedLaunches = @($shape | Where-Object phase -eq 'evaluate_1').Count
            if ($resetLaunches -lt 8 -or $generatedLaunches -lt 8) {
                throw "Resolution-shape launch census is implausibly small: $runId"
            }
            if ($null -eq $resolutionResetLaunches) {
                $resolutionResetLaunches = $resetLaunches
                $resolutionGeneratedLaunches = $generatedLaunches
            } elseif ($resetLaunches -ne $resolutionResetLaunches -or
                      $generatedLaunches -ne $resolutionGeneratedLaunches) {
                throw "Resolution-shape launch count is not repeat-exact: $runId"
            }
            $shapePath = Join-Path $runRoot 'selected-shape.json'
            [IO.File]::WriteAllText(
                $shapePath,
                (($shape | ConvertTo-Json -Depth 8 -Compress) + "`n"),
                $utf8)
            $run = [ordered]@{
                run_id = $runId
                resolution = $resolution.name
                width = $resolution.width
                height = $resolution.height
                poison_byte = $poison
                process_ordinal = $processOrdinal
                process_id = $hostReceipt.process_id
                run_nonce = $hostReceipt.run_nonce
                queue_nonce = $hostReceipt.queue_nonce
                reset_launches = $resetLaunches
                generated_launches = $generatedLaunches
                shape_sha256 = (Get-FileHash -LiteralPath $shapePath -Algorithm SHA256).Hash.ToLowerInvariant()
                output_bytes = $expectedBytes
                generated_output_sha256 = $generated.output_interpolated_sha256
                generated_output_untouched_pixels = $generated.output_untouched_pixels
                estimated_vram_bytes = [UInt64]$hostReceipt.estimated_vram_bytes
                gpu_memory_max_observed_bytes = [UInt64]$hostReceipt.gpu_memory_max_observed_bytes
                host_receipt_sha256 = (Get-FileHash -LiteralPath $hostReceiptPath -Algorithm SHA256).Hash.ToLowerInvariant()
                process_receipt_sha256 = (Get-FileHash -LiteralPath $processReceiptPath -Algorithm SHA256).Hash.ToLowerInvariant()
                nvapi_trace_sha256 = (Get-FileHash -LiteralPath $tracePath -Algorithm SHA256).Hash.ToLowerInvariant()
            }
            $runs += $run
            $resolutionRuns += $run
        }
    }
    if (@($resolutionRuns.shape_sha256 | Select-Object -Unique).Count -ne 1) {
        throw "Resolution shape is not repeat-exact: $($resolution.name)"
    }
    if (@($resolutionRuns.generated_output_sha256 | Select-Object -Unique).Count -ne 1) {
        throw "Resolution output is not poison/process invariant: $($resolution.name)"
    }
    if (@($resolutionRuns.process_id | Select-Object -Unique).Count -ne 4 -or
        @($resolutionRuns.run_nonce | Select-Object -Unique).Count -ne 4 -or
        @($resolutionRuns.queue_nonce | Select-Object -Unique).Count -ne 4) {
        throw "Resolution processes are not fresh: $($resolution.name)"
    }
}

$receipt = [ordered]@{
    schema = 'neuralstudio.fg.resolution-shape-matrix.v1'
    job_id = Split-Path -Leaf $JobRoot
    runtime_sha256 = (Get-FileHash -LiteralPath `
        (Join-Path $runner 'nvngx_dlssg.dll') -Algorithm SHA256).Hash.ToLowerInvariant()
    host_binary_sha256 = (Get-FileHash -LiteralPath `
        (Join-Path $runner 'neuralstudio-fg-standalone.exe') -Algorithm SHA256).Hash.ToLowerInvariant()
    proxy_sha256 = (Get-FileHash -LiteralPath `
        (Join-Path $runner 'nvapi64.dll') -Algorithm SHA256).Hash.ToLowerInvariant()
    orchestration_source_sha256 = (Get-FileHash -LiteralPath `
        (Join-Path $sourceRoot 'scripts\run-fg-resolution-shape-matrix.ps1') `
        -Algorithm SHA256).Hash.ToLowerInvariant()
    resolutions = $resolutions
    runs = $runs
    fresh_processes = 16
    poison_values = @(165, 90)
    trace_instrumented_timings_are_performance_evidence = $false
    expected_or_archived_tensor_ingress = $false
    address_fields_recorded = $false
    result = 'pass'
}
$receiptPath = Join-Path $JobRoot 'resolution-shape-matrix-receipt.json'
[IO.File]::WriteAllText(
    $receiptPath,
    (($receipt | ConvertTo-Json -Depth 9) + "`n"),
    $utf8)
Write-Host "receipt=$receiptPath"
Write-Host "receipt_sha256=$((Get-FileHash -LiteralPath $receiptPath -Algorithm SHA256).Hash.ToLowerInvariant())"
Write-Host 'result=pass'
