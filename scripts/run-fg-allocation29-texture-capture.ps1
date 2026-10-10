[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$JobRoot,
    [Parameter(Mandatory = $true)][string]$SeedJobRoot,
    [Parameter(Mandatory = $true)][ValidateSet('prepare', 'run')][string]$Mode
)

$ErrorActionPreference = 'Stop'
$JobRoot = [IO.Path]::GetFullPath($JobRoot).TrimEnd('\')
$SeedJobRoot = [IO.Path]::GetFullPath($SeedJobRoot).TrimEnd('\')
$prefix = [IO.Path]::GetFullPath('D:\CodexLab').TrimEnd('\') + '\'
if (-not $JobRoot.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase) -or
    -not $SeedJobRoot.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Capture and seed jobs must stay beneath D:\CodexLab'
}

if ($Mode -eq 'prepare') {
    foreach ($target in @('input', 'source', 'build\fg_traced_runner')) {
        if (Test-Path -LiteralPath (Join-Path $JobRoot $target)) {
            throw "Prepared target already exists: $target"
        }
    }
    New-Item -ItemType Directory -Force -Path $JobRoot | Out-Null
    Copy-Item -LiteralPath (Join-Path $SeedJobRoot 'input') `
        -Destination (Join-Path $JobRoot 'input') -Recurse
    Copy-Item -LiteralPath (Join-Path $SeedJobRoot 'source') `
        -Destination (Join-Path $JobRoot 'source') -Recurse
    New-Item -ItemType Directory -Force -Path (Join-Path $JobRoot 'build') | Out-Null
    Copy-Item -LiteralPath `
        (Join-Path $SeedJobRoot 'build\fg_traced_runner-v10-default-v2') `
        -Destination (Join-Path $JobRoot 'build\fg_traced_runner') -Recurse
    New-Item -ItemType Directory -Force -Path `
        (Join-Path $JobRoot 'logs'), (Join-Path $JobRoot 'receipts') | Out-Null
    Write-Host "prepared_job=$JobRoot"
    Write-Host 'result=pass'
    exit 0
}

$sourceRoot = Join-Path $JobRoot 'source'
$inputRoot = Join-Path $JobRoot 'input'
$hostBuild = Join-Path $JobRoot 'build\fg_host-texture-v1'
$runner = Join-Path $JobRoot 'build\fg_traced_runner'
$buildScript = Join-Path $sourceRoot 'scripts\build-fg-standalone.ps1'
$runScript = Join-Path $sourceRoot 'scripts\run-fg-traced-job.ps1'
foreach ($required in @(
    $buildScript,
    $runScript,
    (Join-Path $sourceRoot 'src\fg_host\runtime_allocation_capture.cpp'),
    (Join-Path $inputRoot 'sdk\include\nvsdk_ngx.h'),
    (Join-Path $inputRoot 'runtime\nvngx_dlssg.dll'),
    (Join-Path $runner 'nvapi64.dll'),
    (Join-Path $runner 'nvapi64.real.dll')
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Missing allocation29 capture input: $required"
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
foreach ($point in @('post_create', 'post_evaluate_0')) {
    foreach ($poison in @(165, 90)) {
        $slug = if ($point -eq 'post_create') { 'post-create' } else { 'post-evaluate-0' }
        $runId = "allocation29-$slug-poison-$('{0:x2}' -f $poison)"
        & $runScript -JobRoot $JobRoot -RunnerDirectory $runner `
            -RunId $runId -OutputPoisonByte $poison `
            -DisableInterpolationOutput -CaptureRuntimeAllocations `
            -ExpectedBaselineOutputSha256 $baseline `
            -CaptureRuntimeDefaultBuffers -RuntimeDefaultBufferTokens '29' `
            -RuntimeDefaultBufferCapturePoint $point `
            -ExpectedRuntimeDefaultBufferCount 1 `
            -ExpectedRuntimeDefaultBufferBytes 115200
        $dump = Join-Path $JobRoot `
            "output\$runId\runtime-default-buffers-$slug\default-token-000029.bin"
        $manifest = Join-Path $JobRoot `
            "output\$runId\runtime-default-buffers-$slug.manifest.json"
        if ((Get-Item -LiteralPath $dump).Length -ne 115200) {
            throw "allocation29 dump size drift: $runId"
        }
        $runs += [ordered]@{
            run_id = $runId
            capture_point = $point
            poison_byte = $poison
            dump_bytes = 115200
            dump_sha256 = (Get-FileHash -LiteralPath $dump -Algorithm SHA256).Hash.ToLowerInvariant()
            manifest_sha256 = (Get-FileHash -LiteralPath $manifest -Algorithm SHA256).Hash.ToLowerInvariant()
            process_receipt_sha256 = (Get-FileHash -LiteralPath `
                (Join-Path $JobRoot "receipts\$runId-process.json") `
                -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
}
$receipt = [ordered]@{
    schema = 'neuralstudio.fg.allocation29-texture-capture.v1'
    job_id = Split-Path -Leaf $JobRoot
    seed_job_id = Split-Path -Leaf $SeedJobRoot
    runtime_sha256 = (Get-FileHash -LiteralPath `
        (Join-Path $inputRoot 'runtime\nvngx_dlssg.dll') `
        -Algorithm SHA256).Hash.ToLowerInvariant()
    host_source_sha256 = (Get-FileHash -LiteralPath `
        (Join-Path $sourceRoot 'src\fg_host\main.cpp') `
        -Algorithm SHA256).Hash.ToLowerInvariant()
    allocation_capture_source_sha256 = (Get-FileHash -LiteralPath `
        (Join-Path $sourceRoot 'src\fg_host\runtime_allocation_capture.cpp') `
        -Algorithm SHA256).Hash.ToLowerInvariant()
    host_binary_sha256 = (Get-FileHash -LiteralPath `
        (Join-Path $hostBuild 'neuralstudio-fg-standalone.exe') `
        -Algorithm SHA256).Hash.ToLowerInvariant()
    runs = $runs
    expected_frame_bytes_fed_to_capture = $false
    captured_bytes_fed_back_to_execution = $false
    address_fields_recorded = $false
    result = 'pass'
}
$utf8 = New-Object System.Text.UTF8Encoding($false)
$receiptPath = Join-Path $JobRoot 'allocation29-capture-receipt.json'
[IO.File]::WriteAllText($receiptPath, (($receipt | ConvertTo-Json -Depth 7) + "`n"), $utf8)
Write-Host "receipt=$receiptPath"
Write-Host "receipt_sha256=$((Get-FileHash -LiteralPath $receiptPath -Algorithm SHA256).Hash.ToLowerInvariant())"
Write-Host 'result=pass'
