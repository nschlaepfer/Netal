[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $JobRoot,
    [Parameter(Mandatory = $true)] [string] $AttemptId
)

$ErrorActionPreference = 'Stop'
$attempt = Join-Path $JobRoot (Join-Path 'output\full' $AttemptId)
$build = Join-Path $attempt 'build'
if (-not (Test-Path -LiteralPath $build -PathType Container)) {
    throw "Full-chain attempt is missing: $attempt"
}
$runs = @(Get-ChildItem -LiteralPath $attempt -Directory |
    Where-Object Name -Like 'subrect-full-poison-*-process-*' |
    Sort-Object Name)
if ($runs.Count -ne 4) { throw 'Exactly four preserved runs are required' }
$processes = @()
$checkpointMaps = @()
foreach ($run in $runs) {
    $early = Get-Content -LiteralPath `
        (Join-Path $run.FullName 'early-run-receipt.json') -Raw |
        ConvertFrom-Json
    $tail = Get-Content -LiteralPath `
        (Join-Path $run.FullName 'l18-l83\run-receipt.json') -Raw |
        ConvertFrom-Json
    $combined = Get-Content -LiteralPath `
        (Join-Path $run.FullName 'combined-run-receipt.json') -Raw |
        ConvertFrom-Json
    if ($early.result -ne 'pass' -or $tail.result -ne 'pass' -or
        $combined.result -ne 'pass' -or
        $early.process_id -ne $tail.process_id -or
        $early.process_id -ne $combined.process_id) {
        throw "Nested receipt failed: $($run.Name)"
    }
    if ($combined.launch_range[0] -ne 3 -or
        $combined.launch_range[1] -ne 83 -or
        $combined.launch_count -ne 81 -or
        @($combined.checkpoint_sha256.PSObject.Properties).Count -ne 81) {
        throw "Launch coverage failed: $($run.Name)"
    }
    foreach ($name in @('input_mismatched_bytes','canary_mismatched_bytes',
        'outside_region_mismatched_bytes','tensor_padding_nonzero_bytes',
        'auxiliary_untouched_row_mismatched_bytes')) {
        if ($early.$name -ne 0) { throw "Early integrity failed: $name" }
    }
    foreach ($name in @('input_model_mismatched_bytes',
        'canary_mismatched_bytes','readonly_tail_mismatched_bytes')) {
        if ($tail.$name -ne 0) { throw "Tail integrity failed: $name" }
    }
    $checkpointMaps +=
        ($combined.checkpoint_sha256 | ConvertTo-Json -Compress -Depth 5)
    $processes += [pscustomobject][ordered]@{
        run_id = $combined.run_id
        process_id = $combined.process_id
        poison_byte = $combined.poison_byte
        allocation37_sha256 = $combined.allocation37_sha256
        output_full_sha256 = $combined.output_full_sha256
        output_region_sha256 = $combined.output_region_sha256
        checkpoint_sha256 = $combined.checkpoint_sha256
    }
}
if (@($processes.process_id | Sort-Object -Unique).Count -ne 4 -or
    @($processes.poison_byte | Sort-Object).Count -ne 4 -or
    @($checkpointMaps | Sort-Object -Unique).Count -ne 1 -or
    @($processes.allocation37_sha256 | Sort-Object -Unique).Count -ne 1 -or
    @($processes.output_region_sha256 | Sort-Object -Unique).Count -ne 1) {
    throw 'Fresh-process/two-poison invariant gate failed'
}
if ((@($processes.poison_byte | Where-Object { $_ -eq 90 })).Count -ne 2 -or
    (@($processes.poison_byte | Where-Object { $_ -eq 165 })).Count -ne 2) {
    throw 'A5/5A poison coverage failed'
}
$a5 = @($processes | Where-Object { $_.poison_byte -eq 165 } |
    ForEach-Object { $_.output_full_sha256 } | Sort-Object -Unique)
$fivea = @($processes | Where-Object { $_.poison_byte -eq 90 } |
    ForEach-Object { $_.output_full_sha256 } | Sort-Object -Unique)
if ($a5.Count -ne 1 -or $fivea.Count -ne 1 -or $a5[0] -eq $fivea[0]) {
    throw 'Outside-poison counterexample gate failed'
}
$generated = Join-Path $build 'generated'
$backend = [ordered]@{
    schema = 'neuralstudio.fg.connected-subrect-l3-l83-rtx-backend.v1'
    job_id = Split-Path $JobRoot -Leaf
    attempt_id = $AttemptId
    remote_job_retained = $JobRoot
    gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total `
        --format=csv,noheader,nounits | Select-Object -First 1).Trim()
    generated_source_sha256 = [ordered]@{
        early = (Get-FileHash (Join-Path $generated `
            'rtx_subrect_early_live.cpp') -Algorithm SHA256).Hash.ToLowerInvariant()
        prefix = (Get-FileHash (Join-Path $generated `
            'rtx_subrect_prefix_live.cpp') -Algorithm SHA256).Hash.ToLowerInvariant()
        core = (Get-FileHash (Join-Path $generated `
            'rtx_subrect_core.cpp') -Algorithm SHA256).Hash.ToLowerInvariant()
        tail = (Get-FileHash (Join-Path $generated `
            'rtx_subrect_tail_live.cpp') -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    binary_sha256 = (Get-FileHash (Join-Path $build 'subrect-full.exe') `
        -Algorithm SHA256).Hash.ToLowerInvariant()
    compile_log_sha256 = (Get-FileHash (Join-Path $build 'compile.log') `
        -Algorithm SHA256).Hash.ToLowerInvariant()
    fresh_process_count = 4
    poison_bytes = @(90,165)
    processes = @($processes | ForEach-Object {
        [ordered]@{run_id=$_.run_id;process_id=$_.process_id;
          poison_byte=$_.poison_byte;output_full_sha256=$_.output_full_sha256}
    })
    launch_range = @(3,83)
    launch_count = 81
    checkpoint_sha256 = $processes[0].checkpoint_sha256
    allocation37_sha256 = $processes[0].allocation37_sha256
    output_region_sha256 = $processes[0].output_region_sha256
    poison_specific_full_sha256 = [ordered]@{a5=$a5[0];fivea=$fivea[0]}
    full_surface_poison_invariant = $false
    region_scoped_poison_invariant = $true
    live_device_handoff_count = 80
    live_allocation37_handoff = $true
    input_model_immutability = $true
    canary_integrity = $true
    expected_or_archived_intermediate_ingress = $false
    address_fields_recorded = $false
    result = 'pass'
}
$utf8NoBom = New-Object Text.UTF8Encoding($false)
[IO.File]::WriteAllText((Join-Path $attempt 'backend-receipt.json'),
    (($backend | ConvertTo-Json -Depth 8) + "`n"),$utf8NoBom)
Write-Output "attempt_root=$attempt"
Write-Output "allocation37_sha256=$($backend.allocation37_sha256)"
Write-Output "output_region_sha256=$($backend.output_region_sha256)"
Write-Output 'result=pass'
