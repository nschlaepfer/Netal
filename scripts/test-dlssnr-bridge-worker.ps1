<#
Run the explicitly armed, end-to-end Vulkan -> isolated D3D12 DLSSNR -> Vulkan
acceptance gate. This is standalone and never installs into or launches a game.
#>
[CmdletBinding()]
param(
    [switch]$ArmInference,
    [ValidateRange(1, 5)]
    [int]$Runs = 2
)

$ErrorActionPreference = 'Stop'
if (-not $ArmInference) {
    throw 'DLSSNR inference is disarmed. Re-run with -ArmInference to execute the proprietary model.'
}

$repository = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'stage-dlssnr-worker-runtime.ps1')
$runtime = Join-Path $repository 'build\dlssnr_bridge_worker\runtime'
$signedRuntime = @{
    'nvngx_dlssnr.dll' = 'E16BCF15E16E13F527491CDF7845B2FE6521A738D8F7C9C721866A8496E1FC8E'
    'sl.dlss_nr.dll' = '9F6672E5E0170DC118A3188D21BDA187E1FC1AA3502895B21AB846D23165C11D'
}
$localShim = @{
    'sl.interposer.dll' = 'D7138D0D345569CA40FB36FADEAABB2269041BC85049D76D718328431E7EFE05'
    'sl.common.dll' = '5C144677DE65CCD9759292C6F73196E8C19F4BE423698DB2D73705DEED9F73BA'
    'neuralstudio\nvngx.dll' = 'EF0F22BCDF0D8667CA762944B3AC4637B1B969DBF5F47D8671A9B085F064DF40'
}

foreach ($entry in $signedRuntime.GetEnumerator()) {
    $path = Join-Path $runtime $entry.Key
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required signed runtime is missing: $path"
    }
    $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
    if ($hash -ne $entry.Value) {
        throw "Signed runtime hash mismatch for $($entry.Key)."
    }
    $signature = Get-AuthenticodeSignature -LiteralPath $path
    if ($signature.Status -ne 'Valid' -or
        $signature.SignerCertificate.Subject -notmatch 'NVIDIA') {
        throw "Signed runtime validation failed for $($entry.Key)."
    }
}
foreach ($entry in $localShim.GetEnumerator()) {
    $path = Join-Path $runtime $entry.Key
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required source-built shim is missing: $path"
    }
    $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
    if ($hash -ne $entry.Value) {
        throw "Source-built shim hash mismatch for $($entry.Key)."
    }
}

& (Join-Path $PSScriptRoot 'build-dlssnr-bridge-worker.ps1')
& (Join-Path $PSScriptRoot 'build-dlssnr-bridge-worker-probe.ps1')
$worker = Join-Path $repository 'build\dlssnr_bridge_worker\neuralstudio-dlssnr-bridge-worker.exe'
$probe = Join-Path $repository 'build\dlssnr_bridge_worker\neuralstudio-dlssnr-bridge-worker-probe.exe'

$requiredEvidence = @(
    'model.shared_gpu_resources=true',
    'model.slot_count=3',
    'model.vulkan_input_transport=true',
    'model.vulkan_output_transport=true',
    'model.first_frame_reset_enforced=true',
    'model.resize_rebuild=true',
    'model.temporal_reset_after_resize=true',
    'model.crash_isolated=true',
    'model.timeout_bounded=true',
    'model.device_loss_isolated=true',
    'model.resource_generation_rebuilt=true',
    'model.restart_after_crash=true',
    'model.restart_after_timeout=true',
    'model.restart_after_device_loss=true',
    'model.parent_d3d12_debug_available=true',
    'model.parent_d3d12_severe_messages=0',
    'model.worker_d3d12_debug_available=true',
    'model.worker_d3d12_severe_messages=0',
    'model.restart_after_failure=true',
    'model.shutdown_clean=true',
    'model.inference_executed=true',
    'probe=dlssnr_bridge_worker_passed'
)

for ($run = 1; $run -le $Runs; $run++) {
    Write-Host "DLSSNR isolated-worker run $run of $Runs"
    $output = @(& $probe $worker $runtime 2>&1)
    $exitCode = $LASTEXITCODE
    $output | ForEach-Object { Write-Host $_ }
    if ($exitCode -ne 0) {
        throw "DLSSNR isolated-worker run $run failed with exit code $exitCode."
    }
    foreach ($evidence in $requiredEvidence) {
        if ($output -notcontains $evidence) {
            throw "Run $run did not produce required evidence: $evidence"
        }
    }
    $frames = @($output | Where-Object { $_ -match '^model\.frame=' })
    if ($frames.Count -ne 12) {
        throw "Run $run produced $($frames.Count) verified frames; expected 12."
    }
    foreach ($frame in $frames) {
        if ($frame -notmatch 'changed_bytes=([1-9][0-9]*)' -or
            $frame -notmatch 'changed_canary_bytes=([1-9][0-9]*)') {
            throw "Run $run accepted a frame without fresh changed-output evidence: $frame"
        }
    }
    $resetFrames = @($frames | Where-Object { $_ -match ' reset=true ' })
    if ($resetFrames.Count -ne 5) {
        throw "Run $run produced $($resetFrames.Count) temporal reset frames; expected 5."
    }
    if (@($output | Where-Object { $_ -eq 'pipeline_state=bypassed' }).Count -ne 3) {
        throw "Run $run did not fail open for all three injected worker faults."
    }
}

Write-Host 'model.reshade_involved=false'
Write-Host 'model.game_profile_involved=false'
Write-Host "model.repeat_runs=$Runs"
Write-Host 'probe=dlssnr_bridge_worker_repeatability_passed'
