[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$JobRoot)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$JobRoot = [IO.Path]::GetFullPath($JobRoot).TrimEnd('\')
if (-not $JobRoot.StartsWith('D:\CodexLab\nr-first-inference-')) {
    throw 'NR job must be isolated below D:\CodexLab'
}
$seed = 'D:\NeuralStudio-RTX-Oracle\models\nvidia\nvngx_dlssnr.dll'
$runtime = Join-Path $JobRoot 'input\nvngx_dlssnr.dll'
$expectedRuntime = 'e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e'
if ((Get-FileHash -LiteralPath $seed -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedRuntime) {
    throw 'Seed runtime identity drift'
}
if (Test-Path -LiteralPath $runtime) { throw 'Preserve previous staged runtime' }
Copy-Item -LiteralPath $seed -Destination $runtime
$pins = @{
    'dlssnr-cuda-create-trace.exe'='46b2a5cd51d49f189d050e48d68c86e3301a2b75669a0240c711795625cff616'
    'nvcuda.dll'='445df9c7b2b58eb65163d4470262fef9143044b5819cc095f6a761245ee22554'
    'nvngx.dll'='d8bf063f27df70e304062ebd78ac03990b323c76725a80726770584ea5ca940c'
}
foreach ($name in $pins.Keys) {
    if ((Get-FileHash -LiteralPath (Join-Path $JobRoot "input\$name") -Algorithm SHA256).Hash.ToLowerInvariant() -ne $pins[$name]) {
        throw "Input identity drift: $name"
    }
}
$started = (Get-Date).ToUniversalTime().ToString('o')
$results = @()
foreach ($run in @('process-a','process-b')) {
    $out = Join-Path $JobRoot $run
    $args = @('-NoLogo','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass',
        '-File',(Join-Path $JobRoot 'source\capture-rtx-feature18-cuda-oracle.ps1'),
        '-ArmRealExecution','-ArtifactDirectory',$out,'-RuntimeBinary',$runtime,
        '-HarnessBinary',(Join-Path $JobRoot 'input\dlssnr-cuda-create-trace.exe'),
        '-ProxyBinary',(Join-Path $JobRoot 'input\nvcuda.dll'),
        '-ThunkBinary',(Join-Path $JobRoot 'input\nvngx.dll'))
    & powershell.exe @args *> (Join-Path $JobRoot "$run.log")
    if ($LASTEXITCODE -ne 0) { throw "Fresh NR oracle failed: $run; inspect preserved log" }
    $r = Get-Content -LiteralPath (Join-Path $out 'capture-receipt.json') -Raw | ConvertFrom-Json
    if (-not $r.rtx_inference_proven -or $r.output_nonfinite_elements -ne 0 -or $r.output_rgb_nonzero_elements -le 0) {
        throw "Invalid RGB inference: $run"
    }
    $results += [ordered]@{run=$run;output_sha256=$r.output_sha256;counts=$r.counts;
        output_rgb_nonzero_elements=$r.output_rgb_nonzero_elements;
        output_nonfinite_elements=$r.output_nonfinite_elements;
        capture_receipt_sha256=(Get-FileHash -LiteralPath (Join-Path $out 'capture-receipt.json') -Algorithm SHA256).Hash.ToLowerInvariant()}
}
if ($results[0].output_sha256 -ne $results[1].output_sha256) { throw 'Fresh-process output differs' }
$receipt = [ordered]@{
    schema='neuralstudio.nr.first-inference-rtx-refresh.v1';
    started_utc=$started;ended_utc=(Get-Date).ToUniversalTime().ToString('o');
    gpu=(& nvidia-smi --query-gpu=name --format=csv,noheader);
    driver=(& nvidia-smi --query-gpu=driver_version --format=csv,noheader);
    memory_after=(& nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader);
    runtime_sha256=$expectedRuntime;input_binary_sha256=$pins;runs=$results;
    repeat_equal=$true;fresh_process_count=2;poison_invariance_proven=$false;
    snapshot_parity_proven=$false;mac_inference_proven=$false;result='pass'
}
$receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $JobRoot 'refresh-receipt.json') -Encoding UTF8
$receipt | ConvertTo-Json -Depth 8
