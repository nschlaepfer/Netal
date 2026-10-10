[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $JobRoot,
    [Parameter(Mandatory = $true)] [string] $AttemptId,
    [string] $RepositoryRoot
)

$ErrorActionPreference = 'Stop'
if (-not $RepositoryRoot) {
    $RepositoryRoot = Join-Path $JobRoot 'input\full-source'
}
$attempt = Join-Path $JobRoot (Join-Path 'output' $AttemptId)
if (Test-Path -LiteralPath $attempt) {
    throw "Attempt exists; preserve it: $attempt"
}
$modules = Join-Path $JobRoot 'input\full-modules'
$model = Join-Path $JobRoot 'input\full-model'
$earlyRoot = Join-Path $RepositoryRoot 'src\fg_connected_temporal_l3_l87'
$tailRoot = Join-Path $RepositoryRoot 'src\fg_connected_l20_l87'
$early = Join-Path $earlyRoot 'rtx_temporal.cpp'
$livePrefix = Join-Path $earlyRoot 'rtx_live_prefix_bridge.cpp'
$temporalHandoff = Join-Path $earlyRoot 'rtx_temporal_handoff.cpp'
$tailSource = Join-Path $tailRoot 'rtx_mid_l51_l56_smoke.cpp'
$liveCore = Join-Path $tailRoot 'rtx_live_core_bridge.cpp'
foreach ($required in @($early, $livePrefix, $temporalHandoff,
                         $tailSource, $liveCore,
                         $modules, $model)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required connected source/input missing: $required"
    }
}

$cuda = $env:CUDA_PATH
if (-not $cuda) {
    $cuda = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
}
$vswhere = Join-Path ${env:ProgramFiles(x86)} `
    'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $installation) { throw 'Visual Studio C++ x64 tools unavailable' }
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
$build = Join-Path $attempt 'build'
New-Item -ItemType Directory -Path $build | Out-Null
$generatedTail = Join-Path $build 'rtx_full_tail_live_main.cpp'
$tailText = [IO.File]::ReadAllText($tailSource)
$needle = 'int main(int argc, char** argv) {'
if (($tailText.Split(@($needle), [StringSplitOptions]::None).Count - 1) -ne 1) {
    throw 'Accepted full-tail main signature drifted'
}
$tailText = $tailText.Replace(
    $needle, 'int FgConnectedFullTailMain(int argc, char** argv) {')
$tailText = "#include `"rtx_temporal_handoff.h`"`r`n" + $tailText
$allocationNeedle = @'
    GpuTexture l52_confidence = UploadTexture(
        gpu, DXGI_FORMAT_R16_FLOAT, 320, 180, 2,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        std::vector<std::uint8_t>(kL52ScalarBytes, poison));
'@
$allocationReplacement = @'
    GpuTexture l52_confidence;
    l52_confidence.width = 320;
    l52_confidence.height = 180;
    l52_confidence.bytes_per_pixel = 2;
    l52_confidence.format = DXGI_FORMAT_R16_FLOAT;
    l52_confidence.state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    const auto allocation29_handle = reinterpret_cast<HANDLE>(
        FgConnectedRtxAllocation29Handle());
    if (allocation29_handle == nullptr)
      throw std::runtime_error("live allocation29 binding missing");
    RequireHr(gpu.device->OpenSharedHandle(
                  allocation29_handle,
                  IID_PPV_ARGS(&l52_confidence.resource)),
              "OpenSharedHandle(allocation29)");
    Transition(gpu.list.Get(), l52_confidence.resource.Get(),
               l52_confidence.state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    l52_confidence.state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
'@
if (($tailText.Split(@($allocationNeedle),
                     [StringSplitOptions]::None).Count - 1) -ne 1) {
    throw 'Accepted allocation29 tail anchor drifted'
}
$tailText = $tailText.Replace($allocationNeedle, $allocationReplacement)
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText($generatedTail, $tailText, $utf8NoBom)

$earlyObject = Join-Path $build 'early.obj'
$prefixObject = Join-Path $build 'live-prefix.obj'
$tailObject = Join-Path $build 'full-tail.obj'
$coreObject = Join-Path $build 'live-core.obj'
$handoffObject = Join-Path $build 'temporal-handoff.obj'
$binary = Join-Path $build 'fg-connected-l3-l87.exe'
$compileLog = Join-Path $build 'compile.log'
$command = '"{0}" >nul && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /DFG_CONNECTED_FULL_CHAIN /I"{1}" /I"{13}" /I"{16}" /Fo"{2}" /c "{3}" >"{4}" 2>&1 && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /I"{1}" /I"{13}" /I"{16}" /Fo"{5}" /c "{6}" >>"{4}" 2>&1 && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /I"{1}" /I"{13}" /I"{16}" /Fo"{7}" /c "{8}" >>"{4}" 2>&1 && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /I"{1}" /I"{13}" /I"{16}" /Fo"{9}" /c "{10}" >>"{4}" 2>&1 && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /I"{1}" /I"{13}" /I"{16}" /Fo"{14}" /c "{15}" >>"{4}" 2>&1 && link.exe /nologo /OUT:"{11}" "{2}" "{5}" "{7}" "{9}" "{14}" "{12}" bcrypt.lib d3d12.lib dxgi.lib >>"{4}" 2>&1' -f `
    $vcvars, (Join-Path $cuda 'include'), $earlyObject, $early,
    $compileLog, $prefixObject, $livePrefix, $tailObject, $generatedTail,
    $coreObject, $liveCore, $binary, (Join-Path $cuda 'lib\x64\cuda.lib'),
    $tailRoot, $handoffObject, $temporalHandoff, $earlyRoot
& $env:ComSpec /d /s /c $command
if ($LASTEXITCODE -ne 0) { throw "Build failed; see $compileLog" }

$cases = @(
    [ordered]@{ id = 'connected-temporal-l3-l87-poison-a5-process-a'; poison = 'a5' },
    [ordered]@{ id = 'connected-temporal-l3-l87-poison-a5-process-b'; poison = 'a5' },
    [ordered]@{ id = 'connected-temporal-l3-l87-poison-5a-process-a'; poison = '5a' },
    [ordered]@{ id = 'connected-temporal-l3-l87-poison-5a-process-b'; poison = '5a' }
)
$processes = @()
foreach ($case in $cases) {
    $run = Join-Path $attempt $case.id
    New-Item -ItemType Directory -Path $run | Out-Null
    $process = Start-Process -FilePath $binary `
        -ArgumentList @($modules, $model, $run, $case.id, $case.poison) `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput (Join-Path $run 'stdout.log') `
        -RedirectStandardError (Join-Path $run 'stderr.log')
    $processes += [ordered]@{run_id=$case.id;process_id=$process.Id;exit_code=$process.ExitCode}
    if ($process.ExitCode -ne 0) {
        [ordered]@{schema='neuralstudio.fg.connected-temporal-l3-l87-rig-failure.v1';attempt_id=$AttemptId;failed_run=$case.id;process_id=$process.Id;exit_code=$process.ExitCode;remote_job_retained=$JobRoot;result='fail'} |
            ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $attempt 'rig-failure.json') -Encoding ascii
        throw "Temporal run failed: $($case.id)"
    }
}

$allCheckpoints=@();$finalHashes=@();$runPids=@()
foreach($case in $cases){
    $run=Join-Path $attempt $case.id
    $earlyReceipt=Get-Content -LiteralPath (Join-Path $run 'run-receipt.json') -Raw|ConvertFrom-Json
    $tail1=Get-Content -LiteralPath (Join-Path $run 'generated-1\l20-l87\run-receipt.json') -Raw|ConvertFrom-Json
    $tail2=Get-Content -LiteralPath (Join-Path $run 'generated-2\l20-l87\run-receipt.json') -Raw|ConvertFrom-Json
    $processId=($processes|Where-Object run_id -eq $case.id).process_id
    if(-not($earlyReceipt.process_id -eq $tail1.process_id -and $earlyReceipt.process_id -eq $tail2.process_id -and $earlyReceipt.process_id -eq $processId)){throw "Same-process proof failed: $($case.id)"}
    if($earlyReceipt.result -ne 'pass' -or $tail1.result -ne 'pass' -or $tail2.result -ne 'pass'){throw "Receipt failed: $($case.id)"}
    foreach($key in @('input_mismatched_bytes','canary_mismatched_bytes')){if($earlyReceipt.$key -ne 0){throw "Early integrity: $key"}}
    foreach($tail in @($tail1,$tail2)){foreach($key in @('input_model_mismatched_bytes','canary_mismatched_bytes','readonly_tail_mismatched_bytes')){if($tail.$key -ne 0){throw "Tail integrity: $key"}}}
    if(-not $earlyReceipt.l20_live_handoff -or $earlyReceipt.allocation37_poison_halfwords -ne 0 -or $earlyReceipt.allocation29_poison_halfwords -ne 0){throw 'Temporal coverage/handoff failed'}
    $combined=[ordered]@{}
    foreach($p in $earlyReceipt.checkpoint_sha256.PSObject.Properties){$combined[$p.Name]=$p.Value}
    foreach($p in $tail1.checkpoint_sha256.PSObject.Properties){$combined['generated-1-'+$p.Name]=$p.Value}
    foreach($p in $tail2.checkpoint_sha256.PSObject.Properties){$combined['generated-2-'+$p.Name]=$p.Value}
    if(@($combined.Keys).Count -ne 161){throw 'Temporal checkpoint count failed'}
    $receipt=[ordered]@{schema='neuralstudio.fg.connected-temporal-l3-l87-run.v1';backend='rtx-cuda-external-memory-d3d12-nvapi';run_id=$case.id;process_id=$earlyReceipt.process_id;poison_byte=$earlyReceipt.poison_byte;phase_launch_counts=@(9,76,76);launch_count=161;checkpoint_sha256=$combined;allocation37_sha256=@($earlyReceipt.allocation37_frame1_sha256,$earlyReceipt.allocation37_frame2_sha256);allocation29_sha256=@($earlyReceipt.allocation29_frame1_sha256,$earlyReceipt.allocation29_frame2_sha256);live_device_handoff_l19_l20=$true;host_intermediate_ingress=$false;expected_or_archived_tensor_ingress=$false;address_fields_recorded=$false;final_output_bytes=@($tail1.final_output_bytes,$tail2.final_output_bytes);final_output_sha256=@($tail1.final_output_sha256,$tail2.final_output_sha256);result='pass'}
    [IO.File]::WriteAllText((Join-Path $run 'temporal-receipt.json'),(($receipt|ConvertTo-Json -Depth 8)+"`n"),$utf8NoBom)
    $allCheckpoints+=($combined|ConvertTo-Json -Compress)
    $finalHashes+=(($receipt.final_output_sha256|ConvertTo-Json -Compress))
    $runPids+=$earlyReceipt.process_id
}
if(@($runPids|Sort-Object -Unique).Count -ne 4 -or @($allCheckpoints|Sort-Object -Unique).Count -ne 1 -or @($finalHashes|Sort-Object -Unique).Count -ne 1){throw 'Fresh-process/two-poison invariance failed'}
$first=Get-Content -LiteralPath (Join-Path $attempt ($cases[0].id+'\temporal-receipt.json')) -Raw|ConvertFrom-Json
$receipt=[ordered]@{schema='neuralstudio.fg.connected-temporal-l3-l87-rig.v1';job_id=(Split-Path $JobRoot -Leaf);remote_job_retained=$JobRoot;attempt_id=$AttemptId;gpu=(& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits|Select-Object -First 1).Trim();early_source_sha256=(Get-FileHash $early -Algorithm SHA256).Hash.ToLowerInvariant();live_prefix_source_sha256=(Get-FileHash $livePrefix -Algorithm SHA256).Hash.ToLowerInvariant();temporal_handoff_source_sha256=(Get-FileHash $temporalHandoff -Algorithm SHA256).Hash.ToLowerInvariant();accepted_tail_source_sha256=(Get-FileHash $tailSource -Algorithm SHA256).Hash.ToLowerInvariant();generated_tail_source_sha256=(Get-FileHash $generatedTail -Algorithm SHA256).Hash.ToLowerInvariant();live_core_source_sha256=(Get-FileHash $liveCore -Algorithm SHA256).Hash.ToLowerInvariant();binary_sha256=(Get-FileHash $binary -Algorithm SHA256).Hash.ToLowerInvariant();processes=$processes;fresh_process_count=4;poison_bytes=@(90,165);phase_launch_counts=@(9,76,76);launch_count=161;allocation37_sha256=$first.allocation37_sha256;allocation29_sha256=$first.allocation29_sha256;invariant_output_sha256=$first.final_output_sha256;all_checkpoint_two_poison_invariant=$true;full_write_coverage_from_two_poison_invariance=$true;live_allocation29_temporal_reuse=$true;host_intermediate_ingress=$false;expected_or_archived_tensor_ingress=$false;address_fields_recorded=$false;result='pass'}
[IO.File]::WriteAllText((Join-Path $attempt 'backend-receipt.json'),(($receipt|ConvertTo-Json -Depth 8)+"`n"),$utf8NoBom)
Write-Output "attempt_root=$attempt"
Write-Output "invariant_output_sha256=$($first.final_output_sha256 -join ',')"
Write-Output 'result=pass'
