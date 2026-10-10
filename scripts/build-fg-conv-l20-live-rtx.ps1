[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string]$JobRoot,
    [Parameter(Mandatory = $true)] [string]$AttemptId,
    [string]$RepositoryRoot
)

$ErrorActionPreference = 'Stop'
if (-not $RepositoryRoot) { $RepositoryRoot = Split-Path -Parent $PSScriptRoot }
$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
$JobRoot = [IO.Path]::GetFullPath($JobRoot)
$source = Join-Path $RepositoryRoot 'src\fg_conv\rtx_live_model_oracle.cpp'
$baseSource = Join-Path $RepositoryRoot 'src\fg_conv\rtx_oracle.cpp'
$module = Join-Path $JobRoot 'inputs\cu-module-000028.bin'
$weights = Join-Path $JobRoot 'inputs\default-token-000063-post-eval0.bin'
$manifest = Join-Path $JobRoot 'inputs\runtime-default-buffers-post-evaluate-0.manifest.json'
$attemptRoot = Join-Path $JobRoot (Join-Path 'output' $AttemptId)
if (Test-Path -LiteralPath $attemptRoot) { throw "Attempt exists: $attemptRoot" }
foreach ($required in @($source, $baseSource, $module, $weights, $manifest)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Missing input: $required" }
}
if ((Get-FileHash $module -Algorithm SHA256).Hash -ne '7F9C1F110E8EC9347AA968BC9019E5FFC8839057987A2E5676A0F64637C8DF2D') { throw 'Module hash mismatch' }
if ((Get-FileHash $weights -Algorithm SHA256).Hash -ne '44595B02307A6B934A124ABE476CFD327440E2ACBA7F5ED69F73195CA743E3D9') { throw 'Weight hash mismatch' }
if ((Get-FileHash $manifest -Algorithm SHA256).Hash -ne '2A1179D4A25BE030F772CD90CA7D017ED6EAC9291917F7D43EA9B0CD5AC6A294') { throw 'Manifest hash mismatch' }

$cudaPath = $env:CUDA_PATH
if (-not $cudaPath) { $cudaPath = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3' }
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
$buildRoot = Join-Path $attemptRoot 'build'
New-Item -ItemType Directory -Path $buildRoot | Out-Null
$binary = Join-Path $buildRoot 'fg-conv-l20-live-rtx.exe'
$object = Join-Path $buildRoot 'fg-conv-l20-live-rtx.obj'
$compileLog = Join-Path $buildRoot 'compile.log'
$compileCommand = '"{0}" >nul && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /I"{1}" /Fo"{2}" /c "{3}" >"{4}" 2>&1 && link.exe /nologo /OUT:"{5}" "{2}" "{6}" bcrypt.lib >>"{4}" 2>&1' -f $vcvars, (Join-Path $cudaPath 'include'), $object, $source, $compileLog, $binary, (Join-Path $cudaPath 'lib\x64\cuda.lib')
& $env:ComSpec /d /s /c $compileCommand
if ($LASTEXITCODE -ne 0) { throw "Build failed; see $compileLog" }

$receipts = @()
foreach ($poison in @('a5', '5a')) {
    foreach ($ordinal in @('a', 'b')) {
        $runId = "live-model-poison-$poison-process-$ordinal"
        $runRoot = Join-Path $attemptRoot $runId
        New-Item -ItemType Directory -Path $runRoot | Out-Null
        $process = Start-Process -FilePath $binary -ArgumentList @($module, $weights, $runRoot, $runId, $poison) -NoNewWindow -Wait -PassThru -RedirectStandardOutput (Join-Path $runRoot 'stdout.log') -RedirectStandardError (Join-Path $runRoot 'stderr.log')
        if ($process.ExitCode -ne 0) { throw "Run failed: $runId" }
        $receipts += Get-Content (Join-Path $runRoot 'run-receipt.json') -Raw | ConvertFrom-Json
    }
}
$hashes = @($receipts | ForEach-Object { $_.output_sha256 } | Sort-Object -Unique)
$pids = @($receipts | ForEach-Object { $_.process_id } | Sort-Object -Unique)
if ($hashes.Count -ne 1 -or $pids.Count -ne 4) { throw 'Repeat/poison gate failed' }
$receipt = [ordered]@{
    schema = 'neuralstudio.fg.conv-l20-live-rig.v1'
    job_id = Split-Path -Leaf $JobRoot
    attempt_id = $AttemptId
    gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits | Select-Object -First 1).Trim()
    cuda_toolkit = (& (Join-Path $cudaPath 'bin\nvcc.exe') --version | Select-Object -Last 1).Trim()
    module_sha256 = (Get-FileHash $module -Algorithm SHA256).Hash.ToLowerInvariant()
    weight_source_token = 63
    weight_sha256 = (Get-FileHash $weights -Algorithm SHA256).Hash.ToLowerInvariant()
    weight_capture_manifest_sha256 = (Get-FileHash $manifest -Algorithm SHA256).Hash.ToLowerInvariant()
    weight_capture_point = 'post_evaluate_0_fence_pre_evaluate_1'
    source_sha256 = (Get-FileHash $source -Algorithm SHA256).Hash.ToLowerInvariant()
    base_source_sha256 = (Get-FileHash $baseSource -Algorithm SHA256).Hash.ToLowerInvariant()
    binary_sha256 = (Get-FileHash $binary -Algorithm SHA256).Hash.ToLowerInvariant()
    fresh_process_count = $pids.Count
    two_poison_output_sha256 = $hashes[0]
    full_write_coverage_from_two_poison_invariance = $true
    per_run_poison_collision_is_not_authoritative = $true
    address_fields_recorded = $false
    result = 'pass'
}
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$receiptPath = Join-Path $attemptRoot 'rig-receipt.json'
[IO.File]::WriteAllText($receiptPath, (($receipt | ConvertTo-Json -Depth 6) + "`n"), $utf8NoBom)
$receiptHash = (Get-FileHash $receiptPath -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText((Join-Path $attemptRoot 'rig-receipt.sha256'), ($receiptHash + "  rig-receipt.json`n"), $utf8NoBom)
Write-Host "attempt_root=$attemptRoot"
Write-Host "output_sha256=$($hashes[0])"
Write-Host "rig_receipt_sha256=$receiptHash"
Write-Host 'result=pass'
