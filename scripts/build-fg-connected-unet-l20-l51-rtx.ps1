[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$JobRoot,
    [string]$AttemptId,
    [string]$RepositoryRoot,
    [string]$BaseCommit = '03063c1'
)

$ErrorActionPreference = 'Stop'
$started = (Get-Date).ToUniversalTime().ToString('o')
if (-not $RepositoryRoot) { $RepositoryRoot = Split-Path -Parent $PSScriptRoot }
if (-not $AttemptId) { $AttemptId = 'attempt-' + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ') }
$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
$JobRoot = [IO.Path]::GetFullPath($JobRoot)
$source = Join-Path $RepositoryRoot 'src\fg_connected_unet_l20_l51\rtx_oracle.cpp'
$verifier = Join-Path $RepositoryRoot 'scripts\verify-fg-connected-unet-l20-l51.py'
$harness = Join-Path $RepositoryRoot 'scripts\build-fg-connected-unet-l20-l51-rtx.ps1'
$moduleRoot = Join-Path $JobRoot 'inputs\modules'
$modelRoot = Join-Path $JobRoot 'inputs\model'
$attemptRoot = Join-Path $JobRoot (Join-Path 'output' $AttemptId)
$buildRoot = Join-Path $attemptRoot 'build'

foreach ($required in @($source, $verifier, $harness)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Missing U-Net harness input: $required" }
}
if (Test-Path -LiteralPath $attemptRoot) { throw "Attempt exists; preserve it: $attemptRoot" }
$moduleFiles = @(Get-ChildItem -LiteralPath $moduleRoot -File -Filter 'cu-module-*.bin' | Sort-Object Name)
$modelFiles = @(Get-ChildItem -LiteralPath $modelRoot -File -Filter 'default-token-*.bin' | Sort-Object Name)
if ($moduleFiles.Count -ne 25) { throw "Expected 25 selected module files, found $($moduleFiles.Count)" }
if ($modelFiles.Count -ne 17) { throw "Expected 17 selected model-state files, found $($modelFiles.Count)" }

$cudaPath = $env:CUDA_PATH
if (-not $cudaPath) { $cudaPath = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3' }
$cudaInclude = Join-Path $cudaPath 'include'
$cudaLibrary = Join-Path $cudaPath 'lib\x64\cuda.lib'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $installation) { throw 'Visual Studio C++ x64 build tools were not found.' }
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
New-Item -ItemType Directory -Path $buildRoot | Out-Null
$binary = Join-Path $buildRoot 'fg-connected-unet-l20-l51-rtx.exe'
$object = Join-Path $buildRoot 'fg-connected-unet-l20-l51-rtx.obj'
$compileLog = Join-Path $buildRoot 'compile.log'
$common = '/nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8'
$compileCommand = '"{0}" >nul && cl.exe {1} /I"{2}" /Fo"{3}" /c "{4}" >"{5}" 2>&1 && link.exe /nologo /OUT:"{6}" "{3}" "{7}" bcrypt.lib >>"{5}" 2>&1' -f $vcvars, $common, $cudaInclude, $object, $source, $compileLog, $binary, $cudaLibrary
& $env:ComSpec /d /s /c $compileCommand
if ($LASTEXITCODE -ne 0) { throw "Connected U-Net build failed; see $compileLog" }

$runReceipts = @()
foreach ($poison in @('a5', '5a')) {
    foreach ($ordinal in @('a', 'b')) {
        $runId = "unet-poison-$poison-process-$ordinal"
        $runRoot = Join-Path $attemptRoot $runId
        New-Item -ItemType Directory -Path $runRoot | Out-Null
        $process = Start-Process -FilePath $binary -ArgumentList @($moduleRoot, $modelRoot, $runRoot, $runId, $poison, '0') -NoNewWindow -Wait -PassThru -RedirectStandardOutput (Join-Path $runRoot 'stdout.log') -RedirectStandardError (Join-Path $runRoot 'stderr.log')
        if ($process.ExitCode -ne 0) { throw "Connected U-Net run $runId failed with exit code $($process.ExitCode)" }
        $receipt = Get-Content -LiteralPath (Join-Path $runRoot 'run-receipt.json') -Raw | ConvertFrom-Json
        if ($receipt.result -ne 'pass') { throw "Connected U-Net receipt failed: $runId" }
        $runReceipts += $receipt
    }
}
$processIds = @($runReceipts | ForEach-Object { [int64]$_.process_id } | Sort-Object -Unique)
if ($processIds.Count -ne 4) { throw "Fresh-process gate failed: $($processIds.Count)" }
for ($tensor = 0; $tensor -lt 32; ++$tensor) {
    $hashes = @($runReceipts | ForEach-Object { $_.tensors[$tensor].sha256 } | Sort-Object -Unique)
    if ($hashes.Count -ne 1) { throw "U-Net tensor invariance failed at index $tensor" }
}
$backendReceipt = Join-Path $attemptRoot 'backend-receipt.json'
& python $verifier backend --backend rtx --root $attemptRoot --output $backendReceipt
if ($LASTEXITCODE -ne 0) { throw "Connected U-Net backend verification failed: $LASTEXITCODE" }

$inputRecords = @()
foreach ($file in @($moduleFiles + $modelFiles)) {
    $kind = if ($file.DirectoryName -eq $moduleRoot) { 'selected-module' } else { 'post-evaluate-0-model-state' }
    $inputRecords += [ordered]@{
        relative_path = $file.FullName.Substring($JobRoot.Length + 1).Replace('\', '/')
        bytes = $file.Length
        sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        kind = $kind
    }
}
$gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total,pci.bus_id --format=csv,noheader,nounits | Select-Object -First 1).Trim()
$cudaVersion = (& (Join-Path $cudaPath 'bin\nvcc.exe') --version | Select-Object -Last 1).Trim()
$rigReceipt = [ordered]@{
    schema = 'neuralstudio.fg.connected-unet-l20-l51-rig.v1'
    job_id = Split-Path -Leaf $JobRoot
    attempt_id = $AttemptId
    local_base_commit = $BaseCommit
    started_utc = $started
    ended_utc = (Get-Date).ToUniversalTime().ToString('o')
    gpu = $gpu
    cuda_toolkit = $cudaVersion
    visual_studio = $installation
    inputs = $inputRecords
    source = [ordered]@{ relative_path = 'src/fg_connected_unet_l20_l51/rtx_oracle.cpp'; bytes = (Get-Item -LiteralPath $source).Length; sha256 = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant() }
    harness = @(
        [ordered]@{ relative_path = 'scripts/build-fg-connected-unet-l20-l51-rtx.ps1'; bytes = (Get-Item -LiteralPath $harness).Length; sha256 = (Get-FileHash -LiteralPath $harness -Algorithm SHA256).Hash.ToLowerInvariant() },
        [ordered]@{ relative_path = 'scripts/verify-fg-connected-unet-l20-l51.py'; bytes = (Get-Item -LiteralPath $verifier).Length; sha256 = (Get-FileHash -LiteralPath $verifier -Algorithm SHA256).Hash.ToLowerInvariant() }
    )
    binary = [ordered]@{ relative_path = "output/$AttemptId/build/fg-connected-unet-l20-l51-rtx.exe"; bytes = (Get-Item -LiteralPath $binary).Length; sha256 = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant() }
    selected_launch_first = 20
    selected_launch_last = 51
    selected_launch_count = 32
    run_count = $runReceipts.Count
    fresh_process_count = $processIds.Count
    final_output_sha256 = $runReceipts[0].tensors[31].sha256
    expected_output_fed_to_execution = $false
    archived_intermediate_fed_to_execution = $false
    address_fields_recorded = $false
    result = 'pass'
}
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$rigReceiptPath = Join-Path $attemptRoot 'rig-receipt.json'
[IO.File]::WriteAllText($rigReceiptPath, (($rigReceipt | ConvertTo-Json -Depth 8) + "`n"), $utf8NoBom)
$rigHash = (Get-FileHash -LiteralPath $rigReceiptPath -Algorithm SHA256).Hash.ToLowerInvariant()
$backendHash = (Get-FileHash -LiteralPath $backendReceipt -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText((Join-Path $attemptRoot 'rig-receipt.sha256'), ($rigHash + "  rig-receipt.json`n"), $utf8NoBom)
[IO.File]::WriteAllText((Join-Path $attemptRoot 'backend-receipt.sha256'), ($backendHash + "  backend-receipt.json`n"), $utf8NoBom)
Write-Host "attempt_root=$attemptRoot"
Write-Host "rig_receipt_sha256=$rigHash"
Write-Host "backend_receipt_sha256=$backendHash"
Write-Host 'result=pass'
