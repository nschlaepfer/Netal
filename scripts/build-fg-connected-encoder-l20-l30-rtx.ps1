[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$JobRoot,
    [string]$AttemptId,
    [string]$RepositoryRoot,
    [string]$BaseCommit = 'cfcc660fff97039fcfabd2229fe7e8d20e1f1ffe'
)

$ErrorActionPreference = 'Stop'
$started = (Get-Date).ToUniversalTime().ToString('o')
if (-not $RepositoryRoot) { $RepositoryRoot = Split-Path -Parent $PSScriptRoot }
if (-not $AttemptId) { $AttemptId = 'attempt-' + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ') }
$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
$JobRoot = [IO.Path]::GetFullPath($JobRoot)
$source = Join-Path $RepositoryRoot 'src\fg_connected_encoder_l20_l30\rtx_oracle.cpp'
$verifier = Join-Path $RepositoryRoot 'scripts\verify-fg-connected-encoder-l20-l30.py'
$harness = Join-Path $RepositoryRoot 'scripts\build-fg-connected-encoder-l20-l30-rtx.ps1'
$moduleRoot = Join-Path $JobRoot 'inputs\modules'
$modelRoot = Join-Path $JobRoot 'inputs\model'
$attemptRoot = Join-Path $JobRoot (Join-Path 'output' $AttemptId)
$buildRoot = Join-Path $attemptRoot 'build'

$modules = @(
    [ordered]@{ token = 28; bytes = 16512; sha256 = '7F9C1F110E8EC9347AA968BC9019E5FFC8839057987A2E5676A0F64637C8DF2D' },
    [ordered]@{ token = 29; bytes = 2032; sha256 = 'A29A3D7041CBE7E44707E74DDDF7AFF61871A2397D1C097EF913BD753A149977' },
    [ordered]@{ token = 30; bytes = 4952; sha256 = '31BB08D04011DB9A0DEF8F1A1BC3999EFC08D5DEB8D7CB9B5DEBF52EC489B785' },
    [ordered]@{ token = 31; bytes = 11168; sha256 = 'E925F51165408FAE4657D50247B10304802D46979B863712AEFF844A217ADE63' },
    [ordered]@{ token = 32; bytes = 11416; sha256 = 'C74961F6F71F4E7BAC5B216CC704B10A0FDD6F0987352FDD315F371467CA7187' },
    [ordered]@{ token = 33; bytes = 14752; sha256 = '6F2FFBE7FAACDE8119719D92DACDE40F21DDD562878381B81654DAD7A66775DD' },
    [ordered]@{ token = 34; bytes = 11328; sha256 = '2F490857C6F7AA22F2435057706EC10B28B6B8D3081BD967EBE1D61550AC6419' }
)
$weights = @(
    [ordered]@{ token = 63; bytes = 56320; sha256 = '44595B02307A6B934A124ABE476CFD327440E2ACBA7F5ED69F73195CA743E3D9' },
    [ordered]@{ token = 65; bytes = 10240; sha256 = 'EB99328524FDDC9F5F3FAC12B8E3A4011E66A75E071CD07126A70C553458BC60' },
    [ordered]@{ token = 67; bytes = 299008; sha256 = '4191924FF335D5018987F2526F933CB4CDBB9145411E3CE34F63C8206321BE55' },
    [ordered]@{ token = 69; bytes = 593920; sha256 = 'C0C5648F9CB839E41D0E786C8034AA0BCC931B6AC84C58CFBDE0502DC8A53404' },
    [ordered]@{ token = 71; bytes = 1187840; sha256 = 'FE2C3057F2E894219601D3D7200EABFA97D990E08CCB23B43D52DFCBA2D2FB93' },
    [ordered]@{ token = 73; bytes = 540672; sha256 = 'E23316A86707C1D3D377AA0711DF47D05976AD1A30F6428524D030DD17145DE9' }
)

foreach ($required in @($source, $verifier, $harness)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Missing harness input: $required" }
}
if (Test-Path -LiteralPath $attemptRoot) { throw "Attempt exists; preserve it: $attemptRoot" }
$inputRecords = @()
foreach ($item in $modules) {
    $name = 'cu-module-{0:d6}.bin' -f $item.token
    $path = Join-Path $moduleRoot $name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing module: $path" }
    $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
    if ((Get-Item -LiteralPath $path).Length -ne $item.bytes -or $hash -ne $item.sha256) { throw "Module identity mismatch: $name" }
    $inputRecords += [ordered]@{ relative_path = "inputs/modules/$name"; bytes = $item.bytes; sha256 = $hash.ToLowerInvariant(); kind = 'selected-module' }
}
foreach ($item in $weights) {
    $name = 'default-token-{0:d6}.bin' -f $item.token
    $path = Join-Path $modelRoot $name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing model state: $path" }
    $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
    if ((Get-Item -LiteralPath $path).Length -ne $item.bytes -or $hash -ne $item.sha256) { throw "Model-state identity mismatch: $name" }
    $inputRecords += [ordered]@{ relative_path = "inputs/model/$name"; bytes = $item.bytes; sha256 = $hash.ToLowerInvariant(); kind = 'post-evaluate-0-model-state' }
}

$cudaPath = $env:CUDA_PATH
if (-not $cudaPath) { $cudaPath = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3' }
$cudaInclude = Join-Path $cudaPath 'include'
$cudaLibrary = Join-Path $cudaPath 'lib\x64\cuda.lib'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $installation) { throw 'Visual Studio C++ x64 build tools were not found.' }
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
New-Item -ItemType Directory -Path $buildRoot | Out-Null
$binary = Join-Path $buildRoot 'fg-connected-encoder-l20-l30-rtx.exe'
$object = Join-Path $buildRoot 'fg-connected-encoder-l20-l30-rtx.obj'
$compileLog = Join-Path $buildRoot 'compile.log'
$common = '/nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8'
$compileCommand = '"{0}" >nul && cl.exe {1} /I"{2}" /Fo"{3}" /c "{4}" >"{5}" 2>&1 && link.exe /nologo /OUT:"{6}" "{3}" "{7}" bcrypt.lib >>"{5}" 2>&1' -f $vcvars, $common, $cudaInclude, $object, $source, $compileLog, $binary, $cudaLibrary
& $env:ComSpec /d /s /c $compileCommand
if ($LASTEXITCODE -ne 0) { throw "Connected encoder build failed; see $compileLog" }

$runReceipts = @()
foreach ($poison in @('a5', '5a')) {
    foreach ($ordinal in @('a', 'b')) {
        $runId = "encoder-poison-$poison-process-$ordinal"
        $runRoot = Join-Path $attemptRoot $runId
        New-Item -ItemType Directory -Path $runRoot | Out-Null
        $process = Start-Process -FilePath $binary -ArgumentList @($moduleRoot, $modelRoot, $runRoot, $runId, $poison, '0') -NoNewWindow -Wait -PassThru -RedirectStandardOutput (Join-Path $runRoot 'stdout.log') -RedirectStandardError (Join-Path $runRoot 'stderr.log')
        if ($process.ExitCode -ne 0) { throw "Connected encoder run $runId failed with exit code $($process.ExitCode)" }
        $receipt = Get-Content -LiteralPath (Join-Path $runRoot 'run-receipt.json') -Raw | ConvertFrom-Json
        if ($receipt.result -ne 'pass') { throw "Connected encoder receipt failed: $runId" }
        $runReceipts += $receipt
    }
}
$processIds = @($runReceipts | ForEach-Object { [int64]$_.process_id } | Sort-Object -Unique)
if ($processIds.Count -ne 4) { throw "Fresh-process gate failed: $($processIds.Count)" }
for ($tensor = 0; $tensor -lt 11; ++$tensor) {
    $hashes = @($runReceipts | ForEach-Object { $_.tensors[$tensor].sha256 } | Sort-Object -Unique)
    if ($hashes.Count -ne 1) { throw "Tensor invariance failed at index $tensor" }
}
$backendReceipt = Join-Path $attemptRoot 'backend-receipt.json'
& python $verifier backend --backend rtx --root $attemptRoot --output $backendReceipt
if ($LASTEXITCODE -ne 0) { throw "Connected encoder backend verification failed: $LASTEXITCODE" }

$gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total,pci.bus_id --format=csv,noheader,nounits | Select-Object -First 1).Trim()
$cudaVersion = (& (Join-Path $cudaPath 'bin\nvcc.exe') --version | Select-Object -Last 1).Trim()
$rigReceipt = [ordered]@{
    schema = 'neuralstudio.fg.connected-encoder-l20-l30-rig.v1'
    job_id = Split-Path -Leaf $JobRoot
    attempt_id = $AttemptId
    local_base_commit = $BaseCommit
    started_utc = $started
    ended_utc = (Get-Date).ToUniversalTime().ToString('o')
    gpu = $gpu
    cuda_toolkit = $cudaVersion
    visual_studio = $installation
    inputs = $inputRecords
    source = [ordered]@{ relative_path = 'src/fg_connected_encoder_l20_l30/rtx_oracle.cpp'; bytes = (Get-Item -LiteralPath $source).Length; sha256 = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant() }
    harness = @(
        [ordered]@{ relative_path = 'scripts/build-fg-connected-encoder-l20-l30-rtx.ps1'; bytes = (Get-Item -LiteralPath $harness).Length; sha256 = (Get-FileHash -LiteralPath $harness -Algorithm SHA256).Hash.ToLowerInvariant() },
        [ordered]@{ relative_path = 'scripts/verify-fg-connected-encoder-l20-l30.py'; bytes = (Get-Item -LiteralPath $verifier).Length; sha256 = (Get-FileHash -LiteralPath $verifier -Algorithm SHA256).Hash.ToLowerInvariant() }
    )
    binary = [ordered]@{ relative_path = "output/$AttemptId/build/fg-connected-encoder-l20-l30-rtx.exe"; bytes = (Get-Item -LiteralPath $binary).Length; sha256 = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant() }
    selected_launches = @(20,21,22,23,24,25,26,27,28,29,30)
    run_count = $runReceipts.Count
    fresh_process_count = $processIds.Count
    final_output_sha256 = $runReceipts[0].tensors[10].sha256
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
