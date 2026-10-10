[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$JobRoot,
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [string]$AttemptId = 'oracle-v1',
    [switch]$Smoke,
    [ValidateSet('720p','1080p','1440p','4k')]
    [string]$SmokeResolution = '720p'
)

$ErrorActionPreference = 'Stop'
$JobRoot = [IO.Path]::GetFullPath($JobRoot)
$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
if (-not $JobRoot.StartsWith('D:\CodexLab\',
                              [StringComparison]::OrdinalIgnoreCase)) {
    throw "JobRoot must be beneath D:\CodexLab: $JobRoot"
}
$generator = Join-Path $RepositoryRoot 'scripts\generate-fg-connected-dynamic-l3-l21.py'
$harness = Join-Path $RepositoryRoot 'scripts\build-fg-connected-dynamic-l3-l21-rtx.ps1'
$earlySource = Join-Path $RepositoryRoot 'src\fg_connected_dynamic_early\rtx_connected.cpp'
$tailSource = Join-Path $RepositoryRoot 'src\fg_connected_dynamic_l3_l21\rtx_tail.inc'
$modules = Join-Path $JobRoot 'inputs\modules'
$attemptRoot = Join-Path $JobRoot (Join-Path 'output' $AttemptId)
$buildRoot = Join-Path $attemptRoot 'build'
foreach ($required in @($generator, $harness, $earlySource, $tailSource)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Missing connected L3-L21 input: $required"
    }
}
if (Test-Path -LiteralPath $attemptRoot) { throw "Attempt exists: $attemptRoot" }
$weight = Join-Path $modules 'default-token-000063.bin'
if ((Get-FileHash $weight -Algorithm SHA256).Hash.ToLowerInvariant() -ne
    '44595b02307a6b934a124abe476cfd327440e2acba7f5ed69f73195ca743e3d9') {
    throw 'L20 weight identity mismatch'
}
$l20Module = Join-Path $modules 'cu-module-000028.bin'
$l21Module = Join-Path $modules 'cu-module-000029.bin'
if ((Get-FileHash $l20Module -Algorithm SHA256).Hash.ToLowerInvariant() -ne
        '7f9c1f110e8ec9347aa968bc9019e5ffc8839057987a2e5676a0f64637c8df2d' -or
    (Get-FileHash $l21Module -Algorithm SHA256).Hash.ToLowerInvariant() -ne
        'a29a3d7041cbe7e44707e74dddf7aff61871a2397d1c097ef913bd753a149977') {
    throw 'L20/L21 selected module identity mismatch'
}
$startedUtc = (Get-Date).ToUniversalTime().ToString('o')
New-Item -ItemType Directory -Path $buildRoot | Out-Null
& python $generator --repository $RepositoryRoot `
    --output (Join-Path $buildRoot 'generated')
if ($LASTEXITCODE -ne 0) { throw 'connected L3-L21 generation failed' }
$source = Join-Path $buildRoot 'generated\generated_rtx.cpp'
$generatorReceipt = Join-Path $buildRoot 'generated\generator-receipt.json'
$cudaPath = if ($env:CUDA_PATH) {
    $env:CUDA_PATH
} else {
    'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
}
$vswhere = Join-Path ${env:ProgramFiles(x86)} `
    'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
$env:TEMP = $buildRoot
$env:TMP = $buildRoot
$binary = Join-Path $buildRoot 'fg-connected-dynamic-l3-l21-rtx.exe'
$object = Join-Path $buildRoot 'fg-connected-dynamic-l3-l21-rtx.obj'
$compileLog = Join-Path $buildRoot 'compile.log'
$earlyInclude = Join-Path $RepositoryRoot 'src\fg_connected_dynamic_early'
$tailInclude = Join-Path $RepositoryRoot 'src\fg_connected_dynamic_l3_l21'
$compileCommand = '"{0}" >nul && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /I"{1}" /I"{2}" /Fo"{3}" /c "{4}" >"{5}" 2>&1 && link.exe /nologo /OUT:"{6}" "{3}" bcrypt.lib d3d12.lib dxgi.lib >>"{5}" 2>&1' -f `
    $vcvars, $earlyInclude, $tailInclude, $object, $source, $compileLog, $binary
& $env:ComSpec /d /s /c $compileCommand
if ($LASTEXITCODE -ne 0) {
    throw "connected L3-L21 build failed; see $compileLog"
}
$resolutions = if ($Smoke) { @($SmokeResolution) } else {
    @('720p', '1080p', '1440p', '4k')
}
$poisons = if ($Smoke) { @('a5') } else { @('a5', '5a') }
$ordinals = if ($Smoke) { @('a') } else { @('a', 'b') }
$receipts = @()
foreach ($resolution in $resolutions) {
    foreach ($poison in $poisons) {
        foreach ($ordinal in $ordinals) {
            $runId = "$resolution-poison-$poison-process-$ordinal"
            $runRoot = Join-Path $attemptRoot $runId
            New-Item -ItemType Directory -Path $runRoot | Out-Null
            $artifactMode = if ($poison -eq 'a5' -and $ordinal -eq 'a') {
                'full'
            } else {
                'hash-only'
            }
            $process = Start-Process -FilePath $binary `
                -ArgumentList @($modules, $runRoot, $runId, $poison,
                                $resolution, $artifactMode) `
                -NoNewWindow -Wait -PassThru `
                -RedirectStandardOutput (Join-Path $runRoot 'stdout.log') `
                -RedirectStandardError (Join-Path $runRoot 'stderr.log')
            if ($process.ExitCode -ne 0) {
                throw "connected L3-L21 run failed: $runId exit=$($process.ExitCode)"
            }
            $receipts += Get-Content -LiteralPath `
                (Join-Path $runRoot 'run-receipt.json') -Raw | ConvertFrom-Json
        }
    }
}
$expectedCount = if ($Smoke) { 1 } else { 16 }
if (@($receipts.process_id | Sort-Object -Unique).Count -ne $expectedCount) {
    throw 'connected L3-L21 fresh-process gate failed'
}
if (-not $Smoke) {
    foreach ($resolution in $resolutions) {
        $selected = @($receipts | Where-Object { $_.resolution -eq $resolution })
        foreach ($field in @('allocation37_sha256')) {
            if (@($selected.$field | Sort-Object -Unique).Count -ne 1) {
                throw "connected L3-L21 poison drift: $resolution/$field"
            }
        }
        foreach ($name in $selected[0].checkpoint_sha256.PSObject.Properties.Name) {
            if (@($selected | ForEach-Object {
                $_.checkpoint_sha256.$name
            } | Sort-Object -Unique).Count -ne 1) {
                throw "connected L3-L21 checkpoint drift: $resolution/$name"
            }
        }
    }
}
$rig = [ordered]@{
    schema = 'neuralstudio.fg.connected-dynamic-l3-l21-rig.v1'
    job_id = Split-Path -Leaf $JobRoot
    attempt_id = $AttemptId
    smoke = [bool]$Smoke
    started_utc = $startedUtc
    completed_utc = (Get-Date).ToUniversalTime().ToString('o')
    exit_code = 0
    gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total,pci.bus_id `
        --format=csv,noheader,nounits | Select-Object -First 1).Trim()
    cuda_toolkit = (& (Join-Path $cudaPath 'bin\nvcc.exe') --version |
        Select-Object -Last 1).Trim()
    generator_sha256 = (Get-FileHash $generator -Algorithm SHA256).Hash.ToLowerInvariant()
    generator_receipt_sha256 = (Get-FileHash $generatorReceipt -Algorithm SHA256).Hash.ToLowerInvariant()
    generated_source_sha256 = (Get-FileHash $source -Algorithm SHA256).Hash.ToLowerInvariant()
    accepted_early_source_sha256 = (Get-FileHash $earlySource -Algorithm SHA256).Hash.ToLowerInvariant()
    tail_source_sha256 = (Get-FileHash $tailSource -Algorithm SHA256).Hash.ToLowerInvariant()
    harness_sha256 = (Get-FileHash $harness -Algorithm SHA256).Hash.ToLowerInvariant()
    weight_sha256 = (Get-FileHash $weight -Algorithm SHA256).Hash.ToLowerInvariant()
    l20_module_sha256 = (Get-FileHash $l20Module -Algorithm SHA256).Hash.ToLowerInvariant()
    l21_module_sha256 = (Get-FileHash $l21Module -Algorithm SHA256).Hash.ToLowerInvariant()
    binary_sha256 = (Get-FileHash $binary -Algorithm SHA256).Hash.ToLowerInvariant()
    run_count = $receipts.Count
    fresh_process_count = @($receipts.process_id | Sort-Object -Unique).Count
    selected_launches_per_process = 23
    live_allocation37_device_handoff = $true
    expected_or_archived_intermediate_ingress = $false
    address_fields_recorded = $false
    result = 'pass'
}
$utf8 = New-Object System.Text.UTF8Encoding($false)
$rigPath = Join-Path $attemptRoot 'rig-receipt.json'
[IO.File]::WriteAllText(
    $rigPath, (($rig | ConvertTo-Json -Depth 8) + "`n"), $utf8)
Write-Host "attempt_root=$attemptRoot"
Write-Host "rig_receipt_sha256=$((Get-FileHash $rigPath -Algorithm SHA256).Hash.ToLowerInvariant())"
Write-Host 'result=pass'
