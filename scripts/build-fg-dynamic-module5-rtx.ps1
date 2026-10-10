[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$JobRoot,
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [string]$AttemptId = 'oracle-v1',
    [switch]$Smoke
)

$ErrorActionPreference = 'Stop'
$JobRoot = [IO.Path]::GetFullPath($JobRoot)
$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
if (-not $JobRoot.StartsWith('D:\CodexLab\',
                              [StringComparison]::OrdinalIgnoreCase)) {
    throw "JobRoot must be beneath D:\CodexLab: $JobRoot"
}
$source = Join-Path $RepositoryRoot 'src\fg_dynamic_module5\rtx_nvapi_oracle.cpp'
$contract = Join-Path $RepositoryRoot 'src\fg_dynamic_module5\contract.h'
$earlySource = Join-Path $RepositoryRoot 'src\fg_early_l19\rtx_nvapi_oracle.cpp'
$earlyContract = Join-Path $RepositoryRoot 'src\fg_early_l19\contract.h'
$baseSource = Join-Path $RepositoryRoot 'src\fg_compositor_l87\rtx_nvapi_oracle.cpp'
$baseContract = Join-Path $RepositoryRoot 'src\fg_compositor_l87\compositor_l87_contract.h'
$harness = Join-Path $RepositoryRoot 'scripts\build-fg-dynamic-module5-rtx.ps1'
$module = Join-Path $JobRoot 'inputs\cu-module-000005.bin'
$attemptRoot = Join-Path $JobRoot (Join-Path 'output' $AttemptId)
$buildRoot = Join-Path $attemptRoot 'build'
foreach ($required in @($source, $contract, $earlySource, $earlyContract,
                         $baseSource, $baseContract, $harness, $module)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Missing dynamic module5 input: $required"
    }
}
if (Test-Path -LiteralPath $attemptRoot) { throw "Attempt exists: $attemptRoot" }
if ((Get-Item -LiteralPath $module).Length -ne 14448 -or
    (Get-FileHash -LiteralPath $module -Algorithm SHA256).Hash -ne
        '3F2308A693730B44C00E064ED25DAB3B73A420F2E92D215EF733C4E10A30439C') {
    throw 'dynamic module5 identity mismatch'
}
$startedUtc = (Get-Date).ToUniversalTime().ToString('o')
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
New-Item -ItemType Directory -Path $buildRoot | Out-Null
$env:TEMP = $buildRoot
$env:TMP = $buildRoot
$binary = Join-Path $buildRoot 'fg-dynamic-module5-rtx.exe'
$object = Join-Path $buildRoot 'fg-dynamic-module5-rtx.obj'
$compileLog = Join-Path $buildRoot 'compile.log'
$compileCommand = '"{0}" >nul && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /Fo"{1}" /c "{2}" >"{3}" 2>&1 && link.exe /nologo /OUT:"{4}" "{1}" bcrypt.lib d3d12.lib dxgi.lib >>"{3}" 2>&1' -f `
    $vcvars, $object, $source, $compileLog, $binary
& $env:ComSpec /d /s /c $compileCommand
if ($LASTEXITCODE -ne 0) {
    throw "dynamic module5 build failed; see $compileLog"
}

$resolutions = if ($Smoke) { @('720p') } else {
    @('720p', '1080p', '1440p', '4k')
}
$fixtures = @('first-frame', 'fan-in')
$poisons = if ($Smoke) { @('a5') } else { @('a5', '5a') }
$ordinals = if ($Smoke) { @('a') } else { @('a', 'b') }
$receipts = @()
foreach ($resolution in $resolutions) {
    foreach ($fixture in $fixtures) {
        foreach ($poison in $poisons) {
            foreach ($ordinal in $ordinals) {
                $runId = "$resolution-$fixture-poison-$poison-process-$ordinal"
                $runRoot = Join-Path $attemptRoot $runId
                New-Item -ItemType Directory -Path $runRoot | Out-Null
                $artifactMode = if ($poison -eq 'a5' -and $ordinal -eq 'a') {
                    'full'
                } else {
                    'hash-only'
                }
                $process = Start-Process -FilePath $binary `
                    -ArgumentList @($module, $runRoot, $runId, $poison,
                                    $resolution, $fixture, $artifactMode) `
                    -NoNewWindow -Wait -PassThru `
                    -RedirectStandardOutput (Join-Path $runRoot 'stdout.log') `
                    -RedirectStandardError (Join-Path $runRoot 'stderr.log')
                if ($process.ExitCode -ne 0) {
                    throw "dynamic module5 run failed: $runId exit=$($process.ExitCode)"
                }
                $receipts += Get-Content -LiteralPath `
                    (Join-Path $runRoot 'run-receipt.json') -Raw | ConvertFrom-Json
            }
        }
    }
}
$expectedCount = if ($Smoke) { 2 } else { 32 }
$processIds = @($receipts.process_id | Sort-Object -Unique)
if ($processIds.Count -ne $expectedCount) {
    throw 'dynamic module5 fresh-process gate failed'
}
if (-not $Smoke) {
    foreach ($resolution in $resolutions) {
        foreach ($fixture in $fixtures) {
            $selected = @($receipts | Where-Object {
                $_.resolution -eq $resolution -and
                $_.phase_fixture -eq $fixture
            })
            foreach ($field in @('tensor_output_sha256',
                                  'mask_output_sha256',
                                  'vector_output_sha256')) {
                if (@($selected.$field | Sort-Object -Unique).Count -ne 1) {
                    throw "dynamic module5 repeat/poison drift: $resolution/$fixture/$field"
                }
            }
        }
    }
}
$rig = [ordered]@{
    schema = 'neuralstudio.fg.dynamic-module5-rig.v1'
    job_id = Split-Path -Leaf $JobRoot
    attempt_id = $AttemptId
    smoke = [bool]$Smoke
    started_utc = $startedUtc
    completed_utc = (Get-Date).ToUniversalTime().ToString('o')
    command = 'build-fg-dynamic-module5-rtx.ps1 -JobRoot <job> -RepositoryRoot <staged-repo> -AttemptId <attempt>'
    exit_code = 0
    gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total,pci.bus_id `
        --format=csv,noheader,nounits | Select-Object -First 1).Trim()
    cuda_toolkit = (& (Join-Path $cudaPath 'bin\nvcc.exe') --version |
        Select-Object -Last 1).Trim()
    module_sha256 = (Get-FileHash $module -Algorithm SHA256).Hash.ToLowerInvariant()
    source_sha256 = (Get-FileHash $source -Algorithm SHA256).Hash.ToLowerInvariant()
    contract_sha256 = (Get-FileHash $contract -Algorithm SHA256).Hash.ToLowerInvariant()
    early_source_sha256 = (Get-FileHash $earlySource -Algorithm SHA256).Hash.ToLowerInvariant()
    early_contract_sha256 = (Get-FileHash $earlyContract -Algorithm SHA256).Hash.ToLowerInvariant()
    base_source_sha256 = (Get-FileHash $baseSource -Algorithm SHA256).Hash.ToLowerInvariant()
    base_contract_sha256 = (Get-FileHash $baseContract -Algorithm SHA256).Hash.ToLowerInvariant()
    harness_sha256 = (Get-FileHash $harness -Algorithm SHA256).Hash.ToLowerInvariant()
    binary_sha256 = (Get-FileHash $binary -Algorithm SHA256).Hash.ToLowerInvariant()
    run_count = $receipts.Count
    fresh_process_count = $processIds.Count
    true_typed_d3d12_resources = $true
    expected_or_captured_output_ingress = $false
    address_fields_recorded = $false
    result = 'pass'
}
$utf8 = New-Object System.Text.UTF8Encoding($false)
$rigPath = Join-Path $attemptRoot 'rig-receipt.json'
[IO.File]::WriteAllText(
    $rigPath, (($rig | ConvertTo-Json -Depth 6) + "`n"), $utf8)
Write-Host "attempt_root=$attemptRoot"
Write-Host "rig_receipt_sha256=$((Get-FileHash $rigPath -Algorithm SHA256).Hash.ToLowerInvariant())"
Write-Host 'result=pass'
