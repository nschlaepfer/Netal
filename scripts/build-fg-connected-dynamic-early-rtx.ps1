[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$JobRoot,
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [string]$AttemptId = 'oracle-v1',
    [ValidateSet('720p','1080p','1440p','4k')]
    [string]$SmokeResolution = '720p',
    [switch]$Smoke
)

$ErrorActionPreference = 'Stop'
$JobRoot = [IO.Path]::GetFullPath($JobRoot)
$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
if (-not $JobRoot.StartsWith('D:\CodexLab\',
                              [StringComparison]::OrdinalIgnoreCase)) {
    throw "JobRoot must be beneath D:\CodexLab: $JobRoot"
}
$source = Join-Path $RepositoryRoot 'src\fg_connected_dynamic_early\rtx_connected.cpp'
$contract = Join-Path $RepositoryRoot 'src\fg_connected_dynamic_early\contract.h'
$baseSource = Join-Path $RepositoryRoot 'src\fg_mid_l55\rtx_oracle.cpp'
$baseContract = Join-Path $RepositoryRoot 'src\fg_mid_l55\fg_mid_l55_contract.h'
$baseInputs = Join-Path $RepositoryRoot 'src\fg_mid_l55\fg_mid_l55_inputs.h'
$harness = Join-Path $RepositoryRoot 'scripts\build-fg-connected-dynamic-early-rtx.ps1'
$modules = Join-Path $JobRoot 'inputs\modules'
$attemptRoot = Join-Path $JobRoot (Join-Path 'output' $AttemptId)
$buildRoot = Join-Path $attemptRoot 'build'
foreach ($required in @($source, $contract, $baseSource, $baseContract,
                         $baseInputs, $harness)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Missing connected dynamic input: $required"
    }
}
if (Test-Path -LiteralPath $attemptRoot) { throw "Attempt exists: $attemptRoot" }
$modulePins = [ordered]@{
    '11' = '06eee29ec33202996f97c063a08ccdc767d0167252ad03b5ed457c941cb85206'
    '15' = '9e772df29e9434e4eff1edf1cde32706ea641c2b0156a459a080565a7483f381'
    '13' = 'fd0665969da887dd92540b673009e0361a6c1de74fe54badf48f300b1fccbbf1'
    '21' = '11df926493fca0a981d560427b36c4dcf489ccb84d794501a70bda8d414ea8a2'
    '22' = '7d6602bbed88409e9ad2ef71293869dd22f7153f6387339ed65e91c12341c67a'
    '23' = 'd787cf7b861295d1b4c6a80cb374230de793aa69f288a7e724bf59278bc4a62d'
    '24' = '5a41f9d65d1f6f933ae42f36b493f60687a646b73929291bafbe9d00297b6bdf'
    '25' = '28fe0e153d1ffc80b54995c022f18395c5556f3c041cc9d724bfe7e70dc4e824'
    '14' = '540f686ffcd4508212b19296992a69ed81ca95772fd229d0a252905c8015419c'
    '27' = 'c8b8c082139005c1bf059f156827c4e331f9a6579ded1b2c9646198b1cc2abf7'
    '4' = 'c11cd7474562e20762d272162c94378a17f8f77c7de8206d695e6c469383e840'
    '5' = '3f2308a693730b44c00e064ed25dab3b73a420f2e92d215ef733c4e10a30439c'
}
foreach ($entry in $modulePins.GetEnumerator()) {
    $path = Join-Path $modules ('cu-module-{0:D6}.bin' -f [int]$entry.Key)
    if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or
        (Get-FileHash $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne
            $entry.Value) {
        throw "Module identity mismatch: $($entry.Key)"
    }
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
$binary = Join-Path $buildRoot 'fg-connected-dynamic-early-rtx.exe'
$object = Join-Path $buildRoot 'fg-connected-dynamic-early-rtx.obj'
$compileLog = Join-Path $buildRoot 'compile.log'
$compileCommand = '"{0}" >nul && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /Fo"{1}" /c "{2}" >"{3}" 2>&1 && link.exe /nologo /OUT:"{4}" "{1}" bcrypt.lib d3d12.lib dxgi.lib >>"{3}" 2>&1' -f `
    $vcvars, $object, $source, $compileLog, $binary
& $env:ComSpec /d /s /c $compileCommand
if ($LASTEXITCODE -ne 0) {
    throw "connected dynamic build failed; see $compileLog"
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
                throw "connected dynamic run failed: $runId exit=$($process.ExitCode)"
            }
            $receipts += Get-Content -LiteralPath `
                (Join-Path $runRoot 'run-receipt.json') -Raw | ConvertFrom-Json
        }
    }
}
$expectedCount = if ($Smoke) { 1 } else { 16 }
$processIds = @($receipts.process_id | Sort-Object -Unique)
if ($processIds.Count -ne $expectedCount) {
    throw 'connected dynamic fresh-process gate failed'
}
if (-not $Smoke) {
    foreach ($resolution in $resolutions) {
        $selected = @($receipts | Where-Object { $_.resolution -eq $resolution })
        if (@($selected.allocation37_sha256 | Sort-Object -Unique).Count -ne 1) {
            throw "connected dynamic poison/repeat drift: $resolution"
        }
        foreach ($name in $selected[0].checkpoint_sha256.PSObject.Properties.Name) {
            $values = @($selected | ForEach-Object {
                $_.checkpoint_sha256.$name
            } | Sort-Object -Unique)
            if ($values.Count -ne 1) {
                throw "connected checkpoint drift: $resolution/$name"
            }
        }
    }
}
$dependencyPaths = @(
    'src\fg_dynamic_module11\contract.h',
    'src\fg_dynamic_module13\contract.h',
    'src\fg_dynamic_module15\contract.h',
    'src\fg_dynamic_module21\contract.h',
    'src\fg_dynamic_module23\contract.h',
    'src\fg_dynamic_module25\contract.h',
    'src\fg_dynamic_module5\contract.h',
    'src\fg_early_l19\contract.h',
    'src\fg_resolution_preprocess_l22_l24\contract.h'
)
$dependencyHashes = [ordered]@{}
foreach ($relative in $dependencyPaths) {
    $dependencyHashes[$relative.Replace('\', '/')] =
        (Get-FileHash (Join-Path $RepositoryRoot $relative) `
            -Algorithm SHA256).Hash.ToLowerInvariant()
}
$rig = [ordered]@{
    schema = 'neuralstudio.fg.connected-dynamic-early-rig.v1'
    job_id = Split-Path -Leaf $JobRoot
    attempt_id = $AttemptId
    smoke = [bool]$Smoke
    started_utc = $startedUtc
    completed_utc = (Get-Date).ToUniversalTime().ToString('o')
    command = 'build-fg-connected-dynamic-early-rtx.ps1 -JobRoot <job> -RepositoryRoot <staged-repo> -AttemptId <attempt>'
    exit_code = 0
    gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total,pci.bus_id `
        --format=csv,noheader,nounits | Select-Object -First 1).Trim()
    cuda_toolkit = (& (Join-Path $cudaPath 'bin\nvcc.exe') --version |
        Select-Object -Last 1).Trim()
    source_sha256 = (Get-FileHash $source -Algorithm SHA256).Hash.ToLowerInvariant()
    contract_sha256 = (Get-FileHash $contract -Algorithm SHA256).Hash.ToLowerInvariant()
    base_source_sha256 = (Get-FileHash $baseSource -Algorithm SHA256).Hash.ToLowerInvariant()
    base_contract_sha256 = (Get-FileHash $baseContract -Algorithm SHA256).Hash.ToLowerInvariant()
    base_inputs_sha256 = (Get-FileHash $baseInputs -Algorithm SHA256).Hash.ToLowerInvariant()
    harness_sha256 = (Get-FileHash $harness -Algorithm SHA256).Hash.ToLowerInvariant()
    dependency_sha256 = $dependencyHashes
    module_sha256 = $modulePins
    binary_sha256 = (Get-FileHash $binary -Algorithm SHA256).Hash.ToLowerInvariant()
    run_count = $receipts.Count
    fresh_process_count = $processIds.Count
    selected_launches_per_process = 21
    true_typed_d3d12_resources = $true
    expected_or_captured_output_ingress = $false
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
