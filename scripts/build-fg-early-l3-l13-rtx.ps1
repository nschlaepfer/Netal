[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $JobRoot,
    [Parameter(Mandatory = $true)] [string] $SourceRoot,
    [Parameter(Mandatory = $true)] [string] $AttemptId
)

$ErrorActionPreference = 'Stop'
$module = Join-Path $JobRoot 'input\cu-module-000011.bin'
$source = Join-Path $SourceRoot 'rtx_oracle.cpp'
$cuarraySource = Join-Path $SourceRoot 'rtx_cuarray_probe.cpp'
$contract = Join-Path $SourceRoot 'contract.h'
$attempt = Join-Path $JobRoot (Join-Path 'output' $AttemptId)
foreach ($required in @($module, $source, $cuarraySource, $contract)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required L3/L13 input missing: $required"
    }
}
if (Test-Path -LiteralPath $attempt) {
    throw "Attempt exists; preserve it: $attempt"
}
$moduleItem = Get-Item -LiteralPath $module
$moduleHash = (Get-FileHash -LiteralPath $module -Algorithm SHA256).Hash
if ($moduleItem.Length -ne 53920 -or
    $moduleHash -ne '06EEE29EC33202996F97C063A08CCDC767D0167252AD03B5ED457C941CB85206') {
    throw 'Selected module11 identity mismatch'
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
$object = Join-Path $build 'fg-early-l3-l13.obj'
$binary = Join-Path $build 'fg-early-l3-l13.exe'
$compileLog = Join-Path $build 'compile.log'
$command = '"{0}" >nul && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /I"{1}" /Fo"{2}" /c "{3}" >"{4}" 2>&1 && link.exe /nologo /OUT:"{5}" "{2}" bcrypt.lib d3d12.lib dxgi.lib >>"{4}" 2>&1' -f `
    $vcvars, $SourceRoot, $object, $source, $compileLog, $binary
& $env:ComSpec /d /s /c $command
if ($LASTEXITCODE -ne 0) { throw "Build failed; see $compileLog" }
$cuarrayObject = Join-Path $build 'fg-early-l3-l13-cuarray.obj'
$cuarrayBinary = Join-Path $build 'fg-early-l3-l13-cuarray.exe'
$cuarrayLog = Join-Path $build 'cuarray-compile.log'
$cuarrayCommand = '"{0}" >nul && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /I"{1}" /Fo"{2}" /c "{3}" >"{4}" 2>&1 && link.exe /nologo /OUT:"{5}" "{2}" "{6}" bcrypt.lib >>"{4}" 2>&1' -f `
    $vcvars, (Join-Path $cuda 'include'), $cuarrayObject, $cuarraySource,
    $cuarrayLog, $cuarrayBinary, (Join-Path $cuda 'lib\x64\cuda.lib')
& $env:ComSpec /d /s /c $cuarrayCommand
if ($LASTEXITCODE -ne 0) {
    throw "CUarray diagnostic build failed; see $cuarrayLog"
}

$receipts = @()
$processes = @()
$phases = @('l3-evaluate-0', 'l13-evaluate-1')
$fixtures = @('reset-history', 'generated-history')
foreach ($phase in $phases) {
    foreach ($fixture in $fixtures) {
        foreach ($poison in @('a5', '5a')) {
            foreach ($ordinal in @('a', 'b')) {
                $runId = "$phase-$fixture-poison-$poison-process-$ordinal"
                $run = Join-Path $attempt $runId
                New-Item -ItemType Directory -Path $run | Out-Null
                $process = Start-Process -FilePath $binary `
                    -ArgumentList @($module, $run, $runId, $poison,
                                    $phase, $fixture) `
                    -NoNewWindow -Wait -PassThru `
                    -RedirectStandardOutput (Join-Path $run 'stdout.log') `
                    -RedirectStandardError (Join-Path $run 'stderr.log')
                if ($process.ExitCode -ne 0) {
                    throw "L3/L13 run failed: $runId"
                }
                $receipt = Get-Content -LiteralPath `
                    (Join-Path $run 'run-receipt.json') -Raw | ConvertFrom-Json
                if ($receipt.result -ne 'pass') {
                    throw "L3/L13 receipt failed: $runId"
                }
                $receipts += $receipt
                $processes += [ordered]@{
                    run_id = $runId
                    process_id = $process.Id
                    exit_code = $process.ExitCode
                }
            }
        }
    }
}
if (@($receipts.process_id | Sort-Object -Unique).Count -ne 16) {
    throw 'Fresh-process gate failed'
}
$fixtureOutputs = [ordered]@{}
foreach ($fixture in $fixtures) {
    $subset = @($receipts | Where-Object { $_.fixture -eq $fixture })
    $hashes = @($subset.combined_output_sha256 | Sort-Object -Unique)
    if ($hashes.Count -ne 1) {
        throw "Phase/poison output invariance failed: $fixture"
    }
    $fixtureOutputs[$fixture] = $hashes[0]
}
if ($fixtureOutputs['reset-history'] -eq
    $fixtureOutputs['generated-history']) {
    throw 'History fixture sensitivity failed'
}
$cuarrayResults = [ordered]@{}
foreach ($fixture in $fixtures) {
    $run = Join-Path $attempt "cuarray-$fixture"
    New-Item -ItemType Directory -Path $run | Out-Null
    $process = Start-Process -FilePath $cuarrayBinary `
        -ArgumentList @($module, $run, 'a5', $fixture) `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput (Join-Path $run 'stdout.log') `
        -RedirectStandardError (Join-Path $run 'stderr.log')
    if ($process.ExitCode -ne 0) {
        throw "CUarray diagnostic failed: $fixture"
    }
    $diagnostic = Get-Content -LiteralPath `
        (Join-Path $run 'run-receipt.json') -Raw | ConvertFrom-Json
    $cuarrayResults[$fixture] = [ordered]@{
        output_sha256 = $diagnostic.combined_output_sha256
        authoritative_sha256 = $fixtureOutputs[$fixture]
        exact = $diagnostic.combined_output_sha256 -eq $fixtureOutputs[$fixture]
    }
}
$cuarrayCounterexample = @($cuarrayResults.Values | Where-Object {
    -not $_.exact
}).Count -gt 0
$cuarrayReceipt = [ordered]@{
    schema = 'neuralstudio.fg.early-l3-l13-cuarray-transport.v1'
    results = $cuarrayResults
    generic_cuarray_counterexample = $cuarrayCounterexample
    authoritative_backend = 'd3d12-nvapi-typed-resources'
    expected_or_archived_tensor_ingress = $false
    address_fields_recorded = $false
    result = 'diagnostic-only'
}
$cuarrayReceipt | ConvertTo-Json -Depth 6 |
    Set-Content -LiteralPath (Join-Path $attempt 'cuarray-transport.json') `
        -Encoding ascii
$receipt = [ordered]@{
    schema = 'neuralstudio.fg.early-l3-l13-rtx-backend.v1'
    job_id = Split-Path $JobRoot -Leaf
    remote_job_retained = $JobRoot
    attempt_id = $AttemptId
    gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total `
        --format=csv,noheader,nounits | Select-Object -First 1).Trim()
    module_sha256 = $moduleHash.ToLowerInvariant()
    source_sha256 = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
    contract_sha256 = (Get-FileHash -LiteralPath $contract -Algorithm SHA256).Hash.ToLowerInvariant()
    binary_sha256 = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant()
    fresh_process_count = 16
    phases = $phases
    fixtures = $fixtures
    poison_bytes = @(90, 165)
    fixture_output_sha256 = $fixtureOutputs
    all_phase_poison_process_invariant = $true
    full_write_coverage_from_two_poison_invariance = $true
    typed_resource_backend = $true
    generic_cuarray_counterexample = $cuarrayCounterexample
    expected_or_archived_tensor_ingress = $false
    address_fields_recorded = $false
    processes = $processes
    result = 'pass'
}
$receipt | ConvertTo-Json -Depth 8 |
    Set-Content -LiteralPath (Join-Path $attempt 'backend-receipt.json') `
        -Encoding ascii
Write-Output "attempt_root=$attempt"
Write-Output "reset_output_sha256=$($fixtureOutputs['reset-history'])"
Write-Output "generated_output_sha256=$($fixtureOutputs['generated-history'])"
Write-Output 'result=pass'
