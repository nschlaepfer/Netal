<# Validate Runtime v0 trust, local API, real output, teardown, and repeatability. #>
[CmdletBinding()]
param(
    [switch]$ArmInference,
    [ValidateRange(1, 20)][int]$Iterations = 3
)

$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent $PSScriptRoot
$build = Join-Path $repository 'build\runtime-v0'
$binary = Join-Path $build 'neuralstudio-runtime.exe'
$sourcePackage = Join-Path $repository 'packages\com.nvidia.ngx.dlss-nr'
$runtimePackage = Join-Path $build 'packages\com.nvidia.ngx.dlss-nr'
$output = Join-Path $build 'output\api-diagnostic.bmp'

& (Join-Path $PSScriptRoot 'build-dlssnr-bridge-worker.ps1') | Out-Null
& (Join-Path $PSScriptRoot 'build-ngx-adapter.ps1') | Out-Null
& (Join-Path $PSScriptRoot 'stage-ngx-runtime-package.ps1') -SkipBuild | Out-Null
& (Join-Path $PSScriptRoot 'build-runtime-v0.ps1') | Out-Null

function Assert-That([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}

# Inference must never happen just because the runtime binary was launched.
$disarmedText = & $binary --diagnostic --frames 1 2>$null
$disarmedExit = $LASTEXITCODE
$disarmed = $disarmedText | ConvertFrom-Json
Assert-That ($disarmedExit -ne 0 -and -not $disarmed.ok -and $disarmed.detail -match 'disarmed') `
    'The explicit inference arm gate did not fail closed.'

# A modified adapter must be rejected before it is loaded.
$tamperRoot = Join-Path $build ('tamper-' + [guid]::NewGuid().ToString('N'))
$resolvedBuild = [IO.Path]::GetFullPath($build).TrimEnd('\') + '\'
$resolvedTamper = [IO.Path]::GetFullPath($tamperRoot)
Assert-That ($resolvedTamper.StartsWith($resolvedBuild, [StringComparison]::OrdinalIgnoreCase)) `
    'Refusing to create a tamper fixture outside the Runtime v0 build directory.'
try {
    Copy-Item -LiteralPath $sourcePackage -Destination $tamperRoot -Recurse
    foreach ($provider in @('sl.dlss_nr.dll', 'nvngx_dlssnr.dll')) {
        $providerSource = Join-Path (Join-Path $runtimePackage 'runtime') $provider
        $providerTarget = Join-Path (Join-Path $tamperRoot 'runtime') $provider
        New-Item -ItemType HardLink -Path $providerTarget -Target $providerSource | Out-Null
    }
    $tamperedAdapter = Join-Path $tamperRoot 'ns_adapter_ngx.dll'
    $stream = [IO.File]::Open($tamperedAdapter, [IO.FileMode]::Open,
                              [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    try {
        $stream.Position = $stream.Length - 1
        $original = $stream.ReadByte()
        $stream.Position = $stream.Length - 1
        $stream.WriteByte($original -bxor 0x01)
    } finally { $stream.Dispose() }
    $tamperText = & $binary --diagnostic --arm-inference --package $tamperRoot --frames 1 2>$null
    $tamperExit = $LASTEXITCODE
    $tamper = $tamperText | ConvertFrom-Json
    Assert-That ($tamperExit -ne 0 -and -not $tamper.ok -and $tamper.detail -match 'hash mismatch') `
        'A modified adapter was not rejected by its exact hash before load.'
} finally {
    if (Test-Path -LiteralPath $tamperRoot) {
        $confirmed = [IO.Path]::GetFullPath($tamperRoot)
        Assert-That ($confirmed.StartsWith($resolvedBuild, [StringComparison]::OrdinalIgnoreCase)) `
            'Refusing to remove an unverified tamper fixture path.'
        Remove-Item -LiteralPath $confirmed -Recurse -Force
    }
}

function Send-Rpc([IO.Pipes.NamedPipeClientStream]$Pipe, [hashtable]$Request) {
    $json = $Request | ConvertTo-Json -Compress -Depth 8
    $payload = [Text.Encoding]::UTF8.GetBytes($json)
    $header = [BitConverter]::GetBytes([uint32]$payload.Length)
    $Pipe.Write($header, 0, $header.Length)
    $Pipe.Write($payload, 0, $payload.Length)
    $Pipe.Flush()
    $header = [byte[]]::new(4)
    $read = 0
    while ($read -lt 4) {
        $count = $Pipe.Read($header, $read, 4 - $read)
        if ($count -eq 0) { throw 'Runtime closed the pipe while reading a response header.' }
        $read += $count
    }
    $length = [BitConverter]::ToUInt32($header, 0)
    Assert-That ($length -gt 0 -and $length -le 1MB) 'Runtime response length was invalid.'
    $payload = [byte[]]::new($length)
    $read = 0
    while ($read -lt $length) {
        $count = $Pipe.Read($payload, $read, $length - $read)
        if ($count -eq 0) { throw 'Runtime closed the pipe while reading a response body.' }
        $read += $count
    }
    return ([Text.Encoding]::UTF8.GetString($payload) | ConvertFrom-Json)
}

$arguments = @('--service')
if ($ArmInference) { $arguments += '--arm-inference' }
$stdout = Join-Path $build 'runtime-api.stdout.log'
$stderr = Join-Path $build 'runtime-api.stderr.log'
$previousTestMode = $env:NEURALSTUDIO_TEST_MODE
if ($ArmInference) { $env:NEURALSTUDIO_TEST_MODE = '1' }
try {
    $process = Start-Process -FilePath $binary -ArgumentList $arguments -PassThru `
        -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
} finally {
    $env:NEURALSTUDIO_TEST_MODE = $previousTestMode
}
try {
    $workingSetBefore = $process.WorkingSet64
    $maxHandleDelta = 0
    $workingSetGrowth = 0
    $faultRecoveries = 0
    $pipe = [IO.Pipes.NamedPipeClientStream]::new('.', 'neuralstudio.v1',
        [IO.Pipes.PipeDirection]::InOut, [IO.Pipes.PipeOptions]::Asynchronous)
    $pipe.Connect(10000)
    try {
        $beforeHandshake = Send-Rpc $pipe @{jsonrpc='2.0'; id=1; method='runtime.getInfo'; params=@{}}
        Assert-That ($beforeHandshake.error.code -eq -32000) 'Local API accepted a method before handshake.'
        $wrongMajor = Send-Rpc $pipe @{jsonrpc='2.0'; id=2; method='rpc.handshake'; params=@{apiVersion='2.0'}}
        Assert-That ($wrongMajor.error.code -eq -32001) 'Local API accepted an unsupported major version.'
        $handshake = Send-Rpc $pipe @{jsonrpc='2.0'; id=3; method='rpc.handshake'; params=@{apiVersion='1.0'; client=@{name='runtime-v0-test';version='0.1.0'}}}
        Assert-That ($handshake.result.apiVersion -eq '1.0') 'Local API handshake failed.'
        $validated = Send-Rpc $pipe @{jsonrpc='2.0'; id=4; method='packages.validate'; params=@{}}
        Assert-That ($validated.result.valid -and $validated.result.packageId -eq 'com.nvidia.ngx.dlss-nr') `
            'Local API package validation failed.'
        $opened = Send-Rpc $pipe @{jsonrpc='2.0'; id=5; method='sources.open'; params=@{kind='diagnostic'}}
        Assert-That ($opened.result.sourceId -eq 'diagnostic-1') 'Diagnostic source did not open.'
        $created = Send-Rpc $pipe @{jsonrpc='2.0'; id=6; method='pipelines.create'; params=@{
            width=640; height=360; frames=2; outputPath=$output
        }}
        Assert-That ($created.result.state -eq 'Standby') 'Pipeline did not enter Standby.'
        if (-not $ArmInference) {
            $start = Send-Rpc $pipe @{jsonrpc='2.0'; id=7; method='pipelines.start'; params=@{}}
            Assert-That ($start.error.code -eq -32023) 'Disarmed service allowed inference.'
        } else {
            $starts = @()
            foreach ($index in 1..$Iterations) {
                $start = Send-Rpc $pipe @{jsonrpc='2.0'; id=(10+$index); method='pipelines.start'; params=@{}}
                Assert-That ($start.result.ok -and $start.result.state -eq 'Active') `
                    "Diagnostic iteration $index did not reach evidence-backed Active."
                Assert-That ($start.result.changedFromCanaryBytes -gt 0 -and
                             $start.result.changedFromInputBytes -gt 0 -and
                             $start.result.severeD3dMessages -eq 0) `
                    "Diagnostic iteration $index lacked fresh-output proof."
                $starts += $start.result
                Assert-That (@(Get-Process -Name 'neuralstudio-dlssnr-bridge-worker' -ErrorAction SilentlyContinue).Count -eq 0) `
                    "Diagnostic iteration $index left a model worker running after teardown."
            }
            Assert-That (Test-Path -LiteralPath $output -PathType Leaf) 'The API pipeline did not write its output image.'
            $handleDeltas = @($starts | ForEach-Object { [long]$_.endHandleCount - [long]$_.startHandleCount })
            Assert-That (($handleDeltas | Measure-Object -Maximum).Maximum -le 2) `
                "Runtime leaked more than the two one-time Windows subsystem handles: $($handleDeltas -join ', ')."
            $maxHandleDelta = ($handleDeltas | Measure-Object -Maximum).Maximum
            $process.Refresh()
            $workingSetGrowth = [long]$process.WorkingSet64 - [long]$workingSetBefore
            Assert-That ($workingSetGrowth -le 128MB) `
                "Runtime working set grew by more than 128 MiB across the soak: $workingSetGrowth bytes."
            foreach ($failure in @('crash', 'timeout', 'device-loss')) {
                $faultOutput = Join-Path $build "output\api-bypass-$failure.bmp"
                $createdFault = Send-Rpc $pipe @{jsonrpc='2.0'; id=50; method='pipelines.create'; params=@{
                    width=640; height=360; frames=2; outputPath=$faultOutput; _testFailure=$failure
                }}
                Assert-That ($createdFault.result.state -eq 'Standby') "Unable to create the $failure fault fixture."
                $faulted = Send-Rpc $pipe @{jsonrpc='2.0'; id=51; method='pipelines.start'; params=@{}}
                Assert-That ($faulted.error.code -eq -32024) "The $failure fixture did not surface a model failure."
                $bypassed = Send-Rpc $pipe @{jsonrpc='2.0'; id=52; method='pipelines.getState'; params=@{}}
                Assert-That ($bypassed.result.state -eq 'Bypassed' -and
                             $bypassed.result.inputHash -eq $bypassed.result.outputHash -and
                             (Test-Path -LiteralPath $faultOutput -PathType Leaf)) `
                    "The $failure fixture did not preserve a verified pass-through frame in Bypassed."
                $createdRecovery = Send-Rpc $pipe @{jsonrpc='2.0'; id=53; method='pipelines.create'; params=@{
                    width=640; height=360; frames=1; outputPath=$output
                }}
                Assert-That ($createdRecovery.result.state -eq 'Standby') "Unable to create recovery after $failure."
                $recovered = Send-Rpc $pipe @{jsonrpc='2.0'; id=54; method='pipelines.start'; params=@{}}
                Assert-That ($recovered.result.ok -and $recovered.result.state -eq 'Active') `
                    "Runtime did not recover after $failure."
                $faultRecoveries++
            }
        }
        $shutdown = Send-Rpc $pipe @{jsonrpc='2.0'; id=99; method='runtime.shutdown'; params=@{}}
        Assert-That ($shutdown.result.shuttingDown) 'Runtime shutdown was not acknowledged.'
    } finally { $pipe.Dispose() }
    Assert-That ($process.WaitForExit(10000)) 'Runtime service did not exit after shutdown.'
    Assert-That ($process.ExitCode -eq 0) "Runtime service exited with $($process.ExitCode)."
} finally {
    if (-not $process.HasExited) { Stop-Process -Id $process.Id -Force }
}

[pscustomobject]@{
    Result = 'PASS'
    InferenceArmed = [bool]$ArmInference
    Iterations = if ($ArmInference) { $Iterations } else { 0 }
    TamperRejected = $true
    HandshakeEnforced = $true
    Output = if ($ArmInference) { $output } else { $null }
    MaxHandleDelta = $maxHandleDelta
    WorkingSetGrowthBytes = $workingSetGrowth
    FaultRecoveries = $faultRecoveries
}
