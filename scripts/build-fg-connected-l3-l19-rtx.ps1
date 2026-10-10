[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $JobRoot,
    [Parameter(Mandatory = $true)] [string] $SourceRoot,
    [Parameter(Mandatory = $true)] [string] $AttemptId,
    [Parameter(Mandatory = $true)] [string] $DependencyReceipt
)

$ErrorActionPreference = 'Stop'
$attempt = Join-Path $JobRoot (Join-Path 'output' $AttemptId)
if (Test-Path -LiteralPath $attempt) {
    throw "Attempt exists; preserve it: $attempt"
}
New-Item -ItemType Directory -Path $attempt | Out-Null
if (-not (Test-Path -LiteralPath $DependencyReceipt -PathType Leaf)) {
    throw 'Committed-dependency receipt missing'
}
$dependency = Get-Content -LiteralPath $DependencyReceipt -Raw |
    ConvertFrom-Json
if (-not $dependency.full_executor_ready) {
    $blocker = [ordered]@{
        schema = 'neuralstudio.fg.connected-l3-l19-dependency-blocker.v1'
        missing_committed_dependencies = @($dependency.fail_closed_missing)
        uncommitted_files_ignored = $true
        result = 'blocked-on-commits'
    }
    $blocker | ConvertTo-Json -Depth 5 |
        Set-Content -LiteralPath (Join-Path $attempt 'dependency-blocker.json') `
            -Encoding ascii
    throw 'Connected L3-L19 dependencies are not committed'
}
$source = Join-Path $SourceRoot 'rtx_connected.cpp'
$contract = Join-Path $SourceRoot 'contract.h'
foreach ($required in @($source, $contract)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Connected source input missing: $required"
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
$object = Join-Path $build 'connected-l3-l19.obj'
$binary = Join-Path $build 'connected-l3-l19.exe'
$log = Join-Path $build 'compile.log'
$command = '"{0}" >nul && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /I"{1}" /Fo"{2}" /c "{3}" >"{4}" 2>&1 && link.exe /nologo /OUT:"{5}" "{2}" bcrypt.lib d3d12.lib dxgi.lib >>"{4}" 2>&1' -f `
    $vcvars, $SourceRoot, $object, $source, $log, $binary
& $env:ComSpec /d /s /c $command
if ($LASTEXITCODE -ne 0) { throw "Build failed; see $log" }
foreach ($poison in @('a5','5a')) {
    foreach ($ordinal in @('a','b')) {
        $runId = "connected-l3-l19-poison-$poison-process-$ordinal"
        $run = Join-Path $attempt $runId
        New-Item -ItemType Directory -Path $run | Out-Null
        $process = Start-Process -FilePath $binary `
            -ArgumentList @((Join-Path $JobRoot 'input\modules'),
                            (Join-Path $JobRoot 'input\model'),
                            $run,$runId,$poison) `
            -NoNewWindow -Wait -PassThru `
            -RedirectStandardOutput (Join-Path $run 'stdout.log') `
            -RedirectStandardError (Join-Path $run 'stderr.log')
        if ($process.ExitCode -ne 0) {
            throw "Connected RTX run failed: $runId"
        }
    }
}
Write-Output "attempt_root=$attempt"
Write-Output 'result=execution-complete-verification-required'
