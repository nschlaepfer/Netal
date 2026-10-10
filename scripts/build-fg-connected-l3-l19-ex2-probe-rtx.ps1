[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $JobRoot,
    [Parameter(Mandatory = $true)] [string] $AttemptId
)

$ErrorActionPreference = 'Stop'
$root = Join-Path $JobRoot (Join-Path 'output' $AttemptId)
if (Test-Path -LiteralPath $root) { throw "Attempt exists: $root" }
New-Item -ItemType Directory -Path $root | Out-Null
$source = Join-Path $JobRoot 'input\rtx_ex2_live_probe.cu'
$binary = Join-Path $root 'probe.exe'
$log = Join-Path $root 'compile.log'
$vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
$nvcc = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe'
$command = '"{0}" >nul && "{1}" -std=c++17 -arch=sm_120 -o "{2}" "{3}" >"{4}" 2>&1' -f `
    $vcvars, $nvcc, $binary, $source, $log
& $env:ComSpec /d /s /c $command
if ($LASTEXITCODE -ne 0) { throw "Compile failed: $log" }
$receipt = Join-Path $root 'receipt.json'
$stderr = Join-Path $root 'stderr.log'
$process = Start-Process -FilePath $binary -NoNewWindow -Wait -PassThru `
    -RedirectStandardOutput $receipt -RedirectStandardError $stderr
if ($process.ExitCode -ne 0) { throw "Probe failed: $stderr" }
Get-Content -LiteralPath $receipt -Raw
