param(
    [Parameter(Mandatory = $true)][string]$JobRoot,
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [Parameter(Mandatory = $true)][string]$Attempt
)
$ErrorActionPreference = 'Stop'
$output = Join-Path $JobRoot "output\$Attempt"
if (Test-Path -LiteralPath $output) { throw "attempt exists: $output" }
New-Item -ItemType Directory -Path $output | Out-Null
$source = Join-Path $RepositoryRoot 'src\fg_distortion_path\connected_coordinate_probe.cu'
$cudaPath = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3' }
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
$binary = Join-Path $output 'connected-coordinate-probe.exe'
$samples = Join-Path $output 'connected-coordinate-probe.bin'
$log = Join-Path $output 'compile.log'
$command = '"{0}" >nul && "{1}" -std=c++20 -arch=sm_120 -I"{2}" -o "{3}" "{4}" >"{5}" 2>&1' -f $vcvars, (Join-Path $cudaPath 'bin\nvcc.exe'), (Join-Path $RepositoryRoot 'src\fg_distortion_path'), $binary, $source, $log
& $env:ComSpec /d /s /c $command
if ($LASTEXITCODE) { throw "coordinate probe build failed: $log" }
& $binary $samples
if ($LASTEXITCODE) { throw 'coordinate probe run failed' }
Write-Host "samples_sha256=$((Get-FileHash -LiteralPath $samples -Algorithm SHA256).Hash.ToLowerInvariant())"
Write-Host 'result=pass'
