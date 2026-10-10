param(
    [Parameter(Mandatory = $true)][string]$JobRoot,
    [Parameter(Mandatory = $true)][string]$RepositoryRoot
)
$ErrorActionPreference = 'Stop'
$JobRoot = [IO.Path]::GetFullPath($JobRoot)
$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
if (-not $JobRoot.StartsWith('D:\CodexLab\')) { throw 'JobRoot must be beneath D:\CodexLab' }
$source = Join-Path $RepositoryRoot 'src\fg_early_l19\rtx_coordinate_probe.cu'
$output = Join-Path $JobRoot 'output\rtx-coordinate-probe.exe'
$binary = Join-Path $JobRoot 'output\rtx-coordinate-probe.bin'
$cudaPath = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3' }
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
$command = '"{0}" >nul && "{1}" -std=c++20 -arch=sm_120 -I"{2}" -o "{3}" "{4}"' -f $vcvars,(Join-Path $cudaPath 'bin\nvcc.exe'),(Join-Path $RepositoryRoot 'src\fg_early_l19'),$output,$source
& $env:ComSpec /d /s /c $command
if ($LASTEXITCODE -ne 0) { throw 'L19 coordinate probe compile failed' }
& $output $binary
if ($LASTEXITCODE -ne 0) { throw 'L19 coordinate probe execution failed' }
Write-Host "probe_sha256=$((Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant())"
Write-Host 'result=pass'
