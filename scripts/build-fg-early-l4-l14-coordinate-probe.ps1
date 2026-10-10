param(
    [Parameter(Mandatory = $true)][string]$JobRoot,
    [Parameter(Mandatory = $true)][string]$RepositoryRoot
)
$ErrorActionPreference = 'Stop'
$JobRoot = [IO.Path]::GetFullPath($JobRoot)
$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
if (-not $JobRoot.StartsWith('D:\CodexLab\')) { throw 'JobRoot must be beneath D:\CodexLab' }
$source = Join-Path $RepositoryRoot 'src\fg_early_l4_l14\rtx_coordinate_probe.cu'
$textureSource = Join-Path $RepositoryRoot 'src\fg_early_l4_l14\rtx_texture_probe.cu'
$output = Join-Path $JobRoot 'output\rtx-coordinate-probe.exe'
$textureOutput = Join-Path $JobRoot 'output\rtx-texture-probe.exe'
$text = Join-Path $JobRoot 'output\rtx-coordinate-probe.txt'
$textureText = Join-Path $JobRoot 'output\rtx-texture-probe.txt'
$textureBinary = Join-Path $JobRoot 'output\rtx-texture-probe.bin'
$cudaPath = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3' }
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
$command = '"{0}" >nul && "{1}" -std=c++17 -arch=sm_120 -o "{2}" "{3}"' -f $vcvars,(Join-Path $cudaPath 'bin\nvcc.exe'),$output,$source
& $env:ComSpec /d /s /c $command
if ($LASTEXITCODE -ne 0) { throw 'coordinate probe compile failed' }
& $output | Set-Content -Encoding ascii $text
if ($LASTEXITCODE -ne 0) { throw 'coordinate probe execution failed' }
Get-Content $text | Select-Object -Index 147,148,149
$command = '"{0}" >nul && "{1}" -std=c++17 -arch=sm_120 -o "{2}" "{3}"' -f $vcvars,(Join-Path $cudaPath 'bin\nvcc.exe'),$textureOutput,$textureSource
& $env:ComSpec /d /s /c $command
if ($LASTEXITCODE -ne 0) { throw 'texture probe compile failed' }
& $textureOutput $textureBinary | Set-Content -Encoding ascii $textureText
if ($LASTEXITCODE -ne 0) { throw 'texture probe execution failed' }
Get-Content $textureText
