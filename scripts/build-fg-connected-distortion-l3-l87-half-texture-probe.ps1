param(
    [Parameter(Mandatory = $true)][string]$JobRoot,
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [Parameter(Mandatory = $true)][string]$AttemptId
)
$ErrorActionPreference = 'Stop'
$attempt = Join-Path $JobRoot "output\$AttemptId"
if (Test-Path -LiteralPath $attempt) { throw "attempt exists: $attempt" }
New-Item -ItemType Directory -Path $attempt | Out-Null
$source = Join-Path $RepositoryRoot 'src\fg_connected_distortion_l3_l87\rtx_half_texture_probe.cu'
$cuda = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3' }
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
$binary = Join-Path $attempt 'half-texture-probe.exe'
$output = Join-Path $attempt 'sm120-rg16f-texture-to-f32.bin'
$log = Join-Path $attempt 'compile.log'
$command = '"{0}" >nul && "{1}" -std=c++20 -arch=sm_120 -o "{2}" "{3}" >"{4}" 2>&1' -f $vcvars, (Join-Path $cuda 'bin\nvcc.exe'), $binary, $source, $log
& $env:ComSpec /d /s /c $command
if ($LASTEXITCODE) { throw "texture probe build failed: $log" }
& $binary $output
if ($LASTEXITCODE) { throw 'texture probe execution failed' }
Write-Host "output_sha256=$((Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash.ToLowerInvariant())"
Write-Host 'result=pass'
