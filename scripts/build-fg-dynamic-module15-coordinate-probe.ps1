[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$JobRoot,[string]$RepositoryRoot)
$ErrorActionPreference='Stop'
if(-not $RepositoryRoot){$RepositoryRoot=Join-Path $JobRoot 'repo'}
$JobRoot=[IO.Path]::GetFullPath($JobRoot);$RepositoryRoot=[IO.Path]::GetFullPath($RepositoryRoot)
if(-not $JobRoot.StartsWith('D:\CodexLab\',[StringComparison]::OrdinalIgnoreCase)){throw'job root'}
$source=Join-Path $RepositoryRoot 'src\fg_dynamic_module15\rtx_coordinate_probe.cu'
$out=Join-Path $JobRoot 'output\coordinate-probe-v1';if(Test-Path $out){throw'output exists'};New-Item -ItemType Directory -Path $out|Out-Null
$cuda=if($env:CUDA_PATH){$env:CUDA_PATH}else{'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'}
$vswhere=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe';$install=&$vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath;$vcvars=Join-Path $install 'VC\Auxiliary\Build\vcvars64.bat'
$exe=Join-Path $out 'probe.exe';$log=Join-Path $out 'compile.log';$data=Join-Path $out 'samples.bin'
$cmd='"{0}" >nul && "{1}" -std=c++17 -arch=sm_120 -o "{2}" "{3}" >"{4}" 2>&1'-f$vcvars,(Join-Path $cuda 'bin\nvcc.exe'),$exe,$source,$log
&$env:ComSpec /d /s /c $cmd;if($LASTEXITCODE-ne0){throw'compile failed'};&$exe $data;if($LASTEXITCODE-ne0){throw'probe failed'}
$receipt=[ordered]@{schema='neuralstudio.fg.dynamic-module15-coordinate-probe.v1';source_sha256=(Get-FileHash $source -Algorithm SHA256).Hash.ToLowerInvariant();binary_sha256=(Get-FileHash $exe -Algorithm SHA256).Hash.ToLowerInvariant();samples_bytes=(Get-Item $data).Length;samples_sha256=(Get-FileHash $data -Algorithm SHA256).Hash.ToLowerInvariant();resolution='720p';execution_ingress=$false;result='pass'};$utf8=New-Object Text.UTF8Encoding($false);[IO.File]::WriteAllText((Join-Path $out 'receipt.json'),(($receipt|ConvertTo-Json -Depth 4)+"`n"),$utf8);Write-Host "output=$out";Write-Host result=pass
