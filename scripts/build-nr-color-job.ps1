[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$JobRoot)
$ErrorActionPreference = 'Stop'
$JobRoot = [IO.Path]::GetFullPath($JobRoot).TrimEnd('\')
if (-not $JobRoot.StartsWith('D:\CodexLab\nr-color-')) { throw 'Isolated D: color job required' }
$source = Join-Path $JobRoot 'source'
$manifest = Get-Content -LiteralPath (Join-Path $source 'source-receipt.json') -Raw | ConvertFrom-Json
foreach ($entry in $manifest.files.PSObject.Properties) {
  $path = Join-Path $source $entry.Name
  if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value) { throw "Source drift: $($entry.Name)" }
}
$build = Join-Path $JobRoot 'build'
if (Test-Path -LiteralPath $build) { throw 'Preserve prior build' }
New-Item -ItemType Directory -Path $build | Out-Null
$env:TEMP = Join-Path $JobRoot 'temp'
$env:TMP = $env:TEMP
$env:CUDA_CACHE_PATH = Join-Path $JobRoot 'cuda-cache'
New-Item -ItemType Directory -Path $env:TEMP,$env:CUDA_CACHE_PATH | Out-Null
$pins = @{
  'nvcuda.dll' = @('D:\NeuralStudio-RTX-Oracle\build\rtx_cuda_capture\nvcuda.dll','445df9c7b2b58eb65163d4470262fef9143044b5819cc095f6a761245ee22554')
  'nvngx.dll' = @('D:\NeuralStudio-RTX-Oracle\build\rtx_cuda_capture\nvngx.dll','d8bf063f27df70e304062ebd78ac03990b323c76725a80726770584ea5ca940c')
  'nvngx_dlssnr.dll' = @('D:\NeuralStudio-RTX-Oracle\models\nvidia\nvngx_dlssnr.dll','e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e')
}
foreach ($name in $pins.Keys) {
  if ((Get-FileHash -LiteralPath $pins[$name][0] -Algorithm SHA256).Hash.ToLowerInvariant() -ne $pins[$name][1]) { throw "Seed drift: $name" }
  Copy-Item -LiteralPath $pins[$name][0] -Destination (Join-Path $build $name)
}
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $installation) { throw 'MSVC missing' }
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
$binary = Join-Path $build 'color-cases.exe'
$command = '"{0}" >nul && cl.exe /nologo /std:c++20 /EHsc /MT /W4 /WX /DUNICODE /D_UNICODE /I"{1}" /Fo"{2}" "{3}" /link /OUT:"{4}"' -f $vcvars,(Join-Path $source 'include'),(Join-Path $build 'color-cases.obj'),(Join-Path $source 'color_cases_trace.cpp'),$binary
$started = (Get-Date).ToUniversalTime().ToString('o')
& $env:ComSpec /d /s /c $command *> (Join-Path $build 'compile.log')
$compileExit = $LASTEXITCODE
$record = [ordered]@{schema='neuralstudio.nr.color-build.v1';started_utc=$started;ended_utc=(Get-Date).ToUniversalTime().ToString('o');command=$command;exit_code=$compileExit;source_root=$manifest.receipt_root_sha256;source_receipt_sha256=(Get-FileHash -LiteralPath (Join-Path $source 'source-receipt.json') -Algorithm SHA256).Hash.ToLowerInvariant();gpu=(& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader);files=@{}}
if ($compileExit -eq 0) {
  foreach ($name in @('color-cases.exe','nvcuda.dll','nvngx.dll','nvngx_dlssnr.dll')) { $record.files[$name]=(Get-FileHash -LiteralPath (Join-Path $build $name) -Algorithm SHA256).Hash.ToLowerInvariant() }
}
$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $JobRoot 'build-receipt.json') -Encoding UTF8
if ($compileExit -ne 0) { Get-Content -LiteralPath (Join-Path $build 'compile.log'); throw "Compile failed: $compileExit" }
$record | ConvertTo-Json -Depth 8
