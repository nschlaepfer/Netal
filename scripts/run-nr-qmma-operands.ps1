param([Parameter(Mandatory=$true)][string]$JobRoot)
$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue'
$JobRoot=[IO.Path]::GetFullPath($JobRoot)
if($JobRoot -notmatch '^D:\\CodexLab\\nr-qmma-operands-[0-9TZ]+$'){throw 'Invalid job path'}
$start=(Get-Date).ToUniversalTime().ToString('o')
$manifest=Get-Content -LiteralPath (Join-Path $JobRoot 'inputs.json') -Raw|ConvertFrom-Json
foreach($file in $manifest.files.PSObject.Properties){
  if($file.Name -match '[\\/]' -or $file.Name -eq '..'){throw 'Invalid file name'}
  if((Get-FileHash -LiteralPath (Join-Path $JobRoot $file.Name) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $file.Value){throw "Input hash mismatch $($file.Name)"}
}
New-Item -ItemType Directory -Path (Join-Path $JobRoot 'build'),(Join-Path $JobRoot 'output')|Out-Null
$vswhere=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vs=& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vcvars=Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat';$nvcc=Join-Path $env:CUDA_PATH 'bin\nvcc.exe'
$source=Join-Path $JobRoot 'qmma_operands.cu';$exe=Join-Path $JobRoot 'build\probe.exe'
$cases=if($manifest.cases){[int]$manifest.cases}else{1}
if($cases -lt 1 -or $cases -gt 1024){throw 'Invalid case count'}
$command='"{0}" >nul && "{1}" -std=c++17 -O2 -arch=sm_120 -DNR_MMA_CASES={4} "{2}" -o "{3}"' -f $vcvars,$nvcc,$source,$exe,$cases
Push-Location (Join-Path $JobRoot 'build')
try{& $env:ComSpec /d /s /c $command *> (Join-Path $JobRoot 'compile.log');if($LASTEXITCODE -ne 0){throw 'Compile failed'}}finally{Pop-Location}
& (Join-Path $env:CUDA_PATH 'bin\cuobjdump.exe') --dump-sass $exe *> (Join-Path $JobRoot 'sass.txt')
if($LASTEXITCODE -ne 0){throw 'SASS dump failed'}
foreach($poison in @('a5','5a')){foreach($ordinal in @('a','b')){
  $prefix=Join-Path $JobRoot "output\$poison-$ordinal"
  & $exe (Join-Path $JobRoot 'operands.bin') $prefix $poison *> "$prefix.log"
  if($LASTEXITCODE -ne 0){throw "Execution failed $poison-$ordinal"}
}}
$inventory=@(Get-ChildItem -LiteralPath (Join-Path $JobRoot 'output') -File|ForEach-Object{
  [ordered]@{name=$_.Name;bytes=$_.Length;sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
})
[ordered]@{schema='neuralstudio.nr.qmma-operands-job.v1';gpu=(& nvidia-smi --query-gpu=name,driver_version,uuid --format=csv,noheader);
  start_utc=$start;end_utc=(Get-Date).ToUniversalTime().ToString('o');compile_command=$command;inputs_root_sha256=$manifest.receipt_root_sha256;
  compiler=(& $nvcc --version);executable_sha256=(Get-FileHash $exe -Algorithm SHA256).Hash.ToLowerInvariant();
  sass_sha256=(Get-FileHash (Join-Path $JobRoot 'sass.txt') -Algorithm SHA256).Hash.ToLowerInvariant();artifacts=$inventory;result='pass'}|
  ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $JobRoot 'receipt.json') -Encoding UTF8
