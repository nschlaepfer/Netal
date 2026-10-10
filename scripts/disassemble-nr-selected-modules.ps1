param([Parameter(Mandatory=$true)][string]$JobRoot)
$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue'
$JobRoot=[IO.Path]::GetFullPath($JobRoot)
if($JobRoot -notmatch '^D:\\CodexLab\\nr-module-sass-[0-9TZ]+$'){throw 'Invalid job path'}
$start=(Get-Date).ToUniversalTime().ToString('o')
$manifest=Get-Content -LiteralPath (Join-Path $JobRoot 'inputs.json') -Raw|ConvertFrom-Json
foreach($file in $manifest.files.PSObject.Properties){
  if($file.Name -match '[\\/]' -or $file.Name -eq '..'){throw 'Invalid file name'}
  if((Get-FileHash -LiteralPath (Join-Path $JobRoot $file.Name) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $file.Value){throw "Input hash mismatch $($file.Name)"}
}
New-Item -ItemType Directory -Path (Join-Path $JobRoot 'output'),(Join-Path $JobRoot 'build'),(Join-Path $JobRoot 'jit-cache')|Out-Null
$env:CUDA_CACHE_PATH=Join-Path $JobRoot 'jit-cache'
$ptxas=Join-Path $env:CUDA_PATH 'bin\ptxas.exe';$nvdisasm=Join-Path $env:CUDA_PATH 'bin\nvdisasm.exe';$dump=Join-Path $env:CUDA_PATH 'bin\cuobjdump.exe'
$vswhere=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vs=& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vcvars=Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat';$nvcc=Join-Path $env:CUDA_PATH 'bin\nvcc.exe'
$source=Join-Path $JobRoot 'jit_cubin.cu';$exe=Join-Path $JobRoot 'build\jit_cubin.exe'
$build='"{0}" >nul && "{1}" -std=c++17 -O2 -arch=sm_120 "{2}" -lcuda -o "{3}"' -f $vcvars,$nvcc,$source,$exe
Push-Location (Join-Path $JobRoot 'build')
try{& $env:ComSpec /d /s /c $build *> (Join-Path $JobRoot 'output\host-compile.log');if($LASTEXITCODE -ne 0){throw 'Host compile failed'}}finally{Pop-Location}
$commands=@()
$results=@()
foreach($module in @('module-00','module-01')){
  $ptx=Join-Path $JobRoot "$module.ptx";$cubin=Join-Path $JobRoot "output\$module.cubin"
  $commands += "$exe $ptx $cubin"
  & $exe $ptx $cubin *> (Join-Path $JobRoot "output\$module.jit.log")
  $jitExit=$LASTEXITCODE
  if($jitExit -eq 0){
    & $nvdisasm $cubin *> (Join-Path $JobRoot "output\$module.nvdisasm.txt");if($LASTEXITCODE -ne 0){throw "nvdisasm failed $module"}
    & $dump --dump-sass $cubin *> (Join-Path $JobRoot "output\$module.sass.txt");if($LASTEXITCODE -ne 0){throw "cuobjdump failed $module"}
  }
  $text=[IO.File]::ReadAllText($ptx)
  if(([regex]::Matches($text,'\.version 9\.4')).Count -ne 1){throw 'Unexpected version header'}
  $compat=Join-Path $JobRoot "output\$module.compat93.ptx";$compatCubin=Join-Path $JobRoot "output\$module.compat93.cubin"
  [IO.File]::WriteAllText($compat,$text.Replace('.version 9.4','.version 9.3'),[Text.UTF8Encoding]::new($false))
  $commands += "$ptxas -arch=sm_120 -v $compat -o $compatCubin"
  & $ptxas -arch=sm_120 -v $compat -o $compatCubin *> (Join-Path $JobRoot "output\$module.compat93.compile.log")
  $compatExit=$LASTEXITCODE
  if($compatExit -eq 0){
    & $nvdisasm $compatCubin *> (Join-Path $JobRoot "output\$module.compat93.nvdisasm.txt");if($LASTEXITCODE -ne 0){throw 'compat disassembly failed'}
    & $dump --dump-sass $compatCubin *> (Join-Path $JobRoot "output\$module.compat93.sass.txt");if($LASTEXITCODE -ne 0){throw 'compat dump failed'}
  }
  $results += [ordered]@{module=$module;driver_jit_exit=$jitExit;compatibility93_exit=$compatExit}
}
$inventory=@(Get-ChildItem -LiteralPath (Join-Path $JobRoot 'output') -File|ForEach-Object{
  [ordered]@{name=$_.Name;bytes=$_.Length;sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
})
[ordered]@{schema='neuralstudio.nr.selected-modules-offline-sass.v1';start_utc=$start;end_utc=(Get-Date).ToUniversalTime().ToString('o');
  commands=$commands;inputs_root_sha256=$manifest.receipt_root_sha256;compiler=(& $ptxas --version);disassembler=(& $nvdisasm --version);
  gpu_executed=$false;live_jit_sass_identity_proven=$false;module_results=$results;artifacts=$inventory;result='inspection-complete'}|
  ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $JobRoot 'receipt.json') -Encoding UTF8
