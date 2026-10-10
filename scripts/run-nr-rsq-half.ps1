param([Parameter(Mandatory=$true)][string]$JobRoot)
$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue'
$JobRoot=[IO.Path]::GetFullPath($JobRoot)
if(-not $JobRoot.StartsWith('D:\CodexLab\nr-rsq-half-')){throw 'Invalid job path'}
New-Item -ItemType Directory -Path (Join-Path $JobRoot 'build'),(Join-Path $JobRoot 'output')|Out-Null
$vswhere=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vs=& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vcvars=Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
$nvcc=Join-Path $env:CUDA_PATH 'bin\nvcc.exe'
$source=Join-Path $JobRoot 'rsq_half_oracle.cu';$exe=Join-Path $JobRoot 'build\probe.exe'
$command='"{0}" >nul && "{1}" -std=c++17 -O2 -arch=sm_120 "{2}" -o "{3}"' -f $vcvars,$nvcc,$source,$exe
Push-Location (Join-Path $JobRoot 'build')
try{& $env:ComSpec /d /s /c $command *> (Join-Path $JobRoot 'compile.log');if($LASTEXITCODE -ne 0){throw 'Compile failed'}}finally{Pop-Location}
foreach($poison in @('a5','5a')){foreach($ordinal in @('a','b')){
    $prefix=Join-Path $JobRoot "output\$poison-$ordinal"
    & $exe $prefix $poison *> "$prefix.log"
    if($LASTEXITCODE -ne 0){throw "Oracle failed $poison-$ordinal"}
}}
$inventory=@(Get-ChildItem -LiteralPath (Join-Path $JobRoot 'output') -File|ForEach-Object{
    [ordered]@{name=$_.Name;bytes=$_.Length;sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
})
[ordered]@{schema='neuralstudio.nr.rsq-half-job.v1';gpu=(& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader);
    utc=(Get-Date).ToUniversalTime().ToString('o');source_sha256=(Get-FileHash $source -Algorithm SHA256).Hash.ToLowerInvariant();
    compiler=(& $nvcc --version);executable_sha256=(Get-FileHash $exe -Algorithm SHA256).Hash.ToLowerInvariant();artifacts=$inventory;result='pass'}|
    ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $JobRoot 'receipt.json') -Encoding UTF8
