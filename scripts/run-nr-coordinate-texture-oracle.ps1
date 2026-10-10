[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$JobRoot,[Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ExpectedSourceSha,[ValidateSet('nr_coordinate_texture_oracle.cu','nr_sfu_pair_oracle.cu','nr_sfu_scaled_half_oracle.cu','nr_finite_half_texture_oracle.cu')][string]$SourceName='nr_coordinate_texture_oracle.cu',[switch]$Build,[ValidateSet('a5','5a')][string]$Poison='a5',[ValidatePattern('^[a-z0-9-]+$')][string]$Label='a5-a')
$ErrorActionPreference='Stop';$JobRoot=[IO.Path]::GetFullPath($JobRoot).TrimEnd('\')
if(-not $JobRoot.StartsWith('D:\CodexLab\nr-tex-micro-')){throw 'isolated D job required'}
$source=$JobRoot+'\source\'+$SourceName;$buildDir=$JobRoot+'\build';$binary=$buildDir+'\oracle.exe'
if((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ExpectedSourceSha){throw 'source hash mismatch'}
$env:TEMP=$JobRoot+'\temp';$env:TMP=$env:TEMP;$env:CUDA_CACHE_PATH=$JobRoot+'\cache'
if($Build){
  if(Test-Path -LiteralPath $buildDir){throw 'preserve build'}
  New-Item -ItemType Directory -Path $buildDir,$env:TEMP,$env:CUDA_CACHE_PATH | Out-Null
  $vswhere=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
  $vs=& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
  $vcvars=Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
  $command='"{0}" >nul && nvcc -std=c++17 -arch=sm_120a -O3 --fmad=false -lineinfo --keep "{1}" -o "{2}"' -f $vcvars,$source,$binary
  Push-Location $buildDir;try{& $env:ComSpec /d /s /c $command *> ($buildDir+'\compile.log');$code=$LASTEXITCODE}finally{Pop-Location}
  $r=[ordered]@{source_sha256=$ExpectedSourceSha;command=$command;exit_code=$code;gpu=(& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader);cuda=(& nvcc --version|Select-Object -Last 1)}
  if($code -eq 0){$r.binary_sha256=(Get-FileHash $binary -Algorithm SHA256).Hash.ToLowerInvariant();& cuobjdump --dump-sass $binary *> ($buildDir+'\sass.txt')}
  $r|ConvertTo-Json -Depth 6|Set-Content ($JobRoot+'\build.json') -Encoding UTF8;$r|ConvertTo-Json -Depth 6
  if($code -ne 0){throw 'compile failed; retained'};exit 0
}
$b=Get-Content ($JobRoot+'\build.json') -Raw|ConvertFrom-Json
if($b.exit_code -ne 0 -or $b.source_sha256 -ne $ExpectedSourceSha -or (Get-FileHash $binary -Algorithm SHA256).Hash.ToLowerInvariant() -ne $b.binary_sha256){throw 'unverified binary'}
if((Get-PSDrive D).Free -lt 10GB){throw 'storage floor'}
$out=$JobRoot+'\'+$Label;if(Test-Path $out){throw 'preserve prior run'}
$start=(Get-Date).ToUniversalTime().ToString('o');$before=(& nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader)
& $binary $Poison $out *> ($JobRoot+'\'+$Label+'.log');$code=$LASTEXITCODE
$r=[ordered]@{schema='neuralstudio.nr.coordinate-texture-job.v1';source_sha256=$ExpectedSourceSha;binary_sha256=$b.binary_sha256;started_utc=$start;ended_utc=(Get-Date).ToUniversalTime().ToString('o');command=@($binary,$Poison,$out);exit_code=$code;gpu_before=$before;gpu_after=(& nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader);files=@{}}
if(Test-Path $out){Get-ChildItem $out -File|ForEach-Object{$r.files[$_.Name]=[ordered]@{bytes=$_.Length;sha256=(Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}}}
$r|ConvertTo-Json -Depth 8|Set-Content ($JobRoot+'\'+$Label+'-receipt.json') -Encoding UTF8;$r|ConvertTo-Json -Depth 8
if($code -ne 0){throw 'oracle failed; retained'}
