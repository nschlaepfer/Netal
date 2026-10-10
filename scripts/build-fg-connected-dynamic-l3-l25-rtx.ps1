[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$JobRoot,
  [Parameter(Mandatory=$true)][string]$RepositoryRoot,
  [string]$AttemptId='oracle-v1',
  [switch]$Smoke,
  [ValidateSet('720p','1080p','1440p','4k')][string]$SmokeResolution='720p'
)
$ErrorActionPreference='Stop'
$JobRoot=[IO.Path]::GetFullPath($JobRoot)
$RepositoryRoot=[IO.Path]::GetFullPath($RepositoryRoot)
if(-not $JobRoot.StartsWith('D:\CodexLab\',[StringComparison]::OrdinalIgnoreCase)){
  throw 'D: job required'
}
$g21=Join-Path $RepositoryRoot 'scripts\generate-fg-connected-dynamic-l3-l21.py'
$g23=Join-Path $RepositoryRoot 'scripts\generate-fg-connected-dynamic-l3-l23.py'
$g25=Join-Path $RepositoryRoot 'scripts\generate-fg-connected-dynamic-l3-l25.py'
$harness=Join-Path $RepositoryRoot 'scripts\build-fg-connected-dynamic-l3-l25-rtx.ps1'
$modules=Join-Path $JobRoot 'inputs\modules'
$attempt=Join-Path $JobRoot "output\$AttemptId"
$build=Join-Path $attempt 'build'
foreach($path in @($g21,$g23,$g25,$harness)){
  if(-not(Test-Path -LiteralPath $path -PathType Leaf)){throw "missing $path"}
}
if(Test-Path $attempt){throw 'attempt exists'}
$identities=@{
  'default-token-000063.bin'='44595b02307a6b934a124abe476cfd327440e2acba7f5ed69f73195ca743e3d9'
  'default-token-000065.bin'='eb99328524fddc9f5f3fac12b8e3a4011e66a75e071cd07126a70c553458bc60'
  'default-token-000067.bin'='4191924ff335d5018987f2526f933cb4cdbb9145411e3ce34f63c8206321be55'
  'cu-module-000030.bin'='31bb08d04011db9a0def8f1a1bc3999efc08d5deb8d7cb9b5debf52ec489b785'
  'cu-module-000031.bin'='e925f51165408fae4657d50247b10304802d46979b863712aeff844a217ade63'
}
foreach($entry in $identities.GetEnumerator()){
  $path=Join-Path $modules $entry.Key
  if((Get-FileHash $path -Algorithm SHA256).Hash.ToLowerInvariant()-ne$entry.Value){
    throw "identity $($entry.Key)"
  }
}
$started=(Get-Date).ToUniversalTime().ToString('o')
New-Item -ItemType Directory -Path $build|Out-Null
& python $g21 --repository $RepositoryRoot --output (Join-Path $build 'base21')
if($LASTEXITCODE){throw 'g21'}
& python $g23 --base (Join-Path $build 'base21') --output (Join-Path $build 'base23')
if($LASTEXITCODE){throw 'g23'}
& python $g25 --base (Join-Path $build 'base23') --output (Join-Path $build 'generated')
if($LASTEXITCODE){throw 'g25'}
$source=Join-Path $build 'generated\generated_rtx.cpp'
$cuda=if($env:CUDA_PATH){$env:CUDA_PATH}else{'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'}
$vs=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$install=&$vs -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vc=Join-Path $install 'VC\Auxiliary\Build\vcvars64.bat'
$env:TEMP=$build
$env:TMP=$build
$exe=Join-Path $build 'host.exe'
$obj=Join-Path $build 'host.obj'
$log=Join-Path $build 'compile.log'
$inc0=Join-Path $RepositoryRoot 'src'
$inc1=Join-Path $RepositoryRoot 'src\fg_connected_dynamic_early'
$inc2=Join-Path $RepositoryRoot 'src\fg_connected_dynamic_l3_l21'
$cmd='"{0}" >nul && cl /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /I"{1}" /I"{2}" /I"{3}" /Fo"{4}" /c "{5}" >"{6}" 2>&1 && link /nologo /OUT:"{7}" "{4}" bcrypt.lib d3d12.lib dxgi.lib >>"{6}" 2>&1'-f$vc,$inc0,$inc1,$inc2,$obj,$source,$log,$exe
&$env:ComSpec /d /s /c $cmd
if($LASTEXITCODE){throw "compile $log"}
$resolutions=if($Smoke){@($SmokeResolution)}else{@('720p','1080p','1440p','4k')}
$poisons=if($Smoke){@('a5')}else{@('a5','5a')}
$ordinals=if($Smoke){@('a')}else{@('a','b')}
$receipts=@()
foreach($resolution in $resolutions){
  foreach($poison in $poisons){
    foreach($ordinal in $ordinals){
      $id="$resolution-poison-$poison-process-$ordinal"
      $directory=Join-Path $attempt $id
      mkdir $directory|Out-Null
      $artifact=if($poison-eq'a5'-and$ordinal-eq'a'){'full'}else{'hash-only'}
      $process=Start-Process $exe -ArgumentList @($modules,$directory,$id,$poison,$resolution,$artifact) -Wait -PassThru -NoNewWindow -RedirectStandardOutput (Join-Path $directory 'stdout.log') -RedirectStandardError (Join-Path $directory 'stderr.log')
      if($process.ExitCode){throw "run $id"}
      $receipts+=Get-Content (Join-Path $directory 'run-receipt.json') -Raw|ConvertFrom-Json
    }
  }
}
$count=if($Smoke){1}else{16}
if(@($receipts.process_id|Sort-Object -Unique).Count-ne$count){throw 'pids'}
if(-not $Smoke){
  foreach($resolution in $resolutions){
    $same=@($receipts|Where-Object{$_.resolution -eq $resolution})
    foreach($name in $same[0].checkpoint_sha256.PSObject.Properties.Name){
      if(@($same|ForEach-Object{$_.checkpoint_sha256.$name}|Sort-Object -Unique).Count-ne 1){
        throw "drift $resolution $name"
      }
    }
  }
}
$rig=[ordered]@{
  schema='neuralstudio.fg.connected-dynamic-l3-l25-rig.v1'
  job_id=Split-Path -Leaf $JobRoot
  attempt_id=$AttemptId
  smoke=[bool]$Smoke
  started_utc=$started
  completed_utc=(Get-Date).ToUniversalTime().ToString('o')
  exit_code=0
  gpu=(& nvidia-smi --query-gpu=name,driver_version,memory.total,pci.bus_id --format=csv,noheader,nounits|Select-Object -First 1).Trim()
  generator21_sha256=(Get-FileHash $g21 -Algorithm SHA256).Hash.ToLowerInvariant()
  generator23_sha256=(Get-FileHash $g23 -Algorithm SHA256).Hash.ToLowerInvariant()
  generator25_sha256=(Get-FileHash $g25 -Algorithm SHA256).Hash.ToLowerInvariant()
  generated_source_sha256=(Get-FileHash $source -Algorithm SHA256).Hash.ToLowerInvariant()
  tail25_sha256=(Get-FileHash -LiteralPath (Join-Path $RepositoryRoot 'src\fg_connected_dynamic_l3_l25\rtx_tail.inc') -Algorithm SHA256).Hash.ToLowerInvariant()
  harness_sha256=(Get-FileHash $harness -Algorithm SHA256).Hash.ToLowerInvariant()
  weight67_sha256=(Get-FileHash (Join-Path $modules 'default-token-000067.bin') -Algorithm SHA256).Hash.ToLowerInvariant()
  module31_sha256=(Get-FileHash (Join-Path $modules 'cu-module-000031.bin') -Algorithm SHA256).Hash.ToLowerInvariant()
  run_count=$receipts.Count
  fresh_process_count=@($receipts.process_id|Sort-Object -Unique).Count
  selected_launches_per_process=27
  live_device_handoffs=$true
  expected_or_archived_intermediate_ingress=$false
  address_fields_recorded=$false
  result='pass'
}
$utf8=New-Object System.Text.UTF8Encoding($false)
$receipt_path=Join-Path $attempt 'rig-receipt.json'
[IO.File]::WriteAllText($receipt_path,(($rig|ConvertTo-Json -Depth 6)+"`n"),$utf8)
Write-Host "attempt_root=$attempt"
Write-Host result=pass
