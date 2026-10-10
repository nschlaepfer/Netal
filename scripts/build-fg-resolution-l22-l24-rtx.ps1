[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$JobRoot,
  [string]$AttemptId='oracle-v1',
  [string]$RepositoryRoot,
  [string]$BaseCommit='a9af06a3b46812e8fdbc239c937ab79e4361e463'
)
$ErrorActionPreference='Stop'
if(-not $RepositoryRoot){$RepositoryRoot=Join-Path $JobRoot 'repo'}
$JobRoot=[IO.Path]::GetFullPath($JobRoot)
$RepositoryRoot=[IO.Path]::GetFullPath($RepositoryRoot)
if(-not $JobRoot.StartsWith('D:\CodexLab\',[StringComparison]::OrdinalIgnoreCase)){
  throw "JobRoot outside D:\CodexLab: $JobRoot"
}
$source=Join-Path $RepositoryRoot 'src\fg_resolution_preprocess_l22_l24\rtx_oracle.cpp'
$contract=Join-Path $RepositoryRoot 'src\fg_resolution_preprocess_l22_l24\contract.h'
$baseSource=Join-Path $RepositoryRoot 'src\fg_mid_l52\rtx_oracle.cpp'
$harness=Join-Path $RepositoryRoot 'scripts\build-fg-resolution-l22-l24-rtx.ps1'
$module22=Join-Path $JobRoot 'inputs\cu-module-000022.bin'
$module24=Join-Path $JobRoot 'inputs\cu-module-000024.bin'
$attempt=Join-Path $JobRoot "output\$AttemptId"
$build=Join-Path $attempt 'build'
foreach($file in @($source,$contract,$baseSource,$harness,$module22,$module24)){
  if(-not(Test-Path -LiteralPath $file -PathType Leaf)){throw "Missing $file"}
}
if(Test-Path -LiteralPath $attempt){throw "Attempt exists: $attempt"}
if((Get-Item -LiteralPath $module22).Length -ne 15120 -or
   (Get-FileHash -LiteralPath $module22 -Algorithm SHA256).Hash -ne
     '7D6602BBED88409E9AD2EF71293869DD22F7153F6387339ED65E91C12341C67A'){
  throw 'module22 identity mismatch'
}
if((Get-Item -LiteralPath $module24).Length -ne 25880 -or
   (Get-FileHash -LiteralPath $module24 -Algorithm SHA256).Hash -ne
     '5A41F9D65D1F6F933AE42F36B493F60687A646B73929291BAFBE9D00297B6BDF'){
  throw 'module24 identity mismatch'
}
$vswhere=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$install=& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vcvars=Join-Path $install 'VC\Auxiliary\Build\vcvars64.bat'
New-Item -ItemType Directory -Path $build | Out-Null
$env:TEMP=$build;$env:TMP=$build
$exe=Join-Path $build 'fg-resolution-l22-l24-rtx.exe'
$obj=Join-Path $build 'probe.obj'
$log=Join-Path $build 'compile.log'
$started=(Get-Date).ToUniversalTime().ToString('o')
$command='"{0}" >nul && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /Fo"{1}" /c "{2}" >"{3}" 2>&1 && link.exe /nologo /OUT:"{4}" "{1}" bcrypt.lib d3d12.lib dxgi.lib >>"{3}" 2>&1' -f $vcvars,$obj,$source,$log,$exe
& $env:ComSpec /d /s /c $command
if($LASTEXITCODE -ne 0){throw "build failed: $log"}
$receipts=@()
foreach($resolution in @('720p','1080p','1440p','4k')){
  foreach($phase in @('reset','generated')){
    foreach($poison in @('a5','5a')){
      foreach($ordinal in @('a','b')){
        $id="$resolution-$phase-poison-$poison-process-$ordinal"
        $root=Join-Path $attempt $id
        New-Item -ItemType Directory -Path $root | Out-Null
        $process=Start-Process -FilePath $exe -ArgumentList @(
          $module22,$module24,$root,$id,$poison,$resolution,$phase
        ) -NoNewWindow -Wait -PassThru `
          -RedirectStandardOutput (Join-Path $root 'stdout.log') `
          -RedirectStandardError (Join-Path $root 'stderr.log')
        if($process.ExitCode -ne 0){throw "run $id exit=$($process.ExitCode)"}
        $receipt=Get-Content (Join-Path $root 'run-receipt.json') -Raw | ConvertFrom-Json
        if($receipt.result -ne 'pass' -or $receipt.input_mismatched_bytes -ne 0 -or
           $receipt.canary_mismatched_bytes -ne 0 -or
           $receipt.expected_or_captured_output_ingress -ne $false -or
           $receipt.archived_intermediate_ingress -ne $false){
          throw "receipt rejected $id"
        }
        $receipts += $receipt
      }
    }
  }
}
$pids=@($receipts | ForEach-Object {[int64]$_.process_id} | Sort-Object -Unique)
if($pids.Count -ne 32){throw "fresh process gate: $($pids.Count)/32"}
foreach($resolution in @('720p','1080p','1440p','4k')){
  foreach($phase in @('reset','generated')){
    foreach($poison in @(165,90)){
      $pair=@($receipts | Where-Object {
        $_.resolution -eq $resolution -and $_.phase -eq $phase -and
        $_.poison_byte -eq $poison
      })
      if($pair.Count -ne 2){throw "repeat pair missing: $resolution $phase $poison"}
      if(($pair.module24_output_sha256 | Sort-Object -Unique).Count -ne 1){
        throw "module24 repeat mismatch: $resolution $phase $poison"
      }
      for($index=0;$index -lt 4;$index++){
        $hashes=@($pair | ForEach-Object {$_.module22_output_sha256[$index]} | Sort-Object -Unique)
        if($hashes.Count -ne 1){
          throw "module22 repeat mismatch: $resolution $phase $poison level $index"
        }
      }
    }
  }
}
$completed=(Get-Date).ToUniversalTime().ToString('o')
$rig=[ordered]@{
  schema='neuralstudio.fg.resolution-l22-l24-rig.v1'
  job_id=Split-Path -Leaf $JobRoot
  attempt_id=$AttemptId
  local_base_commit=$BaseCommit
  started_utc=$started
  completed_utc=$completed
  gpu=(& nvidia-smi --query-gpu=name,driver_version,memory.total,pci.bus_id --format=csv,noheader,nounits | Select-Object -First 1).Trim()
  module22_sha256=(Get-FileHash $module22 -Algorithm SHA256).Hash.ToLowerInvariant()
  module24_sha256=(Get-FileHash $module24 -Algorithm SHA256).Hash.ToLowerInvariant()
  source_sha256=(Get-FileHash $source -Algorithm SHA256).Hash.ToLowerInvariant()
  contract_sha256=(Get-FileHash $contract -Algorithm SHA256).Hash.ToLowerInvariant()
  base_source_sha256=(Get-FileHash $baseSource -Algorithm SHA256).Hash.ToLowerInvariant()
  harness_sha256=(Get-FileHash $harness -Algorithm SHA256).Hash.ToLowerInvariant()
  binary_sha256=(Get-FileHash $exe -Algorithm SHA256).Hash.ToLowerInvariant()
  resolutions=@('720p','1080p','1440p','4k')
  phases=@('reset','generated')
  run_count=32
  fresh_process_count=32
  poison_bytes=@(165,90)
  true_d3d12_nvapi_resources=$true
  true_module24_output_format='DXGI_FORMAT_R16G16_FLOAT'
  input_immutable=$true
  buffer_canaries_intact=$true
  fresh_process_repeat_exact=$true
  expected_or_captured_output_ingress=$false
  archived_intermediate_ingress=$false
  address_fields_recorded=$false
  result='pass'
}
$utf8=New-Object System.Text.UTF8Encoding($false)
$receiptPath=Join-Path $attempt 'rig-receipt.json'
[IO.File]::WriteAllText($receiptPath,(($rig | ConvertTo-Json -Depth 8)+"`n"),$utf8)
Write-Host "attempt_root=$attempt"
Write-Host "rig_receipt_sha256=$((Get-FileHash $receiptPath -Algorithm SHA256).Hash.ToLowerInvariant())"
Write-Host 'result=pass'
