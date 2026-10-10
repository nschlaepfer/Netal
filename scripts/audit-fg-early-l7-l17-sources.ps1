[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$JobRoot)
$ErrorActionPreference='Stop'
$JobRoot=[IO.Path]::GetFullPath($JobRoot)
if(-not $JobRoot.StartsWith('D:\CodexLab\',[StringComparison]::OrdinalIgnoreCase)){throw "JobRoot outside D:\CodexLab: $JobRoot"}
$out=Join-Path $JobRoot 'output\source-audit-v1'
if(Test-Path -LiteralPath $out){throw "Source audit exists: $out"}
New-Item -ItemType Directory -Path $out|Out-Null
$paths=[ordered]@{
 unit_source=Join-Path $JobRoot 'repo\src\fg_early_l7_l17\rtx_oracle.cpp'
 contract=Join-Path $JobRoot 'repo\src\fg_early_l7_l17\contract.h'
 shared_cuda_helper=Join-Path $JobRoot 'repo\src\fg_conv\rtx_oracle.cpp'
 build_harness=Join-Path $JobRoot 'repo\scripts\build-fg-early-l7-l17-rtx.ps1'
 module=Join-Path $JobRoot 'inputs\cu-module-000023.bin'
 binary=Join-Path $JobRoot 'output\oracle-v2\build\fg-early-l7-l17-rtx.exe'
}
$expected=[ordered]@{
 unit_source='45c43c799a2987e62fb2c272821405221e1eec91ba2974623e9bb7bf78d5ddbc'
 contract='234c4f716ff1dd8927a9bb6128a4d48975e668c3e8d6ba8240d13d7aed003e10'
 shared_cuda_helper='a83039f524ae70f2c6bc86190c6c8634a2670a86ecbcd5525a055aade533a1b6'
 build_harness='0174b2df4174ca5959d372b842c6c22a224406cc1b7c260feb489c83bdee40bb'
 module='d787cf7b861295d1b4c6a80cb374230de793aa69f288a7e724bf59278bc4a62d'
 binary='67f46664fd9f101fd20162cbabbdfc7481efc0673e2986f1663fe3c4704d2e47'
}
$files=[ordered]@{}
foreach($entry in $paths.GetEnumerator()){
 if(-not(Test-Path -LiteralPath $entry.Value -PathType Leaf)){throw "Missing audit input: $($entry.Value)"}
 $item=Get-Item -LiteralPath $entry.Value
 $hash=(Get-FileHash -LiteralPath $entry.Value -Algorithm SHA256).Hash.ToLowerInvariant()
 if($hash-ne$expected[$entry.Key]){throw "Hash mismatch: $($entry.Key)"}
 $files[$entry.Key]=[ordered]@{bytes=[int64]$item.Length;sha256=$hash}
}
$receipt=[ordered]@{
 schema='neuralstudio.fg.early-l7-l17-source-audit.v1'
 job_id=Split-Path -Leaf $JobRoot
 audited_attempt='oracle-v2'
 captured_utc=(Get-Date).ToUniversalTime().ToString('o')
 gpu=(&nvidia-smi --query-gpu=name,driver_version --format=csv,noheader|Select-Object -First 1).Trim()
 files=$files
 shared_cuda_helper_omitted_from_original_rig_receipt=$true
 shared_cuda_helper_live_hash_recovered=$true
 address_fields_recorded=$false
 result='pass'
}
$utf8=New-Object System.Text.UTF8Encoding($false)
$path=Join-Path $out 'source-audit.json'
[IO.File]::WriteAllText($path,(($receipt|ConvertTo-Json -Depth 6)+"`n"),$utf8)
Write-Host "source_audit=$path"
Write-Host "source_audit_sha256=$((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant())"
Write-Host 'result=pass'
