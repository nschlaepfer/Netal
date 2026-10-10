param([Parameter(Mandatory=$true)][string]$JobRoot)
$ErrorActionPreference='Stop';$JobRoot=[IO.Path]::GetFullPath($JobRoot)
if(-not $JobRoot.StartsWith('D:\CodexLab\')){throw 'invalid job root'}
$cuobjdump='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\cuobjdump.exe';$rows=@()
foreach($token in @('000009','000010','000025')){
 $input=Join-Path $JobRoot "input\a5-trace\dumps\cu-module-$token.bin";$ptx=Join-Path $JobRoot "output\module-$token.ptx.txt";$sass=Join-Path $JobRoot "output\module-$token.sass.txt"
 & $cuobjdump --dump-ptx $input|Set-Content $ptx -Encoding UTF8;& $cuobjdump --dump-sass $input|Set-Content $sass -Encoding UTF8
 $rows+=[ordered]@{module_token=[int]$token;input_sha256=(Get-FileHash $input -Algorithm SHA256).Hash.ToLowerInvariant();ptx_sha256=(Get-FileHash $ptx -Algorithm SHA256).Hash.ToLowerInvariant();sass_sha256=(Get-FileHash $sass -Algorithm SHA256).Hash.ToLowerInvariant()}
}
$r=[ordered]@{schema='neuralstudio.fg.distortion-ptx.v1';job_id=(Split-Path -Leaf $JobRoot);modules=$rows;result='pass'};$r|ConvertTo-Json -Depth 6|Set-Content (Join-Path $JobRoot 'output\receipt.json') -Encoding UTF8;$r|ConvertTo-Json -Compress
