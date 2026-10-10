param([Parameter(Mandatory=$true)][string]$JobRoot,[Parameter(Mandatory=$true)][string]$RepositoryRoot,[Parameter(Mandatory=$true)][string]$Attempt)
$ErrorActionPreference='Stop'
$out=Join-Path $JobRoot "output\$Attempt";if(Test-Path $out){throw 'exists'};New-Item -ItemType Directory $out|Out-Null
$src=Join-Path $RepositoryRoot 'src\fg_distortion_path\module9_rtx.cpp';$vs=& "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath;$vc=Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat';$obj=Join-Path $out 'm9.obj';$exe=Join-Path $out 'm9.exe';$log=Join-Path $out 'compile.log'
$cmd='"{0}" >nul && cl /nologo /std:c++20 /EHsc /MD /W3 /permissive- /Fo"{1}" /c "{2}" >"{4}" 2>&1 && link /nologo /OUT:"{3}" "{1}" bcrypt.lib d3d12.lib dxgi.lib >>"{4}" 2>&1' -f $vc,$obj,$src,$exe,$log;& $env:ComSpec /d /s /c $cmd;if($LASTEXITCODE){throw "build $log"}
$mod=Join-Path $JobRoot 'input\a5-trace\dumps\cu-module-000009.bin';foreach($p in @('a5','5a')){foreach($o in @('a','b')){$id="m9-poison-$p-$o";$d=Join-Path $out $id;New-Item -ItemType Directory $d|Out-Null;$x=Start-Process $exe -ArgumentList @($mod,$d,$id,$p) -Wait -PassThru -NoNewWindow;if($x.ExitCode){throw "run $id"}}}
Write-Host "attempt=$out";Write-Host 'result=pass'
