param(
    [Parameter(Mandatory = $true)][string]$JobRoot,
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [Parameter(Mandatory = $true)][string]$Attempt
)
$ErrorActionPreference = 'Stop'
$output = Join-Path $JobRoot "output\$Attempt"
if (Test-Path -LiteralPath $output) { throw "attempt exists: $output" }
New-Item -ItemType Directory -Path $output | Out-Null
$source = Join-Path $RepositoryRoot 'src\fg_distortion_path\connected_rtx.cpp'
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
$object = Join-Path $output 'connected.obj'
$binary = Join-Path $output 'connected.exe'
$log = Join-Path $output 'compile.log'
$command = '"{0}" >nul && cl /nologo /std:c++20 /EHsc /MD /W3 /permissive- /Fo"{1}" /c "{2}" >"{4}" 2>&1 && link /nologo /OUT:"{3}" "{1}" bcrypt.lib d3d12.lib dxgi.lib >>"{4}" 2>&1' -f $vcvars, $object, $source, $binary, $log
& $env:ComSpec /d /s /c $command
if ($LASTEXITCODE) { throw "connected build failed: $log" }
$module10 = Join-Path $JobRoot 'input\a5-trace\dumps\cu-module-000010.bin'
$module9 = Join-Path $JobRoot 'input\a5-trace\dumps\cu-module-000009.bin'
$module25 = Join-Path $JobRoot 'input\a5-trace\dumps\cu-module-000025.bin'
foreach ($poison in @('a5', '5a')) {
    foreach ($process in @('a', 'b')) {
        $runId = "distortion-six-poison-$poison-$process"
        $runRoot = Join-Path $output $runId
        New-Item -ItemType Directory -Path $runRoot | Out-Null
        $run = Start-Process $binary -ArgumentList @($module10, $module9, $module25, $runRoot, $runId, $poison) -Wait -PassThru -NoNewWindow
        if ($run.ExitCode) { throw "connected run failed: $runId" }
    }
}
Write-Host "attempt=$output"
Write-Host 'result=pass'
