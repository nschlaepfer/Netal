<#
.SYNOPSIS
    Stage the exact development NGX adapter package and local provider runtime.

.DESCRIPTION
    Custom NeuralStudio artifacts are copied into the source package so their
    manifest hashes remain verifiable. NVIDIA-signed provider files are linked
    only into the disposable build package and remain governed by
    backend.runtimeAcquisition; they are never declared redistributable.
#>
[CmdletBinding()]
param([switch]$SkipBuild)

$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent $PSScriptRoot
$sourcePackage = Join-Path $repository 'packages\com.nvidia.ngx.dlss-nr'
$sourceRuntime = Join-Path $sourcePackage 'runtime'
$buildPackage = Join-Path $repository 'build\runtime-v0\packages\com.nvidia.ngx.dlss-nr'

if (-not $SkipBuild) {
    & (Join-Path $PSScriptRoot 'build-dlssnr-bridge-worker.ps1')
    & (Join-Path $PSScriptRoot 'build-ngx-adapter.ps1')
}

$custom = @(
    @{ Source = (Join-Path $repository 'build\dlssnr_bridge_worker\neuralstudio-dlssnr-bridge-worker.exe'); Relative = 'runtime\neuralstudio-dlssnr-bridge-worker.exe'; Hash = '809E35620E14DF9F9B7E0E4A7ADFDDEAF52E3749714544AF34B584E686300B9B' },
    @{ Source = (Join-Path $repository 'runtime\neuralstudio-dlssnr-worker\sl.interposer.dll'); Relative = 'runtime\sl.interposer.dll'; Hash = 'D7138D0D345569CA40FB36FADEAABB2269041BC85049D76D718328431E7EFE05' },
    @{ Source = (Join-Path $repository 'runtime\neuralstudio-dlssnr-worker\sl.common.dll'); Relative = 'runtime\sl.common.dll'; Hash = '5C144677DE65CCD9759292C6F73196E8C19F4BE423698DB2D73705DEED9F73BA' },
    @{ Source = (Join-Path $repository 'runtime\neuralstudio-dlssnr-worker\neuralstudio\nvngx.dll'); Relative = 'runtime\neuralstudio\nvngx.dll'; Hash = 'EF0F22BCDF0D8667CA762944B3AC4637B1B969DBF5F47D8671A9B085F064DF40' }
)
$adapter = Join-Path $sourcePackage 'ns_adapter_ngx.dll'
if ((Get-FileHash -LiteralPath $adapter -Algorithm SHA256).Hash -ne '4C8EC3C47F7D0F2109AB11D97443A615D9C2EC0A10608D938F2E038DD372D2F1') {
    throw 'NGX adapter hash does not match package.json.'
}
foreach ($item in $custom) {
    if ((Get-FileHash -LiteralPath $item.Source -Algorithm SHA256).Hash -ne $item.Hash) {
        throw "Custom runtime hash mismatch: $($item.Source)"
    }
    $destination = Join-Path $sourcePackage $item.Relative
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    Copy-Item -LiteralPath $item.Source -Destination $destination -Force
}

$buildRoot = [IO.Path]::GetFullPath((Join-Path $repository 'build\runtime-v0'))
$resolvedTarget = [IO.Path]::GetFullPath($buildPackage)
if (-not $resolvedTarget.StartsWith($buildRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Refusing to stage outside build\runtime-v0.'
}
if (Test-Path -LiteralPath $buildPackage) { Remove-Item -LiteralPath $buildPackage -Recurse -Force }
New-Item -ItemType Directory -Path (Split-Path -Parent $buildPackage) -Force | Out-Null
Copy-Item -LiteralPath $sourcePackage -Destination $buildPackage -Recurse

$provider = @(
    @{ Source = (Join-Path $repository 'runtime\streamline-2.13\sl.dlss_nr.dll'); Name = 'sl.dlss_nr.dll'; Hash = '9F6672E5E0170DC118A3188D21BDA187E1FC1AA3502895B21AB846D23165C11D' },
    @{ Source = (Join-Path $repository 'models\nvidia\nvngx_dlssnr.dll'); Name = 'nvngx_dlssnr.dll'; Hash = 'E16BCF15E16E13F527491CDF7845B2FE6521A738D8F7C9C721866A8496E1FC8E' }
)
foreach ($item in $provider) {
    if ((Get-FileHash -LiteralPath $item.Source -Algorithm SHA256).Hash -ne $item.Hash) { throw "Provider runtime hash mismatch: $($item.Source)" }
    $signature = Get-AuthenticodeSignature -LiteralPath $item.Source
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notlike '*NVIDIA Corporation*') { throw "Provider signature rejected: $($item.Source)" }
    $destination = Join-Path (Join-Path $buildPackage 'runtime') $item.Name
    New-Item -ItemType HardLink -Path $destination -Target $item.Source | Out-Null
}

Write-Host "SourcePackage=$sourcePackage"
Write-Host "RuntimePackage=$buildPackage"
Write-Host 'provider_runtime.staged_as_hardlinks=true'
