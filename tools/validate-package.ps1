<#
.SYNOPSIS
    Validate a NeuralStudio package manifest against the v1 contract.

.DESCRIPTION
    Implements the validation rules documented in spec/manifest/v1/README.md.

    Schema-shape checking is attempted with Test-Json when the host PowerShell
    supports it, but every substantive rule is also checked explicitly here, so
    the validator stays useful even when the JSON Schema engine cannot handle
    draft 2020-12 constructs.

    A manifest can be schema-valid and still be rejected. Hash mismatches,
    missing reset semantics, out-of-range defaults, and unauthorized
    redistribution are all failures.

.PARAMETER PackagePath
    Directory containing package.json, or a path to the manifest file itself.

.PARAMETER ReleaseGate
    Additionally enforce that every artifact is authorized for redistribution.
    Use this in release packaging. It is the mechanical enforcement of the v1
    rule that a release contains only artifacts NeuralStudio may redistribute.

.PARAMETER HostAbiMajor
    Adapter ABI major version this host speaks.

.EXAMPLE
    .\tools\validate-package.ps1 -PackagePath .\packages\org.neuralstudio.test.passthrough

.EXAMPLE
    Get-ChildItem .\packages -Directory | ForEach-Object {
        .\tools\validate-package.ps1 -PackagePath $_.FullName -ReleaseGate
    }
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PackagePath,

    [switch]$ReleaseGate,

    [int]$HostAbiMajor = 1,

    [switch]$AsJson
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repository = Split-Path -Parent $PSScriptRoot
$schemaPath = Join-Path $repository 'spec\manifest\v1\neuralstudio-package-v1.schema.json'

$results = [System.Collections.Generic.List[object]]::new()

function Add-Result {
    param(
        [Parameter(Mandatory = $true)][string]$Rule,
        [Parameter(Mandatory = $true)][ValidateSet('Pass', 'Fail', 'Warn', 'Skip')][string]$Status,
        [string]$Detail = ''
    )
    $results.Add([pscustomobject]@{ Rule = $Rule; Status = $Status; Detail = $Detail })
}

# Strict-mode-safe nested property read. Returns $null for any missing segment.
function Get-Prop {
    param($Object, [Parameter(Mandatory = $true)][string]$Path)
    $current = $Object
    foreach ($segment in $Path.Split('.')) {
        if ($null -eq $current) { return $null }
        if ($current -isnot [psobject]) { return $null }
        $property = $current.PSObject.Properties[$segment]
        if ($null -eq $property) { return $null }
        $current = $property.Value
    }
    return $current
}

function ConvertTo-Array {
    param($Value)
    if ($null -eq $Value) { return @() }
    if ($Value -is [System.Array]) { return $Value }
    return @($Value)
}

function Get-SafePackageChildPath {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    if ([string]::IsNullOrWhiteSpace($RelativePath) -or
        [IO.Path]::IsPathFullyQualified($RelativePath)) {
        return $null
    }

    try {
        $root = [IO.Path]::GetFullPath($packageRoot).TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
        $candidate = [IO.Path]::GetFullPath((Join-Path $root $RelativePath))
        $prefix = $root + [IO.Path]::DirectorySeparatorChar
        if (-not $candidate.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
            return $null
        }
        return $candidate
    } catch {
        return $null
    }
}

# --- locate the manifest ---------------------------------------------------

$resolved = (Resolve-Path -LiteralPath $PackagePath).Path
if (Test-Path -LiteralPath $resolved -PathType Container) {
    $manifestPath = Join-Path $resolved 'package.json'
    $packageRoot = $resolved
} else {
    $manifestPath = $resolved
    $packageRoot = Split-Path -Parent $resolved
}

if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "No package.json found at '$manifestPath'."
}

$manifestText = [IO.File]::ReadAllText($manifestPath)

# --- rule: manifest parses -------------------------------------------------

$manifest = $null
try {
    $manifest = $manifestText | ConvertFrom-Json
    Add-Result -Rule 'manifest.parses' -Status 'Pass' -Detail $manifestPath
} catch {
    Add-Result -Rule 'manifest.parses' -Status 'Fail' -Detail $_.Exception.Message
}

if ($null -ne $manifest) {

    # --- rule: schema version ----------------------------------------------

    $schemaVersion = Get-Prop $manifest 'schemaVersion'
    if ($schemaVersion -is [string] -and $schemaVersion -match '^1\.[0-9]+$') {
        Add-Result -Rule 'manifest.schemaVersion' -Status 'Pass' -Detail $schemaVersion
    } else {
        Add-Result -Rule 'manifest.schemaVersion' -Status 'Fail' -Detail "Unsupported or missing schemaVersion: '$schemaVersion'. This host speaks major version 1."
    }

    # --- rule: JSON Schema shape (best effort) -----------------------------

    if (-not (Test-Path -LiteralPath $schemaPath -PathType Leaf)) {
        Add-Result -Rule 'manifest.schema' -Status 'Warn' -Detail "Schema not found at $schemaPath; explicit rules still applied."
    } elseif (-not (Get-Command Test-Json -ErrorAction SilentlyContinue)) {
        Add-Result -Rule 'manifest.schema' -Status 'Skip' -Detail 'Test-Json is unavailable in this PowerShell host; explicit rules still applied.'
    } else {
        try {
            $schemaText = [IO.File]::ReadAllText($schemaPath)
            if (Test-Json -Json $manifestText -Schema $schemaText -ErrorAction Stop) {
                Add-Result -Rule 'manifest.schema' -Status 'Pass'
            } else {
                Add-Result -Rule 'manifest.schema' -Status 'Fail' -Detail 'Manifest does not satisfy the v1 JSON Schema.'
            }
        } catch {
            # NJsonSchema does not implement every draft 2020-12 construct. A
            # engine-level error is reported but is not treated as a manifest
            # defect, because the explicit rules below cover the substance.
            Add-Result -Rule 'manifest.schema' -Status 'Warn' -Detail "Schema engine could not validate: $($_.Exception.Message)"
        }
    }

    # --- rule: package identity --------------------------------------------

    $packageId = Get-Prop $manifest 'package.id'
    if ($packageId -is [string] -and $packageId -match '^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+$') {
        Add-Result -Rule 'package.id' -Status 'Pass' -Detail $packageId
    } else {
        Add-Result -Rule 'package.id' -Status 'Fail' -Detail "package.id must be reverse-DNS: '$packageId'"
    }

    $packageVersion = Get-Prop $manifest 'package.version'
    if ($packageVersion -is [string] -and $packageVersion -match '^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$') {
        Add-Result -Rule 'package.version' -Status 'Pass' -Detail $packageVersion
    } else {
        Add-Result -Rule 'package.version' -Status 'Fail' -Detail "package.version must be semantic: '$packageVersion'"
    }

    if ([string]::IsNullOrWhiteSpace([string](Get-Prop $manifest 'package.license.id'))) {
        Add-Result -Rule 'package.license' -Status 'Fail' -Detail 'package.license.id is required.'
    } else {
        Add-Result -Rule 'package.license' -Status 'Pass' -Detail (Get-Prop $manifest 'package.license.id')
    }

    $licenseFile = [string](Get-Prop $manifest 'package.license.file')
    if ([string]::IsNullOrWhiteSpace($licenseFile)) {
        Add-Result -Rule 'package.license.file' -Status 'Skip' -Detail 'No package-local license file declared.'
    } else {
        $licensePath = Get-SafePackageChildPath -RelativePath $licenseFile
        if ($null -eq $licensePath) {
            Add-Result -Rule 'package.license.file' -Status 'Fail' -Detail 'License path must remain inside the package root.'
        } elseif (-not (Test-Path -LiteralPath $licensePath -PathType Leaf)) {
            Add-Result -Rule 'package.license.file' -Status 'Fail' -Detail "Declared license file is missing: $licenseFile"
        } else {
            Add-Result -Rule 'package.license.file' -Status 'Pass' -Detail $licenseFile
        }
    }

    # --- rule: adapter ABI --------------------------------------------------

    $abiVersion = Get-Prop $manifest 'backend.adapter.abiVersion'
    if ($abiVersion -eq $HostAbiMajor) {
        Add-Result -Rule 'backend.adapter.abiVersion' -Status 'Pass' -Detail "ABI $abiVersion"
    } else {
        Add-Result -Rule 'backend.adapter.abiVersion' -Status 'Fail' -Detail "Adapter declares ABI major $abiVersion; this host speaks $HostAbiMajor."
    }

    # --- rule: distribution and artifacts ----------------------------------

    $distribution = Get-Prop $manifest 'provenance.distribution'
    $artifacts = ConvertTo-Array (Get-Prop $manifest 'provenance.artifacts')

    if ($distribution -eq 'bundled' -and $artifacts.Count -eq 0) {
        Add-Result -Rule 'provenance.artifacts' -Status 'Fail' -Detail "distribution 'bundled' requires at least one artifact."
    } else {
        Add-Result -Rule 'provenance.artifacts' -Status 'Pass' -Detail "distribution=$distribution, artifacts=$($artifacts.Count)"
    }

    # --- rule: artifact hashes ---------------------------------------------

    if ($artifacts.Count -eq 0) {
        Add-Result -Rule 'provenance.artifacts.sha256' -Status 'Skip' -Detail 'No artifacts declared.'
    } else {
        foreach ($artifact in $artifacts) {
            $relativePath = [string](Get-Prop $artifact 'path')
            $expectedHash = [string](Get-Prop $artifact 'sha256')
            $isOptional = [bool](Get-Prop $artifact 'optional')
            $ruleName = "provenance.artifacts.sha256[$relativePath]"

            $artifactPath = Get-SafePackageChildPath -RelativePath $relativePath
            if ($null -eq $artifactPath) {
                Add-Result -Rule $ruleName -Status 'Fail' -Detail 'Artifact path must remain inside the package root.'
                continue
            }

            if ($expectedHash -notmatch '^[A-Fa-f0-9]{64}$') {
                Add-Result -Rule $ruleName -Status 'Fail' -Detail "sha256 is not a 64-character hex digest."
                continue
            }
            if ($expectedHash -eq ('0' * 64)) {
                Add-Result -Rule $ruleName -Status 'Fail' -Detail 'Placeholder hash. Build the artifact and record its real digest before packaging.'
                continue
            }
            if (-not (Test-Path -LiteralPath $artifactPath -PathType Leaf)) {
                $status = if ($isOptional) { 'Warn' } else { 'Fail' }
                Add-Result -Rule $ruleName -Status $status -Detail "Artifact missing: $artifactPath"
                continue
            }

            $actualHash = (Get-FileHash -LiteralPath $artifactPath -Algorithm SHA256).Hash
            if ($actualHash -eq $expectedHash.ToUpperInvariant()) {
                Add-Result -Rule $ruleName -Status 'Pass'
            } else {
                Add-Result -Rule $ruleName -Status 'Fail' -Detail "Hash mismatch. Expected $expectedHash, found $actualHash."
            }

            $artifactRole = [string](Get-Prop $artifact 'role')
            $signatureRequired = Get-Prop $artifact 'signature.required'
            $signatureRule = "provenance.artifacts.signature[$relativePath]"
            if ($null -eq $signatureRequired) {
                $status = if ($artifactRole -in @('adapter', 'runtime')) { 'Fail' } else { 'Skip' }
                Add-Result -Rule $signatureRule -Status $status -Detail 'Native adapter/runtime artifacts must declare an explicit signature policy.'
            } elseif ($signatureRequired -eq $true) {
                $signatureKind = [string](Get-Prop $artifact 'signature.kind')
                $expectedSigner = [string](Get-Prop $artifact 'signature.expectedSigner')
                if ($signatureKind -ne 'authenticode' -or [string]::IsNullOrWhiteSpace($expectedSigner)) {
                    Add-Result -Rule $signatureRule -Status 'Fail' -Detail 'Required signatures must use Authenticode and name the expected signer.'
                } else {
                    $signature = Get-AuthenticodeSignature -LiteralPath $artifactPath
                    if ($signature.Status -ne 'Valid') {
                        Add-Result -Rule $signatureRule -Status 'Fail' -Detail "Signature status is $($signature.Status)."
                    } elseif ($signature.SignerCertificate.Subject -notlike "*$expectedSigner*") {
                        Add-Result -Rule $signatureRule -Status 'Fail' -Detail "Signer does not match '$expectedSigner'."
                    } else {
                        Add-Result -Rule $signatureRule -Status 'Pass' -Detail $expectedSigner
                    }
                }
            } else {
                Add-Result -Rule $signatureRule -Status 'Skip' -Detail 'Unsigned artifact is pinned by its declared SHA-256.'
            }
        }
    }

    # --- rule: adapter location and runtime trust ---------------------------

    $adapterLibrary = [string](Get-Prop $manifest 'backend.adapter.library')
    $adapterPath = Get-SafePackageChildPath -RelativePath $adapterLibrary
    if ($null -eq $adapterPath) {
        Add-Result -Rule 'backend.adapter.library' -Status 'Fail' -Detail 'Adapter library path must remain inside the package root.'
    } else {
        $adapterArtifact = @($artifacts | Where-Object {
            ([string](Get-Prop $_ 'role')) -eq 'adapter' -and
            ([string](Get-Prop $_ 'path')).Equals($adapterLibrary, [StringComparison]::OrdinalIgnoreCase)
        })
        if ($adapterArtifact.Count -ne 1) {
            Add-Result -Rule 'backend.adapter.library' -Status 'Fail' -Detail 'Adapter library must match exactly one declared adapter artifact.'
        } else {
            Add-Result -Rule 'backend.adapter.library' -Status 'Pass' -Detail $adapterLibrary
        }
    }

    $runtimeMethod = [string](Get-Prop $manifest 'backend.runtimeAcquisition.method')
    $runtimeSignatureRequired = Get-Prop $manifest 'backend.runtimeAcquisition.signature.required'
    if ($runtimeMethod -in @('locate', 'download')) {
        $runtimeSignatureKind = [string](Get-Prop $manifest 'backend.runtimeAcquisition.signature.kind')
        $runtimeExpectedSigner = [string](Get-Prop $manifest 'backend.runtimeAcquisition.signature.expectedSigner')
        if ($runtimeSignatureRequired -ne $true -or
            $runtimeSignatureKind -ne 'authenticode' -or
            [string]::IsNullOrWhiteSpace($runtimeExpectedSigner)) {
            Add-Result -Rule 'backend.runtimeAcquisition.signature' -Status 'Fail' -Detail 'Located/downloaded native runtimes require Authenticode validation and an expected signer.'
        } else {
            Add-Result -Rule 'backend.runtimeAcquisition.signature' -Status 'Pass' -Detail $runtimeExpectedSigner
        }
    } else {
        Add-Result -Rule 'backend.runtimeAcquisition.signature' -Status 'Skip' -Detail "Runtime acquisition method '$runtimeMethod' does not require a separate located-runtime policy."
    }

    # --- rule: exactly one color input --------------------------------------

    $requiredInputs = ConvertTo-Array (Get-Prop $manifest 'inputs.required')
    $colorInputs = @($requiredInputs | Where-Object { (Get-Prop $_ 'kind') -eq 'color' })
    if ($colorInputs.Count -eq 1) {
        Add-Result -Rule 'inputs.color' -Status 'Pass'
    } else {
        Add-Result -Rule 'inputs.color' -Status 'Fail' -Detail "inputs.required must contain exactly one 'color' guide; found $($colorInputs.Count)."
    }

    # --- rule: temporal reset semantics -------------------------------------

    $isStateful = Get-Prop $manifest 'temporal.stateful'
    if ($isStateful -eq $true) {
        if ($null -eq (Get-Prop $manifest 'temporal.reset')) {
            Add-Result -Rule 'temporal.reset' -Status 'Fail' -Detail 'A stateful package must declare temporal.reset so the broker knows how to clear history.'
        } else {
            $resetTriggers = ConvertTo-Array (Get-Prop $manifest 'temporal.reset.triggers')
            if ($resetTriggers -notcontains 'session-start') {
                Add-Result -Rule 'temporal.reset' -Status 'Warn' -Detail "triggers omits 'session-start'; the first frame after session creation is always a reset."
            } else {
                Add-Result -Rule 'temporal.reset' -Status 'Pass' -Detail "semantics=$(Get-Prop $manifest 'temporal.reset.semantics'), triggers=$($resetTriggers.Count)"
            }
        }
    } else {
        Add-Result -Rule 'temporal.reset' -Status 'Skip' -Detail 'Stateless package.'
    }

    # --- rule: settings -----------------------------------------------------

    $settings = ConvertTo-Array (Get-Prop $manifest 'settings')
    if ($settings.Count -eq 0) {
        Add-Result -Rule 'settings' -Status 'Skip' -Detail 'No settings declared.'
    } else {
        $seenKeys = [System.Collections.Generic.HashSet[string]]::new()
        foreach ($setting in $settings) {
            $key = [string](Get-Prop $setting 'key')
            $type = [string](Get-Prop $setting 'type')
            $ruleName = "settings[$key]"

            if (-not $seenKeys.Add($key)) {
                Add-Result -Rule $ruleName -Status 'Fail' -Detail 'Duplicate setting key.'
                continue
            }

            if ($type -eq 'enum') {
                $options = ConvertTo-Array (Get-Prop $setting 'options')
                if ($options.Count -lt 2) {
                    Add-Result -Rule $ruleName -Status 'Fail' -Detail 'An enum setting requires at least two options.'
                    continue
                }
                $optionValues = @($options | ForEach-Object { Get-Prop $_ 'value' })
                if ($optionValues -notcontains (Get-Prop $setting 'default')) {
                    Add-Result -Rule $ruleName -Status 'Fail' -Detail 'default is not one of the declared options.'
                    continue
                }
            }

            if ($type -eq 'float' -or $type -eq 'int') {
                $default = Get-Prop $setting 'default'
                $safeMin = Get-Prop $setting 'safeRange.min'
                $safeMax = Get-Prop $setting 'safeRange.max'
                if ($null -ne $safeMin -and $null -ne $safeMax) {
                    if ($default -lt $safeMin -or $default -gt $safeMax) {
                        Add-Result -Rule $ruleName -Status 'Fail' -Detail "default $default falls outside safeRange [$safeMin, $safeMax]."
                        continue
                    }
                }
                $min = Get-Prop $setting 'min'
                $max = Get-Prop $setting 'max'
                if ($null -ne $min -and $null -ne $max -and $min -gt $max) {
                    Add-Result -Rule $ruleName -Status 'Fail' -Detail "min $min is greater than max $max."
                    continue
                }
            }

            Add-Result -Rule $ruleName -Status 'Pass' -Detail $type
        }
    }

    # --- rule: redistribution / release gate --------------------------------

    $redistributionAllowed = Get-Prop $manifest 'provenance.redistribution.allowed'
    if ($null -eq $redistributionAllowed) {
        Add-Result -Rule 'provenance.redistribution' -Status 'Fail' -Detail 'provenance.redistribution.allowed must be declared explicitly.'
    } elseif ($ReleaseGate) {
        if ($redistributionAllowed -eq $true) {
            Add-Result -Rule 'release.redistribution' -Status 'Pass' -Detail 'Authorized for redistribution.'
        } else {
            $rationale = [string](Get-Prop $manifest 'provenance.redistribution.rationale')
            Add-Result -Rule 'release.redistribution' -Status 'Fail' -Detail "Not authorized for redistribution and must be excluded from a release package. $rationale"
        }
    } else {
        Add-Result -Rule 'provenance.redistribution' -Status 'Pass' -Detail "allowed=$redistributionAllowed"
    }
}

# --- report ----------------------------------------------------------------

$failed = @($results | Where-Object { $_.Status -eq 'Fail' })
$warned = @($results | Where-Object { $_.Status -eq 'Warn' })

if ($AsJson) {
    [pscustomobject]@{
        package  = if ($null -ne $manifest) { Get-Prop $manifest 'package.id' } else { $null }
        manifest = $manifestPath
        valid    = ($failed.Count -eq 0)
        failed   = $failed.Count
        warned   = $warned.Count
        rules    = $results
    } | ConvertTo-Json -Depth 6
} else {
    Write-Host ""
    Write-Host "Package : $(if ($null -ne $manifest) { Get-Prop $manifest 'package.id' } else { '<unparsed>' })"
    Write-Host "Manifest: $manifestPath"
    if ($ReleaseGate) { Write-Host "Mode    : release gate" }
    Write-Host ""
    $results | Format-Table -AutoSize Status, Rule, Detail
    Write-Host ""
    if ($failed.Count -eq 0) {
        Write-Host "VALID - $($results.Count) rules checked, $($warned.Count) warning(s)."
    } else {
        Write-Host "INVALID - $($failed.Count) rule(s) failed, $($warned.Count) warning(s)."
    }
}

if ($failed.Count -gt 0) { exit 1 }
exit 0
