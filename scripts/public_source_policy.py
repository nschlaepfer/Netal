"""Mechanical public-source policy, separate from ownership and gameplay review."""
from pathlib import PurePosixPath
import hashlib
import re
from netal_sr_calibration import BINARY_FILES, ORACLE_SOURCES, validate_blob
from netal_fg_calibration import BINARY_FILES as FG_BINARY_FILES, ORACLE_SOURCES as FG_ORACLE_SOURCES, validate_blob as validate_fg_blob

SHOWCASE_GIF_PATH = 'docs/brand/netal-menubar-preview.gif'
SHOWCASE_GIF_IDENTITY = {
    'bytes': 1588998,
    'sha256': 'd139b7e84495f9bcb3e411157c11562b7657f20945182dfae055ca1b7334644a',
}


def valid_showcase_gif(data):
    # Only the exact visually reviewed export. Replacement artwork requires a
    # separate policy review before its bytes are accepted from a contribution.
    return (14 <= len(data) <= 10 * 1024 * 1024
            and data[:6] in (b'GIF87a', b'GIF89a') and data[-1:] == b';'
            and 0 < int.from_bytes(data[6:8], 'little') <= 1600
            and 0 < int.from_bytes(data[8:10], 'little') <= 1000
            and len(data) == SHOWCASE_GIF_IDENTITY['bytes']
            and hashlib.sha256(data).hexdigest() == SHOWCASE_GIF_IDENTITY['sha256'])

ROOT_FILES = {'LICENSES/CuMetal-Apache-2.0.txt', 'README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md', 'CONTRIBUTING.md', '.gitignore', '.gitattributes', '.github/CODEOWNERS', '.github/rulesets/main.json'}
DOCS = {'docs/FG_SOURCE_BUILD.md', 'docs/SR_BOUNDARY_KERNELS_AND_MATH.md', 'docs/SOURCE_PROVENANCE_REVIEW.md', 'docs/SR_BOUNDARY_METADATA.md', 'docs/SR_ARITHMETIC_MODES.md', 'docs/SR_SOURCE_BUILD.md', 'docs/MODEL_SETUP.md', 'docs/README.md', 'docs/getting-started.md', 'docs/RELEASE.md', 'docs/architecture.md',
        'docs/roadmap.md', 'docs/menubar.md', 'docs/RUNTIME_BUNDLE.md', 'docs/performance.md', 'docs/REPOSITORY_GOVERNANCE.md', 'docs/PUBLIC_READINESS.md', 'docs/COMPLETE_SOURCE_BUILD.md', 'docs/STATUS.md'}
EXTENSIONS = {'.swift', '.c', '.cc', '.cpp', '.h', '.hpp', '.mm', '.m', '.metal', '.py', '.sh',
              '.zsh', '.ps1', '.cmake', '.def', '.json', '.toml', '.txt', '.md'}
SOURCE_ROOTS = {'src', 'scripts', 'tools', 'runtime', 'adapters', 'packages'}
DENIED_PARTS = {'build', 'work', 'evidence', 'captures', 'capture', 'checkpoints', 'experiments',
                'build-inputs', '.git', '.env', '__pycache__', 'node_modules', 'audit', 'receipts'}
SECRET_PATTERNS = [
    ('private-key', re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----')),
    ('github-token', re.compile(r'\bgh[pousr]_[A-Za-z0-9]{30,}\b')),
    ('github-token', re.compile(r'\bgithub_pat_[A-Za-z0-9_]{50,}\b')),
    ('aws-access-key', re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b')),
    ('provider-key', re.compile(r'\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{40,}\b')),
]
LOCAL_PATH = re.compile('/' + 'Users/' + r'(?!Shared/|example/)[^/\s"\']+/')


def allowed(path):
    p = PurePosixPath(path)
    if p.is_absolute() or '..' in p.parts or any(x in DENIED_PARTS for x in p.parts):
        return False
    if path in ROOT_FILES or path in DOCS:
        return True
    # Exact scalar-data identities and explicitly listed scalar probe sources only.
    # This does not admit model data, arbitrary binary files or compiled libraries.
    if path in BINARY_FILES or path in FG_BINARY_FILES or path in ORACLE_SOURCES or path in FG_ORACLE_SOURCES:
        return True
    if path.startswith('.github/'):
        return p.suffix in {'.yml', '.yaml', '.md'}
    if path.startswith('docs/brand/'):
        return p.name in {'netal-banner.png', 'netal-icon-256.png', 'netal-appicon.png',
                          'netal-menubar-overview.png'} or path == SHOWCASE_GIF_PATH
    if path.startswith('src/macos_app/NetalMenuBar/Resources/'):
        return p.suffix in {'.png', '.json'}
    # FG shader headers and the two host-generation includes are text sources.
    # Keep this bounded; other research .inc payloads still require review.
    if p.suffix == '.metalh' and p.parent.as_posix() in {
            'src/fg_conv', 'src/fg_conv_simd_l20', 'src/fg_math_ex2'}:
        return True
    if path in {'src/fg_runtime/tail_resource_cache.inc',
                'src/fg_connected_l20_l87/guarded_block.inc'}:
        return True
    if p.parts[0] in SOURCE_ROOTS:
        # Research receipt trees and pinned machine-specific matrix experiments remain private.
        if path.startswith('src/nr_matrix_perf/') or path.startswith('tools/xlate/units/'):
            return False
        return p.suffix in EXTENSIONS or p.name in {'CMakeLists.txt', 'Makefile'}
    # Capture utilities contain compiled executables; retain source notices only if reviewed later.
    return False


def content_findings(data, image=False, relative=None):
    if relative == SHOWCASE_GIF_PATH:
        return [] if valid_showcase_gif(data) else ['invalid-showcase-gif']
    if relative in FG_BINARY_FILES:
        try:
            validate_fg_blob(relative, data)
            return []
        except ValueError:
            return ['calibration-identity-mismatch']
    if relative in BINARY_FILES:
        try:
            validate_blob(relative, data)
            return []
        except ValueError:
            return ['calibration-identity-mismatch']
    if image:
        return [] if data.startswith(b'\x89PNG\r\n\x1a\n') else ['invalid-png']
    try:
        text = data.decode('utf-8')
    except UnicodeError:
        return ['non-text-source']
    if '\0' in text:
        return ['binary-source']
    findings = [label for label, pattern in SECRET_PATTERNS if pattern.search(text)]
    if LOCAL_PATH.search(text) or ('nico' + '-rig') in text:
        findings.append('developer-home-path')
    if text.startswith('version https://git-lfs.github.com/spec/'):
        findings.append('lfs-pointer')
    return sorted(set(findings))
