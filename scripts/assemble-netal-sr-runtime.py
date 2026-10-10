#!/usr/bin/env python3
"""Assemble source-built SR assets with explicit local model and scalar inputs.

This never copies a prepared runtime or extracts proprietary data. It verifies
the asset-build receipt, pins the supported model, records supplied scalar-table
identities, and seals the complete local bundle. Inventory is not parity proof.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import netal_sr_runtime_manifest as manifest

SCALAR_NAMES = ('lg2-input-table.bin', 'lg2-output-table.bin',
                'ex2-input-table.bin', 'ex2-output-table.bin')
SCALAR_BYTES = (1 << 20) * 4


def sha(data): return hashlib.sha256(data).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def inventory(root):
    if root.is_symlink() or not root.is_dir(): raise ValueError('Use a real input directory')
    names = set()
    for p in root.rglob('*'):
        if p.is_symlink(): raise ValueError('Input contains a symlink')
        if p.is_file(): names.add(p.relative_to(root).as_posix())
        elif not p.is_dir(): raise ValueError('Input contains a nonregular entry')
    return names


def assemble(assets, model, scalar, output):
    assets, model, scalar, output = map(Path, (assets, model, scalar, output))
    if output.exists() or output.is_symlink(): raise ValueError('Output exists; choose a new directory')
    for path in (assets, model, scalar):
        if output.resolve().is_relative_to(path.resolve()): raise ValueError('Output must be outside all inputs')
    names = inventory(assets)
    receipt_bytes = (assets/'SOURCE_BUILD_RECEIPT.json').read_bytes()
    receipt = json.loads(receipt_bytes, object_pairs_hook=unique_object)
    if receipt.get('schema') != 'netal.sr.source-body-assets.v1': raise ValueError('Unsupported source-build receipt')
    files = receipt.get('files')
    if not isinstance(files, dict) or names != set(files) | {'SOURCE_BUILD_RECEIPT.json'}:
        raise ValueError('Source-asset inventory is incomplete or has extra files')
    payload = {}
    for name, entry in files.items():
        if not manifest.safe_path(name) or name == 'manifest.json' or name.startswith('model/'):
            raise ValueError('Unsafe source asset path')
        data = (assets/name).read_bytes()
        if (not isinstance(entry, dict) or entry.get('bytes') != len(data)
                or entry.get('sha256') != sha(data)):
            raise ValueError('Source asset differs from build receipt: '+name)
        payload[name] = data
    inventory(model)
    for name, (size, digest) in manifest.MODEL.items():
        data = (model/Path(name).name).read_bytes()
        if len(data) != size or sha(data) != digest: raise ValueError('Unsupported model identity: '+name)
        payload[name] = data
    inventory(scalar)
    scalar_records = {}
    for name in SCALAR_NAMES:
        data = (scalar/name).read_bytes()
        if len(data) != SCALAR_BYTES: raise ValueError('Scalar table must contain exactly 1,048,576 uint32 entries: '+name)
        payload['math/sm120-mufu/'+name] = data
        scalar_records[name] = {'bytes':len(data), 'sha256':sha(data)}
    missing = manifest.REQUIRED - set(payload)
    if missing: raise ValueError('Source build lacks required runtime assets: '+', '.join(sorted(missing)))
    record = {
        'schema':'netal.sr.source-assembly.v1',
        'source_build_receipt_sha256':sha(receipt_bytes),
        'source_asset_count':len(files),
        'model':'DLSS_SR_Feature1_Quality_K',
        'scalar_contract_inputs':scalar_records,
        'scalar_contract_origin':'explicit local inputs; not independently regenerated or certified by assembly',
        'complete_public_bootstrap':False,
        'inference_validated':False, 'gameplay_validated':False,
        'redistribution_cleared':False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.sr-source-assembly-', dir=output.parent) as temporary:
        stage = Path(temporary)/'bundle'; stage.mkdir()
        for name, data in payload.items():
            p=stage/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
        (stage/'source-assembly.json').write_text(json.dumps(record, indent=2)+'\n')
        sealed = manifest.seal(stage)
        manifest.verify_engine(stage, sealed)
        if output.exists() or output.is_symlink(): raise ValueError('Output appeared during assembly; preserved')
        stage.rename(output)
    return sealed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','model','scalar-contract','output'):parser.add_argument('--'+name, required=True, type=Path)
    args=parser.parse_args()
    try: result=assemble(args.assets,args.model,args.scalar_contract,args.output.absolute())
    except (ValueError,OSError,KeyError,TypeError) as error:parser.exit(1, 'SR assembly failed: '+str(error)+'\n')
    print(json.dumps({'result':'pass','files':len(result['files']),'asset_root_sha256':result['asset_root_sha256'],
                      'complete_public_bootstrap':False,'inference_validated':False,'gameplay_validated':False}))


if __name__=='__main__':main()
