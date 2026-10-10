#!/usr/bin/env python3
"""Audit repeated RTX primitive captures, then test Fable's frozen sampler.

Candidate failures are evidence, not capture failures. No model or GPU execution.
The synthetic recipe contains all finite half classes, never archived outputs.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np

SOURCE = '19aeb1a1339636acd57513e9ae7371c03a255f0c7e57ae2c0eb667f9b1dc7e5f'
W, H = 1920, 1080
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def canonical(r): return hashlib.sha256(json.dumps(r, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def check(v, label):
    if not v: raise ValueError(label)
def recipe():
    v = np.arange(1, W * H * 4 + 1, dtype=np.uint32)
    v ^= v >> 16; v *= np.uint32(0x7feb352d); v ^= v >> 15
    v *= np.uint32(0x846ca68b); v ^= v >> 16
    return ((v % 0x7c00) | ((v >> 16) & 0x8000)).astype('<u2').reshape(H, W, 4)

def audit(captures, candidate_source):
    check(len(captures) == 4, 'four captures required')
    image = recipe(); image_hash = hashlib.sha256(image.tobytes()).hexdigest()
    check(np.all((image & 0x7c00) != 0x7c00), 'finite recipe')
    runs = []; pids = set(); base_samples = base_queries = binary = None
    for d, poison in zip(captures, (165, 165, 90, 90)):
        job = read(d / (d.name + '-receipt.json')); run = read(d / 'run.json'); build = read(d / 'build.json')
        check(job['exit_code'] == build['exit_code'] == 0, 'execution status')
        check(job['source_sha256'] == build['source_sha256'] == SOURCE, 'source identity')
        check(job['binary_sha256'] == build['binary_sha256'], 'binary identity')
        if binary is None: binary = job['binary_sha256']
        check(binary == job['binary_sha256'], 'build repeat identity')
        check(run['schema'] == 'neuralstudio.nr.finite-half-texture-oracle.v1' and run['result'] == 'pass', 'capture schema/status')
        check(run['exact_write_coverage'] is True and run['input_immutable'] is True, 'coverage/immutability')
        check(run['poison'] == poison and run['pid'] > 0 and run['pid'] not in pids, 'fresh process/poison')
        pids.add(run['pid'])
        check(run['coordinate_rows'] == 3000 and run['sample_queries'] == 1381, 'complete query count')
        for n in ('run.json', 'queries.bin', 'samples.bin', 'coord-1920.bin', 'coord-1080.bin'):
            check(sha(d / n) == job['files'][n]['sha256'] and (d / n).stat().st_size == job['files'][n]['bytes'], 'payload identity: ' + n)
        for n in ('input.rgba16le', 'input-after.rgba16le'):
            check(job['files'][n]['sha256'] == image_hash and job['files'][n]['bytes'] == image.nbytes, 'recipe/texture identity: ' + n)
        q = np.fromfile(d / 'queries.bin', dtype='<u4').reshape(-1, 2)
        samples = np.fromfile(d / 'samples.bin', dtype='<u4').reshape(-1, 4)
        check(q.shape == (1381, 2) and samples.shape == (1381, 4), 'payload sizes')
        check(np.isfinite(q.view('<f4')).all(), 'finite queries')
        if base_samples is None: base_queries, base_samples = q.copy(), samples.copy()
        check(np.array_equal(q, base_queries) and np.array_equal(samples, base_samples), 'repeat/poison equality')
        runs.append({'pid': run['pid'], 'poison': poison, 'receipt_sha256': sha(d / (d.name + '-receipt.json')), 'samples_sha256': sha(d / 'samples.bin')})
    # Extract only the current owner's pure numpy function, retaining exact source.
    text = candidate_source.read_text(); tree = ast.parse(text)
    definitions = {n.name:n for n in tree.body if isinstance(n, ast.FunctionDef)}
    check('tex_bilinear_rtx' in definitions, 'owned candidate function')
    selected = set(); pending = ['tex_bilinear_rtx']
    while pending:
        name = pending.pop()
        if name in selected: continue
        selected.add(name)
        pending.extend(n.id for n in ast.walk(definitions[name]) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in definitions and n.id not in selected)
    function = '\n\n'.join(ast.get_source_segment(text,n) for n in sorted((definitions[name] for name in selected),key=lambda n:n.lineno)); scope = {'np': np}
    exec(compile(function, str(candidate_source), 'exec'), scope)
    q = base_queries.view('<f4'); f32_image = image.view('<f2').astype(np.float32)
    predicted = scope['tex_bilinear_rtx'](f32_image, W, H, q[:, 0].astype(np.float64), q[:, 1].astype(np.float64))
    predicted = np.stack(predicted, axis=1).astype('<f4').view('<u4')
    bad = np.argwhere(predicted != base_samples)
    examples = [{'query': int(i), 'channel': int(c), 'u_bits': f'{int(base_queries[i,0]):08x}', 'v_bits': f'{int(base_queries[i,1]):08x}', 'rtx_bits': f'{int(base_samples[i,c]):08x}', 'candidate_bits': f'{int(predicted[i,c]):08x}'} for i, c in bad[:32]]
    check(sha(candidate_source) == hashlib.sha256(text.encode()).hexdigest(), 'candidate source changed during audit')
    report = {'schema': 'neuralstudio.nr.finite-half-texture-independent-audit.v1', 'capture_result': 'pass_repeated_primitive_observation', 'model_executed': False, 'model_parity': False, 'source_sha256': SOURCE, 'binary_sha256': binary, 'runs': runs, 'queries': len(q), 'values_per_run': int(base_samples.size), 'image_recipe_sha256': image_hash, 'finite_half_patterns_in_image': int(np.unique(image).size), 'candidate_source_sha256': sha(candidate_source), 'candidate_function_sha256': hashlib.sha256(function.encode()).hexdigest(), 'candidate_unequal_values': len(bad), 'candidate_exact_on_dataset': len(bad) == 0, 'first_counterexamples': examples, 'limits': ['Synthetic finite-half RGBA16F at 1920x1080; normalized/clamp/linear descriptor only.', 'A finite query set is not a universal sampler proof.', 'Candidate is CPU arithmetic, not a measured Metal backend or speedup.', 'Coverage/canaries and device-input immutability enforced by the pinned CUDA source; raw output receipts bound here.'], 'verifier_sha256': sha(Path(__file__))}
    return report, image, function, predicted

def main():
    p = argparse.ArgumentParser(); p.add_argument('--captures', type=Path, nargs=4, required=True); p.add_argument('--candidate-source', type=Path, required=True); p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    check(not a.output.exists(), 'preserve prior audit')
    r, image, function, predicted = audit(a.captures, a.candidate_source)
    a.output.mkdir(parents=True)
    (a.output / 'input.rgba16le').write_bytes(image.tobytes())
    (a.output / 'candidate.py').write_text(function + '\n')
    (a.output / 'candidate-samples.bin').write_bytes(predicted.tobytes())
    r['receipt_root_sha256'] = canonical(r)
    (a.output / 'verification.json').write_text(json.dumps(r, indent=2) + '\n')
    print(json.dumps(r, indent=2))
if __name__ == '__main__': main()
