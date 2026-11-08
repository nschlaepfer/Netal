#!/usr/bin/env python3
"""Create a private approximate E4M3FN pack from the exact supported local SR runtime.

No model inputs or derived payloads are supplied by this repository. The output
is not a runtime or inference validation; retain the original FP16 bundle.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from netal_sr_fp8_pack import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='fresh private local directory')
    args = parser.parse_args()
    try:
        manifest = build(args.runtime, args.output)
    except (ValueError, OSError, KeyError, TypeError, IndexError, OverflowError) as error:
        parser.exit(1, 'FP8 weight packing failed: ' + str(error) + '\n')
    print(json.dumps({'result': 'pass', 'pack_root_sha256': manifest['pack_root_sha256'],
                      **manifest['summary'], 'derived_approximate': True, 'dlss45': False,
                      'inference_validated': False, 'image_quality_validated': False}))


if __name__ == '__main__':
    main()
