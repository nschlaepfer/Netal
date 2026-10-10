#!/usr/bin/env python3
"""Identify explicitly marked compiler-generated shaders for source-rights review.

This is an evidence inventory, not a license classifier. Unmarked files are not thereby cleared.
It distinguishes the compiler/helper license from rights in the compiler's input program.
"""
import argparse,hashlib,json
from pathlib import Path


def inventory(root):
    records=[]
    for directory in ('src','tools'):
        for path in sorted((root/directory).rglob('*.metal')):
            if any(part in {'build','evidence','work'} for part in path.relative_to(root).parts):continue
            text=path.read_text()
            markers=[line.strip() for line in text.splitlines()[:30] if 'cumetal-provenance:' in line]
            if not markers:continue
            records.append({'path':path.relative_to(root).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                            'evidence':markers,'generator':'CuMetal','generator_license':'Apache-2.0',
                            'input_origin_and_distribution_rights':'review-required'})
    return {'schema':'netal.source-provenance-inventory.v1','scope':'explicit CuMetal markers in source shader files',
            'records':records,'unmarked_files_cleared':False,'publication_clearance':False,
            'compiler_reference':{'url':'https://github.com/Lulzx/cuda-metal','local_revision':'14a677715904d2e3d14f4973479df5620eb2f8b9',
                                  'license_file':'LICENSES/CuMetal-Apache-2.0.txt'},
            'limitations':['A compiler license does not establish rights in every translated input.',
                           'An absent marker does not prove original authorship or a permissive license.',
                           'This inventory is not a determination of applicable law or contractual permissions.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=inventory(args.root)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(f'marked_shader_files={len(result["records"])} publication_clearance=false')


if __name__=='__main__':main()
