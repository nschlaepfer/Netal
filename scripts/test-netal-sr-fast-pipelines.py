#!/usr/bin/env python3
"""Model-free reproducibility/rejection tests for separate fast sidecars.

All binary fixtures are procedural zero bytes. Test-only patched model digests
allow the real public repacker/generators to run without proprietary model data.
The production CLI has no option to override the supported model pins.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fast_sidecars',ROOT/'scripts/build-netal-sr-fast-pipelines.py')
fast=importlib.util.module_from_spec(spec);spec.loader.exec_module(fast)


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory(prefix='netal-fast-sidecar-tests-')
        cls.base=Path(cls.temporary.name);cls.source=cls.base/'source';cls.source.mkdir()
        synthetic={name:bytes(size)for name,(size,_)in fast.runtime.MODEL.items()}
        cls.pins=mock.patch.dict(fast.runtime.MODEL,{name:(len(data),fast.sha(data))for name,data in synthetic.items()},clear=True)
        cls.pins.start()
        for relative in fast.runtime.REQUIRED:
            path=cls.source/relative;path.parent.mkdir(parents=True,exist_ok=True)
            lengths={'hiluma-input-reset-template.bin':248,'hiluma-output-reset-template.bin':328,'exposure-reset-template.bin':16}
            path.write_bytes(synthetic.get(relative,bytes(lengths.get(path.name,1))))
        cls.abi=fast.module('fast_fixture_boundary_abi',ROOT/'tools/netal_sr_abi.py')
        base={name:(cls.source/'abi'/name).read_bytes()for name in cls.abi.CONTRACTS}
        cls.boundaries=cls.abi.for_geometry(base,(1920,1080),(1280,720))
        for name,data in cls.boundaries.items():(cls.source/'abi'/name).write_bytes(data)
        cls.layouts={};repacker=fast.module('fast_fixture_repacker',ROOT/'tools/pwin_pipeline/repack.py')
        weights=synthetic['model/weights.bin']
        for layer in fast.LAYERS:
            packed,layout=repacker.repack(weights,fast.read_json(ROOT/f'tools/pwin_pipeline/spec-{layer}.json'))
            path=cls.source/f'pipeline/{layer}-weights.bin';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(packed)
            path=cls.source/f'pipeline/layouts/layout-{layer}.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(layout))
            cls.layouts[layer]=layout
        for width,height in((1280,720),(1920,1080)):
            files,metadata=fast.profile(width,height)
            reference=(854,480)if(width,height)==(1280,720)else(1280,720)
            files.update({'abi/'+name:data for name,data in cls.abi.for_geometry(cls.boundaries,(width,height),reference).items()})
            metadata.update({'render_reference':list(reference),'boundary_abi_included':True})
            destinations=[cls.source/f'profiles/{width}x{height}']+([cls.source]if(width,height)==(1920,1080)else[])
            for destination in destinations:
                for relative,data in files.items():
                    path=destination/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            (cls.source/f'profiles/{width}x{height}/layout.json').write_text(json.dumps(metadata))
            (cls.source/f'profiles/{width}x{height}/manifest.json').write_text(json.dumps(cls.abi.profile_manifest(metadata,reference,files)))
        (cls.source/'pwin/enc0-arguments-template.bin').write_bytes((cls.source/'pwin/enc0/arguments-template.bin').read_bytes())
        env={k:v for k,v in os.environ.items()if not k.startswith('PWIN_')}
        subprocess.run([sys.executable,str(ROOT/'tools/pwin_pipeline/build-profile-pipelines.py'),str(cls.source),
                        '--layout-dir',str(cls.source/'pipeline/layouts')],env=env,capture_output=True,check=True)
        cls.manifest=fast.runtime.seal(cls.source)

    @classmethod
    def tearDownClass(cls):cls.pins.stop();cls.temporary.cleanup()

    def clone(self,name):
        destination=self.base/name;shutil.copytree(self.source,destination);return destination

    def test_reproducible_full_graph_and_receipt(self):
        output=self.base/'fast-output';before=(self.source/'manifest.json').read_bytes()
        receipt=fast.build(self.source,output,'fast-half-mlp')
        self.assertEqual(len(receipt['pipelines']),138)
        self.assertEqual(receipt['max_pwin_dispatches'],94)
        self.assertEqual(receipt['expected_library_usage']['builder_variant'],'fast-half-mlp')
        self.assertFalse(receipt['inference_validated']);self.assertFalse(receipt['contains_model_weights'])
        self.assertEqual((self.source/'manifest.json').read_bytes(),before)
        self.assertTrue(all(p.suffix=='.json'for p in output.rglob('*')if p.is_file()))
        totals={}
        for row in receipt['pipelines']:
            original=(self.source/row['path']).read_bytes();data=(output/row['path']).read_bytes()
            self.assertEqual(fast.sha(original),row['source_sha256']);self.assertEqual(fast.sha(data),row['output_sha256'])
            q=json.loads(data);self.assertEqual(q['gemm_intermediate_fp16_handoffs'],'deliberately omitted')
            self.assertEqual(q['mlp_backend'],'mpp-half')
            self.assertEqual(q['gemm_backend'],'mpp');self.assertEqual(q['mpp_variant'],'fast-half-mlp')
            layer,phase=fast.ENTRY.fullmatch(Path(row['path']).name).groups()
            if not phase:
                group=str(Path(row['path']).parent);totals.setdefault(group,[0,0]);totals[group][0]+=row['gemm_dispatches'];totals[group][1]+=row['half_mlp_dispatches']
            for d in q['dispatches']:
                if d['function']=='pwin_mlp2_tokens':self.assertEqual(d['grid'],[(d['bytes'][0]['u32'][0]+63)//64,1,1]);self.assertEqual(d['block'],[128,1,1])
            expected=fast.half_mlp_graph(fast.mpp_backend.convert_pipeline(json.loads(original)))
            self.assertEqual(fast.core(q),fast.core(expected))
        self.assertEqual(list(totals.values()),[[42,6]]*3)
        with self.assertRaises(ValueError):fast.build(self.source,output,'fast-half-mlp')

    def test_mpp_source_and_direct_variant(self):
        source=self.clone('mpp-source');paths,_=fast.graph_files(source)
        for _,path in paths.items():path.write_text(json.dumps(fast.mpp_backend.convert_pipeline(fast.read_json(path))))
        fast.runtime.seal(source,replace=True)
        output=self.base/'direct-output';receipt=fast.build(source,output,'direct-mlp')
        self.assertEqual(len(receipt['pipelines']),138)
        q=fast.read_json(output/'pipeline/dltss_pwin_enc0_layer.json')
        self.assertEqual(q['gemm_intermediate_fp16_handoffs'],'preserved')
        self.assertEqual(q['kernel_variant'],'direct-mlp')
        self.assertEqual(q['gemm_backend'],'mpp');self.assertEqual(q['mpp_variant'],'direct-mlp')

    def test_reject_resealed_weight_mutation(self):
        source=self.clone('corrupt-weight');path=source/'pipeline/enc0-weights.bin';data=bytearray(path.read_bytes());data[0]=1;path.write_bytes(data)
        fast.runtime.seal(source,replace=True)
        with self.assertRaisesRegex(ValueError,'Repacked source weights'):
            fast.build(source,self.base/'bad-weight-output','fast-half-mlp')

    def test_reject_model_mutation_omitted_by_engine_verifier(self):
        source=self.clone('corrupt-model');path=source/'model/weights.bin';data=bytearray(path.read_bytes());data[0]=1;path.write_bytes(data)
        # The engine verifier excludes model bytes; the fast tool must still pin them.
        with self.assertRaisesRegex(ValueError,'pinned model'):
            fast.build(source,self.base/'bad-model-output','fast-half-mlp')

    def test_reject_missing_phase_and_resealed_program_mutation(self):
        source=self.clone('missing-phase');(source/'profiles/1280x720/pipeline/dltss_pwin_enc4_layer.phase7.json').unlink();fast.runtime.seal(source,replace=True)
        with self.assertRaisesRegex(ValueError,'shift phases'):
            fast.build(source,self.base/'bad-phase-output','fast-half-mlp')
        source=self.clone('changed-program');path=source/'pipeline/dltss_pwin_enc0_layer.json';q=fast.read_json(path);q['dispatches'][1]['buffers'][0]['index']=999;path.write_text(json.dumps(q));fast.runtime.seal(source,replace=True)
        with self.assertRaisesRegex(ValueError,'canonical descriptors'):
            fast.build(source,self.base/'bad-program-output','fast-half-mlp')

    def test_transformed_mlp_full_spans_and_binding_indices(self):
        source=fast.read_json(self.source/'pipeline/dltss_pwin_enc0_layer.json')
        q=fast.half_mlp_graph(fast.mpp_backend.convert_pipeline(source))
        size=(self.source/'pipeline/enc0-weights.bin').stat().st_size
        arenas=fast.profile(1920,1080)[1]['arenas']
        for index in(2,4,5,8):
            bad=copy.deepcopy(q);d=next(d for d in bad['dispatches']if d['function']=='pwin_mlp2_tokens');b=next(b for b in d['buffers']if b['index']==index)
            b['offset']=(size if b['name']=='pweights'else bad['scratch_bytes'])-(4 if index==2 else 2)
            with self.subTest(index=index),self.assertRaisesRegex(ValueError,'buffer span'):
                fast.validate_transformed('enc0',bad,size,arenas)
        bad=copy.deepcopy(q);bad['dispatches'][0]['buffers'][0]['index']=999
        with self.assertRaisesRegex(ValueError,'buffer binding'):
            fast.validate_transformed('enc0',bad,size,arenas)

    def test_reject_missing_profile_manifest_and_resealed_small_arena(self):
        for label in('missing-manifest','small-arena'):
            source=self.clone(label);path=source/'profiles/1280x720/manifest.json'
            if label=='missing-manifest':path.unlink()
            else:
                q=fast.read_json(path);q['arenas']['6']-=4096;path.write_text(json.dumps(q))
            fast.runtime.seal(source,replace=True)
            with self.subTest(label=label),self.assertRaisesRegex(ValueError,'Profile manifest'):
                fast.build(source,self.base/(label+'-output'),'fast-half-mlp')

    def test_profile_arrays_layout_and_raw_templates(self):
        groups=[Path('pipeline'),Path('profiles/1280x720/pipeline'),Path('profiles/1920x1080/pipeline')]
        geometry={str(group):fast.profile(*(map(int,group.parts[1].split('x'))if group!=Path('pipeline')else(1920,1080)))[1]for group in groups}
        for label in('array','layout','pwin-template','boundary-template'):
            source=self.clone('profile-'+label);prefix=source/'profiles/1280x720'
            if label=='array':
                path=prefix/'manifest.json';q=fast.read_json(path);q['arrays'][4]['width']-=1;path.write_text(json.dumps(q))
            elif label=='layout':
                path=prefix/'layout.json';q=fast.read_json(path);q['arenas']['7']-=4096;path.write_text(json.dumps(q))
            else:
                path=prefix/('pwin/enc0/arguments-template.bin'if label=='pwin-template'else'abi/hiluma-output-reset-template.bin')
                data=bytearray(path.read_bytes());data[0 if label=='pwin-template'else 136]^=1;path.write_bytes(data)
            fast.runtime.seal(source,replace=True)
            with self.subTest(label=label),self.assertRaisesRegex(ValueError,'geometry mismatch'):
                fast.profile_admission(source,groups,geometry)


if __name__=='__main__':unittest.main(verbosity=2)
