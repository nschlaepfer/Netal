#!/usr/bin/env python3
"""Synthetic assembly fault tests; fixtures are not model/inference validation."""
import importlib.util,json,hashlib
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('assembly',Path(__file__).with_name('assemble-netal-sr-runtime.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class AssemblyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.assets=self.root/'assets';self.model=self.root/'model';self.scalar=self.root/'scalar';self.out=self.root/'bundle'
        for p in (self.assets,self.model,self.scalar):p.mkdir()
        (self.assets/'engine.txt').write_bytes(b'synthetic source build')
        self.data=b'fixture-model';(self.model/'weights.bin').write_bytes(self.data)
        for n in module.SCALAR_NAMES:(self.scalar/n).write_bytes(b'\0'*16)
        self.receipt={'schema':'netal.sr.source-body-assets.v1','files':{'engine.txt':{'bytes':22,'sha256':hashlib.sha256(b'synthetic source build').hexdigest()}}}
        self.write_receipt()
        self.patches=[patch.object(module,'SCALAR_BYTES',16),patch.object(module.manifest,'MODEL',{'model/weights.bin':(len(self.data),hashlib.sha256(self.data).hexdigest())}),patch.object(module.manifest,'REQUIRED',{'engine.txt','model/weights.bin'}|{'math/sm120-mufu/'+n for n in module.SCALAR_NAMES})]
        for p in self.patches:p.start()
    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.tmp.cleanup()
    def write_receipt(self): (self.assets/'SOURCE_BUILD_RECEIPT.json').write_text(json.dumps(self.receipt))
    def run_assembly(self):return module.assemble(self.assets,self.model,self.scalar,self.out)
    def test_complete_inventory(self):
        result=self.run_assembly();module.manifest.verify_engine(self.out,result)
        self.assertFalse(result['claims']['gameplay_validated']);self.assertFalse(json.loads((self.out/'source-assembly.json').read_text())['complete_public_bootstrap'])
    def test_tampered_asset(self):
        (self.assets/'engine.txt').write_bytes(b'changed')
        with self.assertRaises(ValueError):self.run_assembly()
        self.assertFalse(self.out.exists())
    def test_extra_asset(self):
        (self.assets/'extra.txt').write_text('extra')
        with self.assertRaises(ValueError):self.run_assembly()
    def test_bad_model(self):
        (self.model/'weights.bin').write_bytes(b'not the model')
        with self.assertRaises(ValueError):self.run_assembly()
    def test_bad_scalar_size(self):
        (self.scalar/module.SCALAR_NAMES[0]).write_bytes(b'bad')
        with self.assertRaises(ValueError):self.run_assembly()
    def test_linked_input(self):
        p=self.scalar/module.SCALAR_NAMES[0];p.unlink();p.symlink_to(self.scalar/module.SCALAR_NAMES[1])
        with self.assertRaises(ValueError):self.run_assembly()
    def test_existing_output_preserved(self):
        self.out.mkdir();(self.out/'keep').write_text('keep')
        with self.assertRaises(ValueError):self.run_assembly()
        self.assertEqual((self.out/'keep').read_text(),'keep')
    def test_output_inside_inputs_refused(self):
        self.out=self.assets/'new'
        with self.assertRaises(ValueError):self.run_assembly()
    def test_missing_required_asset(self):
        with patch.object(module.manifest,'REQUIRED',module.manifest.REQUIRED|{'missing.metallib'}):
            with self.assertRaises(ValueError):self.run_assembly()
    def test_duplicate_json_key(self):
        p=self.assets/'SOURCE_BUILD_RECEIPT.json';p.write_text('{"schema":"netal.sr.source-body-assets.v1","files":{},"files":{}}')
        with self.assertRaises(ValueError):self.run_assembly()

if __name__=='__main__':unittest.main()
