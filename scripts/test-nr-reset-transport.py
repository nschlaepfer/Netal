#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import random
import tempfile
import unittest
import zipfile
def module(name,file):
    s=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
pack=module('pack','pack-nr-reset-retrieval.py');get=module('get','retrieve-nr-reset-parts.py')
class Transport(unittest.TestCase):
    def test_pack_join_unpack(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'input';a.write_bytes(random.Random(7).randbytes(700000));before=pack.sha(a)
            r=pack.pack({'data.bin':a},root/'transport','test');self.assertGreater(len(r['parts']),1)
            self.assertEqual(pack.sha(a),before);self.assertFalse(r['model_executed'])
            joined=root/'joined.zip'
            with joined.open('xb') as f:
                for part in r['parts']:
                    p=root/'transport'/part['name'];self.assertTrue(get.valid(p,part));f.write(p.read_bytes())
            self.assertTrue(get.valid(joined,r['archive']));get.unpack(joined,root/'out',r['files']);self.assertEqual((root/'out/data.bin').read_bytes(),a.read_bytes())
            with self.assertRaises(AssertionError):pack.pack({'data.bin':a},root/'transport','test')
    def test_corrupt_part_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'part';p.write_bytes(b'bad');self.assertFalse(get.valid(p,{'bytes':3,'sha256':'0'*64}))
    def test_unsafe_archive_name_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'unsafe.zip'
            with zipfile.ZipFile(p,'x') as z:z.writestr('../escape',b'x')
            with self.assertRaises(AssertionError):get.unpack(p,Path(tmp)/'out',{'../escape':{'bytes':1,'sha256':'0'*64}})
            self.assertFalse((Path(tmp)/'out').exists())
    def test_duplicate_zip_names_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'duplicate.zip'
            with zipfile.ZipFile(p,'x') as z:
                z.writestr('x',b'a')
                with self.assertWarns(UserWarning):z.writestr('x',b'b')
            with self.assertRaises(AssertionError):get.unpack(p,Path(tmp)/'out',{'x':{'bytes':1,'sha256':'0'*64}})
    def test_payload_hash_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.zip'
            with zipfile.ZipFile(p,'x') as z:z.writestr('x',b'a')
            with self.assertRaises(AssertionError):get.unpack(p,Path(tmp)/'out',{'x':{'bytes':1,'sha256':'0'*64}})
if __name__=='__main__':unittest.main(verbosity=2)
