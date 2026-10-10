#!/usr/bin/env python3
"""CPU-only retrieval and prepared recipe contract checks; no inference."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('retrieval',ROOT/'scripts/retrieve-nr-sixteen-raw-reference.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class RemoteInventory(unittest.TestCase):
    def setUp(self):
        self.expected={('case','file'):'a'*64}
        self.r={'host':'NICO-RIG','job':m.REMOTE,'rows':[{'label':'case','name':'file','bytes':m.BYTES,'sha256':'a'*64}]}
    def test_match(self):m.validate_remote(self.r,self.expected)
    def test_missing(self):
        self.r['rows']=[]
        with self.assertRaisesRegex(ValueError,'complete unique'):m.validate_remote(self.r,self.expected)
    def test_duplicate(self):
        self.r['rows']*=2
        with self.assertRaisesRegex(ValueError,'complete unique'):m.validate_remote(self.r,self.expected)
    def test_wrong_fields(self):
        for k,v in [('label','other'),('name','other'),('bytes',0),('sha256','b'*64)]:
            with self.subTest(k=k):
                r=copy.deepcopy(self.r);r['rows'][0][k]=v
                with self.assertRaises(ValueError):m.validate_remote(r,self.expected)
    def test_wrong_host(self):
        self.r['host']='other'
        with self.assertRaisesRegex(ValueError,'remote source identity'):m.validate_remote(self.r,self.expected)
    def test_wrong_job(self):
        self.r['job']='other'
        with self.assertRaisesRegex(ValueError,'remote source identity'):m.validate_remote(self.r,self.expected)

class PreparedRecipe(unittest.TestCase):
    def test_complete_repaired_sequence(self):
        recipe=json.loads((ROOT/'build/nr_sixteen/recovery-config-v1/recipe.json').read_text())
        refs=m.sealed(ROOT/'build/nr_sixteen/raw-reference-recovery-v1/receipt.json','21f516d4b05b88b5edb869b30d245e85f6732765ab1216f7e846b44c0300a78c')
        self.assertEqual(recipe['encoder_object_sha256'],'830d280992f9d2a2032bd6af625a5de462bb3d770bd2c5cb65e0003e7033fff3')
        self.assertEqual(recipe['reset_sequence'],'1'+',0'*15)
        self.assertEqual(len(recipe['frames']),16)
        for i,frame in enumerate(recipe['frames']):
            self.assertEqual(frame['reference'],refs['local_files'][i]['path'])
            self.assertEqual(frame['reference_sha256'],refs['local_files'][i]['sha256'])
            poisons=[r['poisons'][i] for r in recipe['replicas']]
            self.assertEqual(poisons.count('a5'),2);self.assertEqual(poisons.count('5a'),2)
        self.assertTrue(recipe['comparison_references_required'])
        self.assertEqual(recipe['storage_floor_gib'],12)

if __name__=='__main__':unittest.main()
