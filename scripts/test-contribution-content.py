#!/usr/bin/env python3
"""Offline boundary cases for the changed-content guard; no real secrets or assets."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import hashlib
import base64

spec=importlib.util.spec_from_file_location('guard',Path(__file__).with_name('check-contribution-content.py'))
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)

class ContentTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.repo=Path(self.tmp.name)
        self.git('init','-q');self.git('config','user.name','Fixture');self.git('config','user.email','fixture@example.invalid')
        self.write('README.md',b'Local synthetic fixture\n');self.base=self.commit('README.md')
    def tearDown(self):self.tmp.cleanup()
    def git(self,*args):return subprocess.check_output(['git','-C',str(self.repo),*args]).decode().strip()
    def write(self,name,data):p=self.repo/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    def commit(self,*names):self.git('add','--',*names);self.git('commit','-qm','fixture');return self.git('rev-parse','HEAD')
    def inspect(self,name,data):self.write(name,data);return guard.check(self.repo,self.base,self.commit(name))
    def test_text_passes(self):self.assertEqual(self.inspect('docs/status.md',b'Recorded limits\n')['status'],'pass')
    def test_dll_rejected(self):self.assertEqual(self.inspect('vendor/sample.dll',b'fake')['status'],'fail')
    def test_archive_rejected(self):self.assertEqual(self.inspect('docs/pack.zip',b'fake')['status'],'fail')
    def test_home_path_rejected(self):self.assertEqual(self.inspect('docs/local.md',('/'+'Users/'+'private-person/file').encode())['status'],'fail')
    def test_example_path_passes(self):self.assertEqual(self.inspect('docs/example.md',('/'+'Users/example/file').encode())['status'],'pass')
    def test_secret_never_returned(self):
        fake=('gh'+'p_'+'A'*36).encode();result=self.inspect('token.txt',fake)
        self.assertEqual(result['status'],'fail');self.assertNotIn(fake.decode(),str(result))
    def test_symlink_rejected(self):
        (self.repo/'link.md').symlink_to('README.md');result=guard.check(self.repo,self.base,self.commit('link.md'));self.assertEqual(result['status'],'fail')
    def test_existing_legacy_file_not_rescanned(self):
        self.write('legacy.dll',b'fake');base=self.commit('legacy.dll');self.write('notes.md',b'no assets');head=self.commit('notes.md')
        self.assertEqual(guard.check(self.repo,base,head)['status'],'pass')
    def test_deletion_allowed(self):
        self.write('legacy.dll',b'fake');base=self.commit('legacy.dll');(self.repo/'legacy.dll').unlink();head=self.commit('legacy.dll')
        self.assertEqual(guard.check(self.repo,base,head)['status'],'pass')
    def test_binary_disguised_as_source_rejected(self):self.assertEqual(self.inspect('docs/payload.md',b'a\0b')['status'],'fail')
    def test_newline_filename_handled(self):self.assertEqual(self.inspect('docs/new\nname.md',b'documented')['changed_files_checked'],1)
    def test_revision_options_rejected(self):
        with self.assertRaises(ValueError):guard.check(self.repo,'--all',self.base)
    def test_initial_commit_scanned(self):self.assertEqual(guard.check(self.repo,'0'*40,self.base)['status'],'pass')
    def test_scalar_exception_requires_exact_identity(self):
        data=b'measured scalar fixture';name='runtime/arithmetic-calibration/fixture.bin'
        with patch.dict(guard.CALIBRATION,{name:{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}}):
            self.assertEqual(self.inspect(name,data)['status'],'pass')
            self.assertEqual(self.inspect(name,b'X'+data[1:])['status'],'fail')
    def test_export_does_not_drop_privacy_checker(self):
        from public_source_policy import content_findings
        p=Path(__file__).with_name('check-contribution-content.py')
        self.assertEqual(content_findings(p.read_bytes()),[])
    def test_showcase_gif_is_a_bounded_named_exception(self):
        from public_source_policy import SHOWCASE_GIF_PATH, allowed, content_findings
        data=base64.b64decode('R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7')
        self.assertTrue(allowed(SHOWCASE_GIF_PATH))
        self.assertFalse(allowed('docs/brand/arbitrary.gif'))
        self.assertEqual(self.inspect(SHOWCASE_GIF_PATH,data)['status'],'pass')
        self.assertEqual(content_findings(data,relative=SHOWCASE_GIF_PATH),[])
        self.assertEqual(self.inspect('docs/brand/arbitrary.gif',data)['status'],'fail')
        for invalid in (data[:-1],b'not an image',data[:6]+b'\xff\xff'+data[8:]):
            self.assertEqual(self.inspect(SHOWCASE_GIF_PATH,invalid)['status'],'fail')
            self.assertEqual(content_findings(invalid,relative=SHOWCASE_GIF_PATH),['invalid-showcase-gif'])

if __name__=='__main__':unittest.main()
