#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('prep',Path(__file__).with_name('prepare-nr-spatial-motion.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Spatial(unittest.TestCase):
    def test_exact_cpp_half_encoder(self):
        source=(ROOT/'build/nr_spatial_motion/source-v1/color_cases_trace.cpp').read_text()
        match=re.search(r'auto encode=\[\]\(int q\)->uint16_t\{[^\n]+?\};',source);self.assertIsNotNone(match)
        code='#include <cstdint>\n#include <iostream>\nint main(){'+match.group(0)+'for(int q=-128;q<=128;++q)std::cout<<encode(q)<<"\\n";}\n'
        with tempfile.TemporaryDirectory(prefix='nr-motion-half-') as d:
            binary=Path(d)/'test';subprocess.run(['clang++','-std=c++20','-Wall','-Wextra','-Werror','-x','c++','-','-o',str(binary)],input=code,text=True,check=True,capture_output=True)
            values=np.array(subprocess.check_output([str(binary)],text=True).split(),dtype=np.uint16)
        expected=(np.arange(-128,129,dtype=np.float64)/16).astype(np.float16).view(np.uint16)
        self.assertTrue(np.array_equal(values,expected))
    def test_recipe_shape_and_range(self):
        data=m.recipe();self.assertEqual(len(data),8294400);values=np.frombuffer(data,dtype='<f2')
        self.assertEqual(float(values.min()),-8);self.assertEqual(float(values.max()),8);self.assertEqual(len(np.unique(values)),257)
    def test_transform_and_duplicate_rejection(self):
        parent=ROOT/'build/nr_motion_reference/source-v1/color_cases_trace.cpp';new=m.host(parent.read_text())
        self.assertEqual(new,(ROOT/'build/nr_spatial_motion/source-v1/color_cases_trace.cpp').read_text())
        with self.assertRaises(ValueError):m.host(new)
if __name__=='__main__':unittest.main(verbosity=2)
