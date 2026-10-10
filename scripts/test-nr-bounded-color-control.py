#!/usr/bin/env python3
import importlib.util
import pathlib
import unittest
import numpy as np
spec=importlib.util.spec_from_file_location('control',pathlib.Path(__file__).with_name('prepare-nr-bounded-color-control.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class ColorControlTests(unittest.TestCase):
    def test_zero_and_reference_white(self):
        x=np.array([[0,0,0,1],[1,1,1,1]],dtype='<f2')
        y=module.transform(x)
        np.testing.assert_array_equal(y[0],x[0])
        self.assertAlmostEqual(float(y[1,0]),0.73535698,places=3)

    def test_full_nonnegative_finite_half_domain_is_monotone_bounded(self):
        # All31744nonnegative finite FP16 bit patterns, including subnormals.
        values=np.arange(0x7c00,dtype='<u2').view('<f2')
        x=np.stack([values,values,values,np.ones_like(values)],axis=-1)
        y=module.transform(x)[:,0].astype('float32')
        self.assertTrue(np.isfinite(y).all())
        self.assertTrue((np.diff(y)>=0).all())
        self.assertGreaterEqual(y.min(),0)
        self.assertLessEqual(y.max(),1)

    def test_preserves_alpha_and_source(self):
        x=np.array([[2,4,8,0.25],[16,32,64,0.75]],dtype='<f2');before=x.tobytes()
        y=module.transform(x)
        self.assertEqual(before,x.tobytes())
        np.testing.assert_array_equal(y[:,3],x[:,3])

    def test_invalid_input_rejected(self):
        for value in (-1,np.inf,np.nan):
            with self.assertRaises(ValueError):
                module.transform(np.array([[value,0,0,1]],dtype='<f2'))


if __name__=='__main__':unittest.main()
