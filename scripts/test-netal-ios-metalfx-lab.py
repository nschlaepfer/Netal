#!/usr/bin/env python3
"""Model/device-free checks for padded transfers and MetalFX lab build boundaries.

The archive build separately compiles the real SDK API. Neither check simulates
successful MetalFX execution; texture roundtrips run only inside the device lab.
"""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('metalfx_builder',ROOT/'scripts/build-netal-ios-metalfx-lab.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)

class Tests(unittest.TestCase):
    def test_padded_layout_geometry_and_independent_row_copy(self):
        compiler=shutil.which('clang++') or shutil.which('c++')
        if not compiler:self.skipTest('C++ compiler unavailable')
        source=r'''
#include "NetalMetalFXLabSupport.h"
#include <cassert>
#include <cstring>
#include <vector>
int main(){
 using namespace netal_metalfx;
 for(unsigned w:{1u,427u,854u,960u,1280u,1920u})
  for(unsigned h:{1u,3u,240u})for(unsigned bpp:{2u,4u,8u}){
   std::size_t row=0,bytes=0;assert(RowLayout(w,h,bpp,&row,&bytes));
   assert(row%256==0&&row>=w*bpp&&row-w*bpp<256);
   assert(bytes==row*h+512);
   std::vector<unsigned char> packed(bytes,0xa5),tightly(w*h*bpp),roundtrip(bytes,0xa5);
   for(std::size_t i=0;i<tightly.size();++i)tightly[i]=(i*37+19)&255;
   for(unsigned y=0;y<h;++y)std::memcpy(packed.data()+kGuard+y*row,tightly.data()+y*w*bpp,w*bpp);
   // Independent texture-row simulation: the GPU must copy logical width, not padded stride.
   for(unsigned y=0;y<h;++y)std::memcpy(roundtrip.data()+kGuard+y*row,packed.data()+kGuard+y*row,w*bpp);
   assert(roundtrip==packed);
   for(std::size_t i=0;i<kGuard;++i)assert(packed[i]==0xa5&&packed[bytes-1-i]==0xa5);
  }
 std::size_t row=0,bytes=0;
 for(auto dims:{netal_probe::Dimensions{0,1},{1,0},{4097,1},{1,4097}})
  assert(!RowLayout(dims.width,dims.height,8,&row,&bytes));
 assert(!RowLayout(640,360,3,&row,&bytes));
 assert(ValidGeometry({640,360},{1280,720}));
 assert(ValidGeometry({427,240},{1280,720}));
 assert(ValidGeometry({960,540},{1920,1080}));
 assert(!ValidGeometry({1280,720},{640,360}));
 assert(!ValidGeometry({640,480},{1280,720}));
 assert(!ValidGeometry({8192,8192},{8192,8192}));
 assert(WithinScaleRange({640,360},{1280,720},1,2));
 assert(!WithinScaleRange({640,360},{1920,1080},1,2));
 assert(!WithinScaleRange({640,360},{1280,720},2,1));
 assert(!WithinScaleRange({640,360},{1280,720},0,2));
 // Independent reference for the shared, versioned scene. Compare the actual
 // binary16 values transferred to Metal, including nonzero jitter and history.
 unsigned long long identity=1469598103934665603ull;
 for(auto size:{netal_probe::Dimensions{854,480},{640,360},{427,240},{960,540}})
  for(unsigned frame:{0u,17u,18u,135u})for(unsigned sy:{0u,1u,7u})for(unsigned sx:{0u,3u,11u}){
   const double x=(size.width-1)*sx/11.0,y=(size.height-1)*sy/7.0,jx=double(-.3125f),jy=double(-.351851851f);
   auto actual=netal_probe::PlaneColor(x,y,jx,jy,size,frame);
   const double u=(x+.5+jx)/size.width-double(frame)*(1.0/1280.0),v=(y+.5+jy)/size.height;
   const float wave=float(.5+.5*std::sin(6.28318530717958647692*12*u)*std::cos(6.28318530717958647692*7*v));
   const float reference[4]={2.f*wave,.25f+.7f*wave,1.f-.8f*wave,1.f};
   for(unsigned c=0;c<4;++c){
    _Float16 a=(_Float16)actual[c],b=(_Float16)reference[c];
    unsigned short ab=0,bb=0;std::memcpy(&ab,&a,2);std::memcpy(&bb,&b,2);assert(ab==bb);
    identity^=ab;identity*=1099511628211ull;
   }
  }
 // Independently encoded with binary32 arithmetic + Python binary16 RNE.
 assert(identity==0xe718a8ac4340bef1ull);
}
'''
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);(folder/'check.cpp').write_text(source)
            subprocess.run([compiler,'-std=c++17','-Wall','-Wextra','-Werror','-I',str(ROOT/'src/ios_metalfx_lab'),
                            str(folder/'check.cpp'),'-o',str(folder/'check')],check=True)
            subprocess.run([str(folder/'check')],check=True)

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);(folder/'important').write_text('preserve')
            with self.assertRaises(ValueError):builder.build(folder)
            self.assertEqual((folder/'important').read_text(),'preserve')

    def test_symlink_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);link=folder/'new';link.symlink_to(folder/'missing')
            with self.assertRaises(ValueError):builder.build(link)
            self.assertTrue(link.is_symlink())

    def test_deployment_rejected_before_compiler(self):
        with tempfile.TemporaryDirectory() as temp:
            for target in ['25.0','27.0;false','bad']:
                with self.subTest(target=target),self.assertRaises(ValueError):
                    builder.build(Path(temp)/'new',target)

if __name__=='__main__':unittest.main()
