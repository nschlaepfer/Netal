#!/usr/bin/env python3
"""CPU mock of the actual readback hook; no CUDA, SSH, or model execution."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('builder',Path(__file__).with_name('prepare-nr-history-capture.py'));builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
class History(unittest.TestCase):
    def test_host_checks_export_and_observes_around_evaluate(self):
        s=builder.history_host((ROOT/'build/nr_reset_sequence/source-v2/color_cases_trace.cpp').read_text())
        self.assertIn('if(!history_readback)throw',s)
        self.assertLess(s.index('observe_history(0);'),s.index('const NVSDK_NGX_Result evaluate_result ='))
        self.assertLess(s.index('const NVSDK_NGX_Result evaluate_result ='),s.index('observe_history(1);'))
        self.assertNotIn('CopyHostToArray(cu_memcpy_2d,history',s)
    def test_unknown_source_rejected(self):
        with self.assertRaises(ValueError):builder.history_host('unknown')
    def test_sampler_trace_does_not_modify_forwarded_descriptor(self):
        source=(ROOT/'build/nr_reset_sequence/source-v2/prefix_proxy.cpp').read_text();new=builder.sampler_trace(source)
        self.assertEqual(source.count('real(object, resource, texture, view)'),new.count('real(object, resource, texture, view)'))
        self.assertIn('std::memcpy(sampler,texture,sizeof(sampler))',new)
        self.assertIn('\\"filter_mode\\":%u',new)
        self.assertNotIn('std::memcpy(texture,',new)
    def test_actual_hook_direction_domains_and_failures(self):
        cpp=r'''
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstring>
#define WINAPI
#define __declspec(x)
using CUresult=int;
constexpr int kCudaSuccess=0,kCudaErrorInvalidValue=1,kCudaErrorNotFound=500;
struct ArrayRecord{void* handle;long id;size_t width,height;int format;unsigned channels;bool active;};
ArrayRecord g_arrays[1]={{reinterpret_cast<void*>(0x1234),1,1920,1080,16,4,true}};
struct CudaMemcpy2D{int source_memory_type;void* source_array;int destination_memory_type;void* destination_host;size_t destination_pitch,width_bytes,height;};
int sync_calls=0,copy_calls=0,trace_calls=0,sync_result=0,copy_result=0;bool available=true;
int sync_mock(){++sync_calls;return sync_result;}
int copy_mock(const void* raw){++copy_calls;auto& d=*static_cast<const CudaMemcpy2D*>(raw);assert(d.source_memory_type==3&&d.destination_memory_type==1);assert(d.source_array==g_arrays[0].handle&&d.destination_host);assert(d.width_bytes==15360&&d.destination_pitch==15360&&d.height==1080);return copy_result;}
using Sync=int(*)();using Copy=int(*)(const void*);
template<class T>T Real(const char*);
template<> Sync Real<Sync>(const char* n){assert(std::strcmp(n,"cuCtxSynchronize")==0);return available?sync_mock:nullptr;}
template<> Copy Real<Copy>(const char* n){assert(std::strcmp(n,"cuMemcpy2D")==0);return available?copy_mock:nullptr;}
template<size_t N,class...A>int sprintf_s(char(&b)[N],const char*f,A...a){return std::snprintf(b,N,f,a...);}
void TraceCall(const char* name,const char* fields){assert(std::strcmp(name,"history_readback")==0);assert(std::strstr(fields,"\"array_id\":1"));++trace_calls;}
#include "@HOOK@"
int main(){char dummy=0;constexpr size_t bytes=16588800;
assert(NeuralStudioCudaCaptureHistory(0,0,&dummy,bytes)==0);assert(sync_calls==1&&copy_calls==1);
assert(NeuralStudioCudaCaptureHistory(1,1,&dummy,bytes)==0);
const int before=copy_calls;
assert(NeuralStudioCudaCaptureHistory(2,0,&dummy,bytes)==1);
assert(NeuralStudioCudaCaptureHistory(0,2,&dummy,bytes)==1);
assert(NeuralStudioCudaCaptureHistory(0,0,nullptr,bytes)==1);
assert(NeuralStudioCudaCaptureHistory(0,0,&dummy,bytes-1)==1);
g_arrays[0].active=false;assert(NeuralStudioCudaCaptureHistory(0,0,&dummy,bytes)==1);g_arrays[0].active=true;
g_arrays[0].width=1919;assert(NeuralStudioCudaCaptureHistory(0,0,&dummy,bytes)==1);g_arrays[0].width=1920;
g_arrays[0].format=32;assert(NeuralStudioCudaCaptureHistory(0,0,&dummy,bytes)==1);g_arrays[0].format=16;
assert(copy_calls==before&&sync_calls==2);
sync_result=77;assert(NeuralStudioCudaCaptureHistory(0,0,&dummy,bytes)==77);assert(copy_calls==before);sync_result=0;
copy_result=55;assert(NeuralStudioCudaCaptureHistory(0,0,&dummy,bytes)==55);copy_result=0;
available=false;assert(NeuralStudioCudaCaptureHistory(0,0,&dummy,bytes)==500);assert(trace_calls==12);
}
'''.replace('@HOOK@',str(ROOT/'src/nr_probes/nr_history_readback.inl'))
        with tempfile.TemporaryDirectory(prefix='nr-history-hook-test-') as tmp:
            exe=Path(tmp)/'test';subprocess.run(['clang++','-std=c++20','-Wall','-Wextra','-Werror','-x','c++','-','-o',str(exe)],input=cpp,text=True,check=True,capture_output=True);subprocess.run([str(exe)],check=True,capture_output=True)
if __name__=='__main__':unittest.main(verbosity=2)
