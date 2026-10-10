// Synthetic SM120 LG2 characterization. No model, game or reference input is read.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
constexpr unsigned G=64,N=1u<<23,SAMPLES=8192*254;
__device__ float lg2(float x){float y;asm volatile("lg2.approx.ftz.f32 %0, %1;":"=f"(y):"f"(x));return y;}
__global__ void sweep(unsigned* output,unsigned* coverage,unsigned count,int mode){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    unsigned exponent,mantissa;
    if(mode<2){exponent=mode?126:127;mantissa=i;}
    else{exponent=1+i/8192;mantissa=((i%8192)*0x9e3779b9u)&0x7fffffu;}
    output[i]=__float_as_uint(lg2(__uint_as_float((exponent<<23)|mantissa)));atomicAdd(coverage+i,1u);
}
static void check(cudaError_t value,const char* action){if(value!=cudaSuccess)throw std::runtime_error(std::string(action)+": "+cudaGetErrorString(value));}
static void save(const std::string& name,const std::vector<unsigned>& data){std::ifstream exists(name,std::ios::binary);if(exists.good())throw std::runtime_error("output exists");std::ofstream file(name,std::ios::binary);file.write((const char*)data.data(),data.size()*4);file.close();if(!file)throw std::runtime_error("write failed");}
int main(int argc,char** argv){try{
    if(argc!=2)throw std::runtime_error("usage: lg2-characterization <new output prefix>");
    cudaDeviceProp device{};check(cudaGetDeviceProperties(&device,0),"device");if(device.major!=12||device.minor!=0)throw std::runtime_error("SM120 required");
    printf("device=%s\ncompute=%d.%d\n",device.name,device.major,device.minor);
    unsigned *output,*coverage;size_t capacity=(size_t)(N+2*G)*4;
    check(cudaMalloc(&output,capacity),"output allocation");check(cudaMalloc(&coverage,capacity),"coverage allocation");
    for(int mode=0;mode<3;mode++){
        unsigned count=mode<2?N:SAMPLES;size_t bytes=(size_t)(count+2*G)*4;std::vector<unsigned> words(count+2*G),prior;
        for(int pass=0;pass<2;pass++){
            unsigned poison=pass?0x5a5a5a5a:0xa5a5a5a5;
            check(cudaMemset(output,pass?0x5a:0xa5,bytes),"output poison");check(cudaMemset(coverage,0x67,bytes),"coverage guard");check(cudaMemset(coverage+G,0,(size_t)count*4),"coverage zero");
            sweep<<<(count+255)/256,256>>>(output+G,coverage+G,count,mode);check(cudaGetLastError(),"launch");check(cudaDeviceSynchronize(),"completion");
            check(cudaMemcpy(words.data(),coverage,bytes,cudaMemcpyDeviceToHost),"coverage read");
            for(unsigned i=0;i<count+2*G;i++)if(words[i]!=(i<G||i>=count+G?0x67676767u:1u))throw std::runtime_error("coverage or guard failure");
            check(cudaMemcpy(words.data(),output,bytes,cudaMemcpyDeviceToHost),"output read");for(unsigned i=0;i<G;i++)if(words[i]!=poison||words[count+G+i]!=poison)throw std::runtime_error("output guard failure");
            if(pass){if(memcmp(prior.data(),words.data()+G,(size_t)count*4))throw std::runtime_error("poison-dependent result");}
            else prior.assign(words.begin()+G,words.begin()+G+count);
        }
        const char* suffix=mode==0?"-binade-zero.bin":mode==1?"-binade-minus-one.bin":"-exponent-samples.bin";
        save(std::string(argv[1])+suffix,prior);printf("mode=%d entries=%u passes=2 coverage_failures=0 canary_failures=0\n",mode,count);fflush(stdout);
    }
    cudaFree(output);cudaFree(coverage);
    std::ofstream receipt(std::string(argv[1])+"-receipt.json");receipt<<"{\"schema\":\"netal.lg2.characterization.v1\",\"model_inputs\":0,\"normalized_mantissas_per_band\":8388608,\"sample_mantissas_per_exponent\":8192,\"sample_exponents\":254,\"passes\":2,\"coverage_failures\":0,\"canary_failures\":0,\"general_reconstruction_verified\":false}\n";
    return 0;
}catch(const std::exception& error){fprintf(stderr,"error=%s\n",error.what());return 1;}}
