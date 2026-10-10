#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>

constexpr size_t count = 1u << 24, bytes = count * 4, guard = 256;
static std::string sha(const void* data) {
    unsigned char digest[32]; CC_SHA256(data, static_cast<CC_LONG>(bytes), digest);
    std::string result; const char* hex="0123456789abcdef";
    for (auto value: digest) { result+=hex[value>>4]; result+=hex[value&15]; }
    return result;
}
int main(int argc, const char** argv) { @autoreleasepool { try {
    if (argc!=4 || (std::string(argv[2])!="rcp" && std::string(argv[2])!="rsq")) throw std::runtime_error("usage: m3-baseline <metallib> rcp|rsq <new output>");
    if (std::filesystem::exists(argv[3])) throw std::runtime_error("Output exists");
    id<MTLDevice> device=MTLCreateSystemDefaultDevice(); if (!device) throw std::runtime_error("No Metal device");
    NSError* error=nil;
    id<MTLLibrary> library=[device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
    if (!library) throw std::runtime_error("Cannot load baseline library");
    NSString* function=[NSString stringWithFormat:@"netal_%s_baseline",argv[2]];
    id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:function] error:&error];
    if (!pipeline) throw std::runtime_error("Cannot build baseline pipeline");
    id<MTLCommandQueue> queue=[device newCommandQueue];
    std::string previous;
    for (unsigned pass=0; pass<2; ++pass) {
        id<MTLBuffer> output=[device newBufferWithLength:bytes+guard*2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> counters=[device newBufferWithLength:bytes+guard*2 options:MTLResourceStorageModeShared];
        if (!output || !counters) throw std::runtime_error("Cannot allocate sweep buffers");
        auto* data=static_cast<unsigned char*>(output.contents);
        auto* writes=static_cast<unsigned char*>(counters.contents);
        std::memset(data,pass?0x5a:0xa5,bytes+guard*2);std::memset(writes,0x67,bytes+guard*2);std::memset(writes+guard,0,bytes);
        id<MTLCommandBuffer> command=[queue commandBuffer];id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];[encoder setBuffer:output offset:guard atIndex:0];[encoder setBuffer:counters offset:guard atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[encoder endEncoding];[command commit];[command waitUntilCompleted];
        if (command.status!=MTLCommandBufferStatusCompleted) throw std::runtime_error("GPU command failed");
        for (size_t i=0;i<guard;++i) if (data[i]!=(pass?0x5a:0xa5) || data[guard+bytes+i]!=(pass?0x5a:0xa5) || writes[i]!=0x67 || writes[guard+bytes+i]!=0x67) throw std::runtime_error("Canary changed");
        auto* words=reinterpret_cast<uint32_t*>(writes+guard);
        for (size_t i=0;i<count;++i) if(words[i]!=1) throw std::runtime_error("Incomplete or duplicate coverage");
        auto digest=sha(data+guard);
        if (pass && digest!=previous) throw std::runtime_error("Output depends on poison");
        previous=digest;
        if (pass) {
            std::ofstream file(argv[3],std::ios::binary);file.write(reinterpret_cast<char*>(data+guard),bytes);
            if (!file) throw std::runtime_error("Output write failed");
        }
    }
    std::cout<<"operation="<<argv[2]<<"\npid="<<getpid()<<"\ndevice="<<device.name.UTF8String<<"\ncount="<<count<<"\npasses=2\ncount_failures=0\ncanary_failures=0\nsha256="<<previous<<"\nresult=pass\n";
    return 0;
} catch (const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; } } }
