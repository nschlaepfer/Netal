#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <vector>
#include <array>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

static constexpr NSUInteger G=256;
struct NetalDimensions { uint32_t width,height; };
static id<MTLComputePipelineState> pipeline(id<MTLDevice>d,id<MTLLibrary>l,NSString*n) {
    NSError* error=nil;auto p=[d newComputePipelineStateWithFunction:[l newFunctionWithName:n] error:&error];
    if(!p)throw std::runtime_error(std::string("pipeline: ")+n.UTF8String);return p;
}
static void finish(id<MTLCommandBuffer>c) {[c commit];[c waitUntilCompleted];if(c.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("GPU command failed");}
static id<MTLTexture> texture(id<MTLDevice>d,MTLPixelFormat f,NSUInteger w,NSUInteger h,MTLTextureUsage usage) {
    auto desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:f width:w height:h mipmapped:NO];desc.usage=usage;desc.storageMode=MTLStorageModeShared;
    auto t=[d newTextureWithDescriptor:desc];if(!t)throw std::runtime_error("texture allocation");return t;
}
static id<MTLBuffer> guarded(id<MTLDevice>d,NSUInteger n) {auto b=[d newBufferWithLength:n+2*G options:MTLResourceStorageModeShared];if(!b)throw std::runtime_error("buffer allocation");memset(b.contents,0xa5,b.length);return b;}
static void guards(id<MTLBuffer>b,NSUInteger n) {auto*p=(uint8_t*)b.contents;for(NSUInteger i=0;i<G;i++)if(p[i]!=0xa5||p[G+n+i]!=0xa5)throw std::runtime_error("guard changed");}
static void compare(const void*a,const void*b,NSUInteger n,const char*name) {
    auto*x=(const uint8_t*)a;auto*y=(const uint8_t*)b;NSUInteger bad=0,first=n;
    for(NSUInteger i=0;i<n;i++)if(x[i]!=y[i]){if(first==n)first=i;bad++;}
    std::cout<<"case="<<name<<" bytes="<<n<<" mismatches="<<bad;if(bad)std::cout<<" first="<<first<<" reference="<<unsigned(x[first])<<" candidate="<<unsigned(y[first]);std::cout<<'\n';
    if(bad)throw std::runtime_error("converter mismatch");
}
static id<MTLBuffer> runImport(id<MTLDevice>d,id<MTLCommandQueue>q,id<MTLLibrary>l,NSString*name,id<MTLTexture>t,NetalDimensions size,NSUInteger stride,bool dimensions=true) {
    NSUInteger n=size.width*size.height*stride;auto out=guarded(d,n);auto c=[q commandBuffer];auto e=[c computeCommandEncoder];
    [e setComputePipelineState:pipeline(d,l,name)];[e setTexture:t atIndex:0];[e setBuffer:out offset:G atIndex:0];if(dimensions)[e setBytes:&size length:sizeof(size) atIndex:1];
    [e dispatchThreads:MTLSizeMake(dimensions?size.width+3:1,dimensions?size.height+2:1,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];[e endEncoding];finish(c);guards(out,n);return out;
}
static void runMetal4(id<MTLDevice>d,id<MTLLibrary>library,NSString*name,id<MTLTexture>texture,id<MTLBuffer>buffer,NetalDimensions size) {
    auto queue=[d newMTL4CommandQueue];auto allocator=[d newCommandAllocator];auto command=[d newCommandBuffer];auto done=[d newSharedEvent];
    auto constants=[d newBufferWithLength:64 options:MTLResourceStorageModeShared];memset(constants.contents,0,64);memcpy((uint8_t*)constants.contents+32,&size,sizeof(size));
    auto rd=[MTLResidencySetDescriptor new];rd.initialCapacity=3;auto residency=[d newResidencySetWithDescriptor:rd error:nil];
    [residency addAllocation:texture];[residency addAllocation:buffer];[residency addAllocation:constants];[residency commit];
    auto td=[MTL4ArgumentTableDescriptor new];td.maxBufferBindCount=2;td.maxTextureBindCount=1;auto table=[d newArgumentTableWithDescriptor:td error:nil];
    if(!queue||!allocator||!command||!done||!constants||!residency||!table)throw std::runtime_error("Metal 4 setup");
    [table setAddress:buffer.gpuAddress+G atIndex:0];[table setAddress:constants.gpuAddress+32 atIndex:1];[table setTexture:texture.gpuResourceID atIndex:0];
    [command beginCommandBufferWithAllocator:allocator];[command useResidencySet:residency];auto encoder=[command computeCommandEncoder];[encoder setComputePipelineState:pipeline(d,library,name)];[encoder setArgumentTable:table];[encoder dispatchThreads:MTLSizeMake(size.width+2,size.height+2,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];[encoder endEncoding];[command endCommandBuffer];
    id<MTL4CommandBuffer> list[1]={command};[queue commit:list count:1];[queue signalEvent:done value:1];if(![done waitUntilSignaledValue:1 timeoutMS:20000])throw std::runtime_error("Metal 4 timeout");
}
int main(int argc,const char**argv){@autoreleasepool{try{
    if(argc!=3)throw std::runtime_error("usage: converter-probe <reference> <candidate>");auto d=MTLCreateSystemDefaultDevice();if(!d)throw std::runtime_error("no Metal device");auto q=[d newCommandQueue];
    NSError* error=nil;id<MTLLibrary> libs[2]={[d newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error],[d newLibraryWithURL:[NSURL fileURLWithPath:@(argv[2])] error:&error]};
    if(!libs[0]||!libs[1])throw std::runtime_error("library load");
    NetalDimensions size{257,128};NSUInteger pixels=size.width*size.height;
    auto color=texture(d,MTLPixelFormatRGB9E5Float,size.width+3,size.height+2,MTLTextureUsageShaderRead);
    std::vector<uint32_t> packed(color.width*color.height,0x12345678);uint32_t random=37;
    for(NSUInteger y=0;y<size.height;y++)for(NSUInteger x=0;x<size.width;x++){
        NSUInteger i=y*size.width+x;uint32_t word;
        if(i<16384){uint32_t m=i&511,e=i>>9;word=(e<<27)|((511-m)<<18)|(((m*37+17)&511)<<9)|m;}
        else{random^=random<<13;random^=random>>17;random^=random<<5;word=random;}
        packed[y*color.width+x]=word;
    }
    [color replaceRegion:MTLRegionMake2D(0,0,color.width,color.height) mipmapLevel:0 withBytes:packed.data() bytesPerRow:color.width*4];
    auto a=runImport(d,q,libs[0],@"neuralstudio_sr_import_rgb9e5",color,size,8);auto b=runImport(d,q,libs[1],@"neuralstudio_sr_import_rgb9e5",color,size,8);
    compare((uint8_t*)a.contents+G,(uint8_t*)b.contents+G,pixels*8,"rgb9e5-import");
    auto* imported=(const uint16_t*)((uint8_t*)b.contents+G);
    for(NSUInteger y=0;y<size.height;y++)for(NSUInteger x=0;x<size.width;x++){
        uint32_t word=packed[y*color.width+x],exponent=word>>27;NSUInteger pixel=y*size.width+x;
        for(unsigned channel=0;channel<3;channel++){
            _Float16 value=static_cast<_Float16>(std::ldexp(float((word>>(9*channel))&511u),int(exponent)-24));uint16_t bits;memcpy(&bits,&value,2);
            if(imported[pixel*4+channel]!=bits)throw std::runtime_error("RGB9E5 CPU decode mismatch");
        }
        if(imported[pixel*4+3]!=0x3c00)throw std::runtime_error("RGB9E5 alpha mismatch");
    }
    std::cout<<"rgb9e5_cpu_decode=pass pixels="<<pixels<<'\n';

    // Every binary16 red/green pattern plus independently varied blue/alpha; output padding stays untouched.
    NetalDimensions outputSize{256,256};NSUInteger outputPixels=65536;auto input=guarded(d,outputPixels*8);auto* half=(uint16_t*)((uint8_t*)input.contents+G);
    for(NSUInteger i=0;i<outputPixels;i++){half[i*4]=i;half[i*4+1]=i^0x8000;half[i*4+2]=(i*257+17)&65535;half[i*4+3]=0x3c00;}
    std::vector<uint32_t> exports[2];
    for(unsigned version=0;version<2;version++){
        auto t=texture(d,MTLPixelFormatRGB9E5Float,258,258,MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite);std::vector<uint32_t> data(t.width*t.height,0x2468ace0);
        [t replaceRegion:MTLRegionMake2D(0,0,t.width,t.height) mipmapLevel:0 withBytes:data.data() bytesPerRow:t.width*4];
        auto c=[q commandBuffer];auto e=[c computeCommandEncoder];[e setComputePipelineState:pipeline(d,libs[version],@"neuralstudio_sr_export_rgb9e5")];[e setBuffer:input offset:G atIndex:0];[e setTexture:t atIndex:0];[e setBytes:&outputSize length:sizeof(outputSize) atIndex:1];[e dispatchThreads:MTLSizeMake(t.width,t.height,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];[e endEncoding];finish(c);guards(input,outputPixels*8);
        [t getBytes:data.data() bytesPerRow:t.width*4 fromRegion:MTLRegionMake2D(0,0,t.width,t.height) mipmapLevel:0];
        for(NSUInteger y=0;y<t.height;y++)for(NSUInteger x=0;x<t.width;x++)if((x>=outputSize.width||y>=outputSize.height)&&data[y*t.width+x]!=0x2468ace0)throw std::runtime_error("output margin changed");
        exports[version]=std::move(data);
    }
    compare(exports[0].data(),exports[1].data(),exports[0].size()*4,"rgb9e5-export");
    // All binary16 encodings, in bounded batches of one-pixel array slices.
    auto exposureDesc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float width:1 height:1 mipmapped:NO];exposureDesc.textureType=MTLTextureType2DArray;exposureDesc.arrayLength=256;exposureDesc.storageMode=MTLStorageModeShared;exposureDesc.usage=MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;
    auto exposureArray=[d newTextureWithDescriptor:exposureDesc];if(!exposureArray)throw std::runtime_error("exposure array allocation");
    NSMutableArray<id<MTLTexture>>* views=[NSMutableArray array];
    for(NSUInteger i=0;i<256;i++){auto view=[exposureArray newTextureViewWithPixelFormat:MTLPixelFormatR16Float textureType:MTLTextureType2D levels:NSMakeRange(0,1) slices:NSMakeRange(i,1)];if(!view)throw std::runtime_error("exposure view");[views addObject:view];}
    id<MTLBuffer> exposures[2]={guarded(d,65536*4),guarded(d,65536*4)};
    id<MTLComputePipelineState> exposurePipelines[2]={pipeline(d,libs[0],@"neuralstudio_sr_import_exposure_r16"),pipeline(d,libs[1],@"neuralstudio_sr_import_exposure_r16")};
    for(NSUInteger batch=0;batch<256;batch++){
        for(NSUInteger i=0;i<256;i++){uint16_t value=batch*256+i;[exposureArray replaceRegion:MTLRegionMake2D(0,0,1,1) mipmapLevel:0 slice:i withBytes:&value bytesPerRow:2 bytesPerImage:2];}
        auto command=[q commandBuffer];auto encoder=[command computeCommandEncoder];
        for(unsigned version=0;version<2;version++){
            [encoder setComputePipelineState:exposurePipelines[version]];
            for(NSUInteger i=0;i<256;i++){[encoder setTexture:views[i] atIndex:0];[encoder setBuffer:exposures[version] offset:G+(batch*256+i)*4 atIndex:0];[encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];}
        }
        [encoder endEncoding];finish(command);
    }
    guards(exposures[0],65536*4);guards(exposures[1],65536*4);
    compare((uint8_t*)exposures[0].contents+G,(uint8_t*)exposures[1].contents+G,65536*4,"r16-exposure-all-encodings");
    // Nonuniform depth/stencil texture, including padded allocation beyond the content rectangle.
    NetalDimensions depthSize{31,17},physical{34,19};
    auto desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8 width:physical.width height:physical.height mipmapped:NO];desc.storageMode=MTLStorageModePrivate;desc.usage=MTLTextureUsageShaderRead|MTLTextureUsageRenderTarget;
    auto depth=[d newTextureWithDescriptor:desc];if(!depth)throw std::runtime_error("depth allocation");
    const char* source=R"MSL(#include <metal_stdlib>
using namespace metal;
vertex float4 fill_vertex(uint index [[vertex_id]]) { float2 positions[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};return float4(positions[index],0,1); }
struct DepthOutput { float depth [[depth(any)]]; };
fragment DepthOutput fill_depth(float4 position [[position]],device const float* values [[buffer(0)]],constant uint2& size [[buffer(1)]]) {uint2 p=uint2(position.xy);return DepthOutput{values[p.y*size.x+p.x]};}
)MSL";
    auto fillLibrary=[d newLibraryWithSource:@(source) options:nil error:&error];if(!fillLibrary)throw std::runtime_error(error.localizedDescription.UTF8String);
    auto pd=[MTLRenderPipelineDescriptor new];pd.vertexFunction=[fillLibrary newFunctionWithName:@"fill_vertex"];pd.fragmentFunction=[fillLibrary newFunctionWithName:@"fill_depth"];pd.depthAttachmentPixelFormat=MTLPixelFormatDepth32Float_Stencil8;pd.stencilAttachmentPixelFormat=MTLPixelFormatDepth32Float_Stencil8;
    auto ps=[d newRenderPipelineStateWithDescriptor:pd error:&error];if(!ps)throw std::runtime_error("depth fill pipeline");
    std::vector<float> values(physical.width*physical.height);for(NSUInteger i=0;i<values.size();i++)values[i]=float(i%509)/508.0f;
    auto buffer=[d newBufferWithBytes:values.data() length:values.size()*4 options:MTLResourceStorageModeShared];
    auto ds=[MTLDepthStencilDescriptor new];ds.depthCompareFunction=MTLCompareFunctionAlways;ds.depthWriteEnabled=YES;auto state=[d newDepthStencilStateWithDescriptor:ds];
    auto pass=[MTLRenderPassDescriptor renderPassDescriptor];pass.depthAttachment.texture=depth;pass.depthAttachment.loadAction=MTLLoadActionClear;pass.depthAttachment.storeAction=MTLStoreActionStore;pass.depthAttachment.clearDepth=1;pass.stencilAttachment.texture=depth;pass.stencilAttachment.loadAction=MTLLoadActionClear;pass.stencilAttachment.storeAction=MTLStoreActionStore;pass.stencilAttachment.clearStencil=193;
    auto command=[q commandBuffer];auto encoder=[command renderCommandEncoderWithDescriptor:pass];[encoder setRenderPipelineState:ps];[encoder setDepthStencilState:state];[encoder setFragmentBuffer:buffer offset:0 atIndex:0];[encoder setFragmentBytes:&physical length:sizeof(physical) atIndex:1];[encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];[encoder endEncoding];finish(command);
    auto first=runImport(d,q,libs[0],@"neuralstudio_sr_import_depth32_stencil8",depth,depthSize,4);auto second=runImport(d,q,libs[1],@"neuralstudio_sr_import_depth32_stencil8",depth,depthSize,4);
    compare((uint8_t*)first.contents+G,(uint8_t*)second.contents+G,depthSize.width*depthSize.height*4,"depth-import");
    auto* depths=(const float*)((uint8_t*)second.contents+G);
    for(NSUInteger y=0;y<depthSize.height;y++)for(NSUInteger x=0;x<depthSize.width;x++)if(depths[y*depthSize.width+x]!=values[y*physical.width+x])throw std::runtime_error("depth CPU mismatch");
    if([d supportsFamily:MTLGPUFamilyMetal4]){
        auto imported4=guarded(d,pixels*8);runMetal4(d,libs[1],@"neuralstudio_sr_import_rgb9e5",color,imported4,size);guards(imported4,pixels*8);
        compare((uint8_t*)a.contents+G,(uint8_t*)imported4.contents+G,pixels*8,"metal4-import");
        auto exported4=texture(d,MTLPixelFormatRGB9E5Float,258,258,MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite);std::vector<uint32_t> data(258*258,0x2468ace0);
        [exported4 replaceRegion:MTLRegionMake2D(0,0,258,258) mipmapLevel:0 withBytes:data.data() bytesPerRow:258*4];
        runMetal4(d,libs[1],@"neuralstudio_sr_export_rgb9e5",exported4,input,outputSize);guards(input,outputPixels*8);
        [exported4 getBytes:data.data() bytesPerRow:258*4 fromRegion:MTLRegionMake2D(0,0,258,258) mipmapLevel:0];compare(exports[0].data(),data.data(),data.size()*4,"metal4-export");
    }else std::cout<<"metal4=unverified\n";
    std::cout<<"result=pass device="<<d.name.UTF8String<<" depth_tested=true\n";return 0;
}catch(const std::exception&e){std::cerr<<"error="<<e.what()<<'\n';return 1;}}}
