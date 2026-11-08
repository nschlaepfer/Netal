// Original model-free operator lab using Apple's public MSL 4.1 API.
// https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
// Direct FP8 input TensorOps; no FP8-to-half conversion in the matmul path.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct LabParams { uint m,n,k,paddedK,aStride,bStride,yStride; float scaleA,scaleB; };

template<typename Format>
inline void pack_operand(device const float* source,device uchar* packed,
                         device float* decoded,uint index,float scale) {
    const uint offset=index*4;
    const float4 values=float4(source[offset],source[offset+1],source[offset+2],source[offset+3])/scale;
    const auto value=pack<Format,rounding_mode::to_nearest_even,saturation_mode::saturate>(values);
    const auto bytes=value.as_storage_type();const auto unpacked=unpack<float>(value);
    for(uint j=0;j<4;++j){packed[offset+j]=bytes[j];decoded[offset+j]=unpacked[j]*scale;}
}
kernel void netal_lab_pack_e4m3(device const float* source [[buffer(0)]],device uchar* packed [[buffer(1)]],
    device float* decoded [[buffer(2)]],constant float& scale [[buffer(3)]],uint index [[thread_position_in_grid]]) {
    pack_operand<metal_fp8_e4m3_format>(source,packed,decoded,index,scale);
}
kernel void netal_lab_pack_e5m2(device const float* source [[buffer(0)]],device uchar* packed [[buffer(1)]],
    device float* decoded [[buffer(2)]],constant float& scale [[buffer(3)]],uint index [[thread_position_in_grid]]) {
    pack_operand<metal_fp8_e5m2_format>(source,packed,decoded,index,scale);
}

template<typename Format,typename Storage>
inline void matmul_body(device Storage* a,device Storage* b,device float* y,
                        constant LabParams& p,device const float* bias,uint3 tg) {
    const uint row=tg.x*64,col=tg.y*32;
    if(row>=p.m||col>=p.n)return;
    using Input=tensor<device Format,dextents<int,2>,tensor_inline>;
    // FP8 inline tensor bases must be 128-byte aligned. Keep the full aligned
    // allocation as the tensor base and express K/N tile offsets as slices;
    // constructing a fresh tensor from b+32 or a+32 would break that contract.
    auto fullA=Input(a,dextents<int,2>(int(p.paddedK),int(p.m)),array<int,2>{1,int(p.aStride)});
    auto fullB=Input(b,dextents<int,2>(int(p.bStride),int(p.paddedK)),array<int,2>{1,int(p.bStride)});
    auto A=fullA.slice(0,row);
    auto B=fullB.slice(col,0);
    constexpr auto descriptor=matmul2d_descriptor(64,32,32,false,false,false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor,execution_simdgroups<4>> op;
    auto C=op.template get_destination_cooperative_tensor<decltype(A),decltype(B),float>();
    for(uint16_t i=0;i<C.get_capacity();++i)if(C.is_valid_element(i))C[i]=0.0f;
    for(uint k=0;k<p.paddedK;k+=32) {
        auto sliceA=fullA.slice(k,row);
        auto sliceB=fullB.slice(col,k);
        op.run(sliceA,sliceB,C);
    }
    for(uint16_t i=0;i<C.get_capacity();++i)if(C.is_valid_element(i)) {
        const auto index=C.get_multidimensional_index(i);
        const uint rr=row+uint(index[1]),cc=col+uint(index[0]);
        // Bias is in real units, after rescaling the FP8 accumulator. It must
        // never initialize C in real units before the scale product is applied.
        if(rr<p.m&&cc<p.n)y[rr*p.yStride+cc]=C[i]*(p.scaleA*p.scaleB)+bias[cc];
    }
}
kernel void netal_lab_e4m3(device uchar* a [[buffer(0)]],device uchar* b [[buffer(1)]],
    device float* y [[buffer(2)]],constant LabParams& p [[buffer(3)]],device const float* bias [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]]) {matmul_body<metal_fp8_e4m3_format>(a,b,y,p,bias,tg);}
kernel void netal_lab_e5m2(device uchar* a [[buffer(0)]],device uchar* b [[buffer(1)]],
    device float* y [[buffer(2)]],constant LabParams& p [[buffer(3)]],device const float* bias [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]]) {matmul_body<metal_fp8_e5m2_format>(a,b,y,p,bias,tg);}
kernel void netal_lab_fp16(device half* a [[buffer(0)]],device half* b [[buffer(1)]],
    device float* y [[buffer(2)]],constant LabParams& p [[buffer(3)]],device const float* bias [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]]) {matmul_body<half>(a,b,y,p,bias,tg);}
