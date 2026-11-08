#pragma once
#import <Foundation/Foundation.h>
#include "NetalProbeMetrics.h"

namespace netal_probe {
inline bool DispatchRingFits(NSUInteger baseline,NSUInteger outputRepeat) {
    return outputRepeat>=1&&outputRepeat<=8&&baseline>=3&&baseline<=160&&outputRepeat-1<=160-baseline;
}
inline bool FP8Bool(id value,bool expected){bool parsed=false;return value&&ReadOptionalBoolean(value,false,&parsed)&&parsed==expected;}
inline bool FP8U32(id value,std::uint64_t* out) {
    if(![value isKindOfClass:NSNumber.class] || CFGetTypeID((__bridge CFTypeRef)value)==CFBooleanGetTypeID())return false;
    const double n=[value doubleValue];
    if(!std::isfinite(n)||n<0||n>UINT32_MAX||n!=std::floor(n))return false;
    *out=static_cast<std::uint64_t>(n);return true;
}
inline bool FP8Hex(id value) {
    if(![value isKindOfClass:NSString.class]||[value length]!=64)return false;
    NSCharacterSet* hex=[NSCharacterSet characterSetWithCharactersInString:@"0123456789abcdef"];
    return [value rangeOfCharacterFromSet:hex.invertedSet].location==NSNotFound;
}
inline NSDictionary* FP8Binding(NSDictionary* d,NSUInteger index) {
    if(![d[@"buffers"] isKindOfClass:NSArray.class])return nil;
    NSDictionary* found=nil;
    for(id b in d[@"buffers"]){if(![b isKindOfClass:NSDictionary.class])return nil;
        if([b[@"index"] isEqual:@(index)]){if(found)return nil;found=b;}}
    return found;
}
inline NSArray* FP8Words(NSDictionary* d,NSUInteger index,NSUInteger count) {
    NSArray* bytes=d[@"bytes"];
    if(![bytes isKindOfClass:NSArray.class]||bytes.count!=1||![bytes[0] isKindOfClass:NSDictionary.class])return nil;
    if(![bytes[0][@"index"] isEqual:@(index)])return nil;
    NSArray* p=bytes[0][@"u32"];
    if(![p isKindOfClass:NSArray.class]||p.count!=count)return nil;
    for(id n in p){std::uint64_t v=0;if(!FP8U32(n,&v))return nil;}return p;
}
inline bool FP8SameBinding(NSDictionary* a,NSDictionary* b) {
    return a&&b&&[a[@"name"] isEqual:b[@"name"]]&&[a[@"offset"] isEqual:b[@"offset"]];
}
inline bool FP8Span(NSDictionary* b,std::uint64_t bytes,NSDictionary* capacities,NSUInteger alignment=2) {
    std::uint64_t offset=0,capacity=0;
    return b&&[b[@"name"] isKindOfClass:NSString.class]&&FP8U32(b[@"offset"],&offset)&&
        FP8U32(capacities[b[@"name"]],&capacity)&&offset%alignment==0&&offset<=capacity&&bytes<=capacity-offset;
}
// Structural validation is deliberately separate from operator-specific read/write spans.
// A valid base binding does not establish the memory safety of an arbitrary original kernel.
inline bool FP8DispatchStructure(NSDictionary* d,NSDictionary* capacities) {
    NSArray* buffers=d[@"buffers"];NSArray* bytes=d[@"bytes"];
    if(![buffers isKindOfClass:NSArray.class]||buffers.count==0||buffers.count>17||
       ![bytes isKindOfClass:NSArray.class]||bytes.count!=1)return false;
    NSMutableSet* indices=[NSMutableSet set];
    for(id raw in buffers){
        if(![raw isKindOfClass:NSDictionary.class])return false;
        NSDictionary* b=raw;std::uint64_t index=0;
        if(!FP8U32(b[@"index"],&index)||index>16||[indices containsObject:@(index)]||!FP8Span(b,2,capacities))return false;
        [indices addObject:@(index)];
    }
    id raw=bytes[0];if(![raw isKindOfClass:NSDictionary.class])return false;
    NSDictionary* value=raw;std::uint64_t index=0;
    if(!FP8U32(value[@"index"],&index)||index>16||[indices containsObject:@(index)]||
       ![value[@"u32"] isKindOfClass:NSArray.class]||[value[@"u32"] count]==0||[value[@"u32"] count]>1024)return false;
    for(id word in value[@"u32"]){std::uint64_t n=0;if(!FP8U32(word,&n))return false;}
    for(NSString* key in @[@"grid",@"block"]){
        NSArray* dimensions=d[key];if(![dimensions isKindOfClass:NSArray.class]||dimensions.count!=3)return false;
        for(id dimension in dimensions){std::uint64_t n=0;if(!FP8U32(dimension,&n)||n==0)return false;}
    }
    id constants=d[@"constants"];
    if(constants){
        if(![constants isKindOfClass:NSDictionary.class])return false;
        for(id key in constants){
            if(![key isKindOfClass:NSString.class])return false;
            const long long constantIndex=[key longLongValue];std::uint64_t n=0;
            if(constantIndex<0||constantIndex>65535||![[NSString stringWithFormat:@"%lld",constantIndex] isEqual:key]||
               !FP8U32(constants[key],&n))return false;
        }
    }
    return true;
}
inline bool FP8BindingSet(NSDictionary* d,NSArray* required) {
    if([d[@"buffers"] count]!=required.count)return false;
    for(NSNumber* index in required)if(!FP8Binding(d,index.unsignedIntegerValue))return false;
    return true;
}
inline bool FP8MatrixBytes(std::uint64_t rows,std::uint64_t stride,std::uint64_t columns,
                            std::uint64_t elementBytes,std::uint64_t* bytes) {
    if(!rows||rows>UINT32_MAX||!columns||columns>UINT32_MAX||stride>UINT32_MAX||
       (stride<columns&&!(rows==1&&stride==0))||!elementBytes||elementBytes>4)return false;
    // U32 operands make the product/addition safe in U64. Bound before multiplying by element size.
    const std::uint64_t elements=(rows-1)*stride+columns;
    if(!elements||elements-1>UINT32_MAX||elements>UINT32_MAX/elementBytes)return false;
    *bytes=elements*elementBytes;return true;
}
inline bool FP8MatrixBinding(NSDictionary* d,NSUInteger index,std::uint64_t rows,std::uint64_t stride,
                              std::uint64_t columns,std::uint64_t elementBytes,NSDictionary* capacities) {
    std::uint64_t bytes=0;
    return FP8MatrixBytes(rows,stride,columns,elementBytes,&bytes)&&FP8Span(FP8Binding(d,index),bytes,capacities,elementBytes);
}
inline bool FP8EpilogueSpans(NSDictionary* d,std::uint64_t tokens,std::uint64_t initialColumns,
                             std::uint64_t storedColumns,std::uint64_t cs,std::uint64_t mode,
                             std::uint64_t epilogue,std::uint64_t rs,NSDictionary* capacities) {
    // Buffer2 is a typed float pointer even when this particular mode does not read it.
    if(!FP8Span(FP8Binding(d,2),4,capacities,4))return false;
    if(mode==0){
        if((cs&&cs<initialColumns)||!FP8MatrixBinding(d,2,cs?std::min(tokens,std::uint64_t(16)):1,
            cs,initialColumns,4,capacities))return false;
    }else if(mode==1){
        if(cs<initialColumns||!FP8MatrixBinding(d,5,tokens,cs,initialColumns,2,capacities)||
           !FP8MatrixBinding(d,6,1,0,initialColumns,2,capacities))return false;
    }else return false;
    return epilogue!=3||(rs>=storedColumns&&FP8MatrixBinding(d,5,tokens,rs,storedColumns,2,capacities));
}
inline NSDictionary* AuditFP8Descriptor(NSDictionary* q,NSString* layer,NSDictionary* ledger,
                                       NSDictionary* tensors,NSDictionary* arenas,NSString* packRoot) {
    auto fail=[](NSString* reason){return @{@"validated":@NO,@"reason":reason};};
    if(![q isKindOfClass:NSDictionary.class]||![q[@"library"] isEqual:@"pwin_mpp_all"]||
       ![q[@"gemm_backend"] isEqual:@"fp8-e4m3"]||![q[@"mlp_backend"] isEqual:@"mpp-half"]||
       !FP8Bool(q[@"requires_ordered_dispatches"],true)||!FP8Bool(q[@"candidate_approximate"],true)||
       !FP8Bool(q[@"reference_exact"],false)||![q[@"fp8_pack_root_sha256"] isEqual:packRoot])
        return fail(@"FP8 backend identity/ordered-dispatch contract mismatch");
    NSDictionary* fs=q[@"fp8_scratch"];
    std::uint64_t legacy=0,aq=0,aqBytes=0,sa=0,saBytes=0,total=0;
    if(![fs isKindOfClass:NSDictionary.class]||!FP8U32(fs[@"legacy_bytes"],&legacy)||!legacy||
       !FP8U32(fs[@"activation_offset"],&aq)||!FP8U32(fs[@"activation_bytes"],&aqBytes)||!aqBytes||
       !FP8U32(fs[@"scale_offset"],&sa)||!FP8U32(fs[@"scale_bytes"],&saBytes)||!saBytes||
       !FP8U32(q[@"scratch_bytes"],&total)||aq%256||sa%256||aq<legacy||sa<aq+aqBytes||sa+saBytes>total)
        return fail(@"FP8 scratch reservations overlap or exceed the allocation");
    NSArray* files=q[@"files"];
    if(![files isKindOfClass:NSArray.class]||files.count!=3)return fail(@"FP8 file bindings incomplete");
    NSMutableDictionary* paths=[NSMutableDictionary dictionary];
    if(![arenas isKindOfClass:NSDictionary.class])return fail(@"Invalid arena capacity metadata");
    NSMutableDictionary* capacities=[NSMutableDictionary dictionary];
    for(NSString* name in @[@"arena6",@"arena7",@"arena8"])if(arenas[name])capacities[name]=arenas[name];
    capacities[@"params"]=@176; // The supported immutable stage argument-template ABI.
    capacities[@"scratch"]=@(legacy);
    for(id f in files){if(![f isKindOfClass:NSDictionary.class]||![f[@"name"] isKindOfClass:NSString.class]||
        ![f[@"path"] isKindOfClass:NSString.class]||paths[f[@"name"]])return fail(@"Invalid duplicate FP8 file binding");
        NSDictionary* row=ledger[f[@"path"]];std::uint64_t size=0;
        if(![row isKindOfClass:NSDictionary.class]||!FP8U32(row[@"bytes"],&size)||!size||!FP8Hex(row[@"sha256"]))return fail(@"Unpinned FP8 file binding");
        paths[f[@"name"]]=f[@"path"];capacities[f[@"name"]]=@(size);}
    if(![paths[@"pweights"] isEqual:[NSString stringWithFormat:@"pipeline/%@-weights.bin",layer]]||
       ![paths[@"pweights_fp8"] isEqual:[NSString stringWithFormat:@"pipeline/fp8/%@-weights.e4m3fn.bin",layer]]||
       ![paths[@"pweights_fp8_scales"] isEqual:[NSString stringWithFormat:@"pipeline/fp8/%@-scales.f32le.bin",layer]])
        return fail(@"FP8 payload paths disagree with the selected layer");
    NSMutableDictionary* fullCapacities=[capacities mutableCopy];fullCapacities[@"scratch"]=@(total);
    NSArray* dispatches=q[@"dispatches"];NSDictionary* coverage=q[@"fp8_gemm_coverage"];
    if(![dispatches isKindOfClass:NSArray.class]||![coverage isKindOfClass:NSDictionary.class]||
       ![coverage[@"tensor_ids"] isKindOfClass:NSArray.class]||![coverage[@"fp16_fallbacks"] isKindOfClass:NSArray.class])return fail(@"Missing FP8 coverage metadata");
    NSMutableArray* used=[NSMutableArray array];NSMutableArray* kept=[NSMutableArray array];
    NSUInteger quantizers=0,native=0,half=0,mlps=0;
    for(NSUInteger i=0;i<dispatches.count;++i){
        id raw=dispatches[i];if(![raw isKindOfClass:NSDictionary.class])return fail(@"Invalid FP8 dispatch");
        NSDictionary* d=raw;NSString* fn=d[@"function"];
        if(![fn isKindOfClass:NSString.class])return fail(@"Unnamed FP8 dispatch");
        const BOOL fp8BufferTails=[fn isEqual:@"pwin_quantize_rows_e4m3"]||[fn isEqual:@"pwin_gemm_fp8_e4m3"];
        if(!FP8DispatchStructure(d,fp8BufferTails?fullCapacities:capacities))
            return fail(@"Invalid dispatch binding/byte indices, base bounds, or structural ABI");
        if([fn isEqual:@"pwin_quantize_rows_e4m3"]){
            if(i+1>=dispatches.count||![dispatches[i+1] isKindOfClass:NSDictionary.class]||![dispatches[i+1][@"function"] isEqual:@"pwin_gemm_fp8_e4m3"])return fail(@"Unpaired FP8 row quantizer");
            ++quantizers;continue;
        }
        if([fn isEqual:@"pwin_gemm_fp8_e4m3"]){
            if(!i||![dispatches[i-1] isKindOfClass:NSDictionary.class]||![dispatches[i-1][@"function"] isEqual:@"pwin_quantize_rows_e4m3"])return fail(@"Missing immediate FP8 row quantizer");
            NSDictionary* quant=dispatches[i-1];NSArray* p=FP8Words(d,4,16);NSArray* z=FP8Words(quant,3,8);
            if(!p||!z||!FP8BindingSet(d,@[@0,@1,@2,@3,@5,@6,@7,@8])||!FP8BindingSet(quant,@[@0,@1,@2]))return fail(@"FP8 16/8-u32 ABI mismatch");
            auto n=[&](NSUInteger j){return [p[j] unsignedLongLongValue];};
            const std::uint64_t t=n(0),k=n(1),cols=n(2),stride=((k+127)/128)*128,kpad=((k+31)/32)*32;
            const std::uint64_t tpad=((t+63)/64)*64,npad=((cols+31)/32)*32,store=n(12)?n(12):cols;
            if(!t||!k||k>640||k%16||!cols||cols>768||cols%16||n(3)!=stride||n(11)!=stride||n(15)!=kpad||
               n(4)<store||store>cols||n(6)>3||n(7)>1||!n(9)||n(14)>1||
               ![d[@"constants"] isEqual:@{@"0":p[1],@"1":p[2]}]||
               ![d[@"block"] isEqual:@[@128,@1,@1]]||![quant[@"block"] isEqual:@[@128,@1,@1]]||
               ![d[@"grid"] isEqual:@[@((t+63)/64),@((store+31)/32),@1]]||
               ![quant[@"grid"] isEqual:@[@(tpad/4),@1,@1]]||
               ![z isEqual:@[p[0],p[1],z[2],@(stride),@(kpad),@(tpad),@0,@0]]||[z[2] unsignedLongLongValue]<k)
                return fail(@"FP8 matrix dimensions/grid/quantizer parameters mismatch");
            NSDictionary* a=FP8Binding(d,0);NSDictionary* w=FP8Binding(d,1);NSDictionary* s=FP8Binding(d,7);NSDictionary* sw=FP8Binding(d,8);
            if(![a[@"name"] isEqual:@"scratch"]||![a[@"offset"] isEqual:@(aq)]||
               ![s[@"name"] isEqual:@"scratch"]||![s[@"offset"] isEqual:@(sa)]||
               !FP8SameBinding(a,FP8Binding(quant,1))||!FP8SameBinding(s,FP8Binding(quant,2))||
               tpad*stride>aqBytes||tpad*4>saBytes||
               !FP8MatrixBinding(quant,0,t,[z[2] unsignedLongLongValue],k,2,capacities)||
               !FP8MatrixBinding(d,3,t,n(4),store,2,capacities)||
               ![w[@"name"] isEqual:@"pweights_fp8"]||!FP8Span(w,npad*stride,capacities,128)||
               ![sw[@"name"] isEqual:@"pweights_fp8_scales"]||!FP8Span(sw,npad*4,capacities,128))return fail(@"FP8 pair buffer/span/alignment mismatch");
            if(!FP8EpilogueSpans(d,t,store,store,n(5),n(7),n(6),n(13),capacities))
                return fail(@"FP8 C/Cin/bias/residual spans or strides exceed their buffers");
            if(native>=[coverage[@"tensor_ids"] count])return fail(@"FP8 tensor count mismatch");
            NSString* identity=coverage[@"tensor_ids"][native];NSDictionary* tensor=tensors[identity];
            NSDictionary* payload=tensor[@"payload"];NSDictionary* scales=tensor[@"scales"];
            if(![payload isKindOfClass:NSDictionary.class]||![scales isKindOfClass:NSDictionary.class]||
               ![tensor[@"layer"] isEqual:layer]||![tensor[@"precision"] isEqual:@"e4m3fn"]||![tensor[@"shape"] isEqual:@[p[1],p[2]]]||
               ![payload[@"file"] isEqual:paths[@"pweights_fp8"]]||![payload[@"offset"] isEqual:w[@"offset"]]||
               ![payload[@"order"] isEqual:@"N,K"]||![payload[@"padded_k"] isEqual:@(kpad)]||
               ![payload[@"shape"] isEqual:@[@(npad),@(kpad)]]||
               ![payload[@"padded_n"] isEqual:@(npad)]||![payload[@"row_stride_bytes"] isEqual:@(stride)]||
               ![payload[@"bytes"] isEqual:@(npad*stride)]||![scales[@"file"] isEqual:paths[@"pweights_fp8_scales"]]||
               ![scales[@"offset"] isEqual:sw[@"offset"]]||![scales[@"count"] isEqual:@(npad)]||
               ![scales[@"bytes"] isEqual:@(npad*4)]||![scales[@"dtype"] isEqual:@"float32-le"]||
               ![scales[@"axis"] isEqual:@"output_column"]||![scales[@"direction"] isEqual:@"decoded_value * dequant_scale"])
                return fail(@"FP8 bindings disagree with the pinned pack tensor metadata");
            [used addObject:identity];++native;continue;
        }
        if([fn isEqual:@"pwin_gemm_mpp"]){
            NSArray* p=FP8Words(d,4,16);NSString* identity=[layer isEqual:@"enc0"]?@"enc0.W1":([layer isEqual:@"dec0"]?@"dec0.W7":nil);
            NSDictionary* tensor=identity?tensors[identity]:nil;NSDictionary* w=FP8Binding(d,1);
            if(!p||!identity||![tensor[@"source"] isKindOfClass:NSDictionary.class]||
               ![tensor[@"precision"] isEqual:@"float16-le"]||![tensor[@"shape"] isEqual:@[p[1],p[2]]]||
               ![w[@"name"] isEqual:@"pweights"]||![w[@"offset"] isEqual:tensor[@"source"][@"offset"]]||
               !FP8Span(w,[p[1] unsignedLongLongValue]*[p[2] unsignedLongLongValue]*2,capacities)||
               ([layer isEqual:@"enc0"]&&(![p[1] isEqual:@16]||![p[2] isEqual:@32]))||
               ([layer isEqual:@"dec0"]&&(![p[1] isEqual:@32]||![p[2] isEqual:@48]||![p[12] isEqual:@40]))||
               ![d[@"block"] isEqual:@[@128,@1,@1]]||![p[15] isEqual:@0]||
               ![d[@"constants"] isEqual:@{@"0":p[1],@"1":p[2]}])return fail(@"Unexpected sensitive FP16 GEMM fallback");
            auto n=[&](NSUInteger j){return [p[j] unsignedLongLongValue];};
            const std::uint64_t t=n(0),k=n(1),cols=n(2),ws=n(11)?n(11):cols,stored=n(12)?n(12):cols;
            const std::uint64_t mode=n(8)&15,rounds=n(8)>>4;
            if(!FP8BindingSet(d,@[@0,@1,@2,@3,@5,@6])||!t||!n(9)||n(3)<k||ws<cols||!stored||stored>cols||
               n(4)<stored||n(6)>3||n(7)>1||n(14)>1||mode>6||(rounds&&((mode!=5&&mode!=6)||rounds>=k/16))||
               ![d[@"grid"] isEqual:@[@((t+63)/64),@((stored+31)/32),@1]]||
               ![tensor[@"source"][@"path"] isEqual:paths[@"pweights"]]||
               ![tensor[@"source"][@"row_stride_elements"] isEqual:@(ws)]||
               !FP8MatrixBinding(d,0,t,n(3),k,2,capacities)||!FP8MatrixBinding(d,1,k,ws,cols,2,capacities)||
               !FP8MatrixBinding(d,3,t,n(4),stored,2,capacities)||
               !FP8EpilogueSpans(d,t,cols,stored,n(5),n(7),n(6),n(13),capacities))
                return fail(@"Sensitive FP16 GEMM full ABI, grid, source-weight or operand spans mismatch");
            [kept addObject:identity];++half;continue;
        }
        if([fn hasPrefix:@"pwin_gemm"])return fail(@"Unknown GEMM route in FP8 candidate");
        if([fn isEqual:@"pwin_mlp2_tokens"]){
            NSArray* p=FP8Words(d,7,8);NSDictionary* c=d[@"constants"];
            if(!p||![c isKindOfClass:NSDictionary.class]||!FP8BindingSet(d,@[@0,@1,@2,@3,@4,@5,@6,@8])||
               !([c[@"0"] isEqual:@32]||[c[@"0"] isEqual:@64])||![c[@"1"] isEqual:c[@"0"]]||
               [c[@"14"] unsignedIntegerValue]!=4*[c[@"0"] unsignedIntegerValue]||![p[6] isEqual:@0]||![p[7] isEqual:@0]||
               ![d[@"block"] isEqual:@[@128,@1,@1]]||![d[@"grid"] isEqual:@[@(([p[0] unsignedLongLongValue]+63)/64),@1,@1]])return fail(@"Half MLP fallback ABI mismatch");
            NSSet* required=[NSSet setWithArray:@[@"0",@"1",@"2",@"3",@"7",@"14"]];
            NSSet* allowed=[required setByAddingObject:@"15"];NSSet* keys=[NSSet setWithArray:c.allKeys];
            const std::uint64_t t=[p[0] unsignedLongLongValue],k=[c[@"0"] unsignedLongLongValue],cols=[c[@"1"] unsignedLongLongValue];
            const std::uint64_t hidden=[c[@"14"] unsignedLongLongValue],xs=[c[@"2"] unsignedLongLongValue],cs=[c[@"3"] unsignedLongLongValue],ys=[c[@"7"] unsignedLongLongValue];
            const std::uint64_t wcs=[c[@"15"] unsignedLongLongValue]?:cols,split=[p[1] unsignedLongLongValue],scale=[p[4] unsignedLongLongValue];
            if(![required isSubsetOfSet:keys]||![keys isSubsetOfSet:allowed]||!t||![p[2] unsignedLongLongValue]||
               !(split==0||split==2||split==5||split==6)||scale>1||[p[5] unsignedLongLongValue]>=k/16||
               xs<k||cs<cols||ys<cols||wcs<cols||
               !FP8MatrixBinding(d,0,t,xs,k,2,capacities)||!FP8MatrixBinding(d,1,k,hidden,hidden,2,capacities)||
               !FP8MatrixBinding(d,2,16,hidden,hidden,4,capacities)||!FP8MatrixBinding(d,3,hidden,wcs,cols,2,capacities)||
               !FP8MatrixBinding(d,4,t,cs,cols,2,capacities)||!FP8MatrixBinding(d,5,1,0,cols,2,capacities)||
               !FP8MatrixBinding(d,6,t,ys,cols,2,capacities)||(scale&&!FP8MatrixBinding(d,8,1,0,k,2,capacities)))
                return fail(@"Half MLP strides, function constants, input/weight/Ce/bias/output spans mismatch");
            ++mlps;
        }
    }
    NSDictionary* expected=@{@"enc0":@[@2,@1,@1,@12],@"enc1":@[@3,@0,@1,@10],@"enc2":@[@3,@0,@1,@10],
        @"enc3":@[@5,@0,@0,@14],@"enc4":@[@5,@0,@0,@14],@"dec5":@[@4,@0,@0,@11],@"dec4":@[@5,@0,@0,@15],
        @"dec3":@[@5,@0,@0,@15],@"dec2":@[@3,@0,@1,@11],@"dec1":@[@3,@0,@1,@11],@"dec0":@[@2,@1,@1,@11]};
    NSArray* e=expected[layer];
    if(!e||native!=[e[0] unsignedIntegerValue]||quantizers!=native||half!=[e[1] unsignedIntegerValue]||
       mlps!=[e[2] unsignedIntegerValue]||dispatches.count!=[e[3] unsignedIntegerValue]||
       ![coverage[@"dispatches"] isEqual:@(native)]||![used isEqual:coverage[@"tensor_ids"]]||![kept isEqual:coverage[@"fp16_fallbacks"]]||
       [NSSet setWithArray:used].count!=used.count)return fail(@"Mixed FP8/quantizer/FP16/MLP coverage mismatch");
    return @{@"validated":@YES,@"validation_scope":@"all-dispatch structural base bounds and transformed FP8/FP16/MLP operand spans; original operator semantics remain offline-verified",
        @"general_shader_memory_safety_proven":@NO,@"fp8_gemms":@(native),@"row_quantizers":@(quantizers),@"sensitive_fp16_gemms":@(half),
        @"half_mlps":@(mlps),@"dispatches":@(dispatches.count)};
}
} // namespace netal_probe
