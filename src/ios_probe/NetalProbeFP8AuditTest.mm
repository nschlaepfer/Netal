#include "NetalProbeFP8Audit.h"
#include <cassert>

// Metadata-only fixtures: no model weights, tensors, Metal device or inference result.
static NSMutableDictionary* Clone(id value) {
    NSData* data=[NSJSONSerialization dataWithJSONObject:value options:0 error:nil];
    return [NSJSONSerialization JSONObjectWithData:data options:NSJSONReadingMutableContainers error:nil];
}
static NSDictionary* Bind(NSUInteger index,NSString* name,NSUInteger offset) {
    return @{@"index":@(index),@"name":name,@"offset":@(offset)};
}
int main(){@autoreleasepool {
    assert(netal_probe::DispatchRingFits(153,8));
    assert(!netal_probe::DispatchRingFits(154,8));
    assert(!netal_probe::DispatchRingFits(161,1));
    assert(!netal_probe::DispatchRingFits(137,0));
    assert(!netal_probe::DispatchRingFits(137,9));
    NSString* root=[@"a" stringByPaddingToLength:64 withString:@"a" startingAtIndex:0];
    NSArray* ids=@[@"enc1.W1",@"enc1.W2",@"enc1.W5"];
    NSArray* files=@[@{@"name":@"pweights",@"path":@"pipeline/enc1-weights.bin"},
        @{@"name":@"pweights_fp8",@"path":@"pipeline/fp8/enc1-weights.e4m3fn.bin"},
        @{@"name":@"pweights_fp8_scales",@"path":@"pipeline/fp8/enc1-scales.f32le.bin"}];
    NSDictionary* ledger=@{@"pipeline/enc1-weights.bin":@{@"bytes":@65536,@"sha256":root},
        @"pipeline/fp8/enc1-weights.e4m3fn.bin":@{@"bytes":@12288,@"sha256":root},
        @"pipeline/fp8/enc1-scales.f32le.bin":@{@"bytes":@384,@"sha256":root}};
    NSMutableDictionary* tensors=[NSMutableDictionary dictionary];NSMutableArray* dispatches=[NSMutableArray array];
    NSDictionary* norm=@{@"function":@"pwin_norm64",@"grid":@[@1,@1,@1],@"block":@[@256,@1,@1],
        @"buffers":@[Bind(0,@"scratch",0),Bind(1,@"scratch",0),Bind(2,@"pweights",0)],
        @"bytes":@[@{@"index":@3,@"u32":@[@64,@32,@32,@0]}]};
    for(NSUInteger i=0;i<3;++i){
        [dispatches addObject:norm];
        [dispatches addObject:@{@"function":@"pwin_quantize_rows_e4m3",@"grid":@[@16,@1,@1],@"block":@[@128,@1,@1],
            @"buffers":@[Bind(0,@"scratch",0),Bind(1,@"scratch",4096),Bind(2,@"scratch",12288)],
            @"bytes":@[@{@"index":@3,@"u32":@[@64,@32,@32,@128,@32,@64,@0,@0]}]}];
        [dispatches addObject:@{@"function":@"pwin_gemm_fp8_e4m3",@"grid":@[@1,@1,@1],@"block":@[@128,@1,@1],
            @"constants":@{@"0":@32,@"1":@32},
            @"buffers":@[Bind(0,@"scratch",4096),Bind(1,@"pweights_fp8",i*4096),Bind(2,@"pweights",0),
                Bind(3,@"scratch",0),Bind(5,@"pweights",0),Bind(6,@"pweights",0),Bind(7,@"scratch",12288),Bind(8,@"pweights_fp8_scales",i*128)],
            @"bytes":@[@{@"index":@4,@"u32":@[@64,@32,@32,@128,@32,@32,@0,@0,@0,@8,@0,@128,@0,@0,@0,@32]}]}];
        tensors[ids[i]]=@{@"id":ids[i],@"layer":@"enc1",@"precision":@"e4m3fn",@"shape":@[@32,@32],
            @"payload":@{@"file":@"pipeline/fp8/enc1-weights.e4m3fn.bin",@"offset":@(i*4096),@"order":@"N,K",
                @"padded_k":@32,@"padded_n":@32,@"row_stride_bytes":@128,@"bytes":@4096,@"shape":@[@32,@32]},
            @"scales":@{@"file":@"pipeline/fp8/enc1-scales.f32le.bin",@"offset":@(i*128),@"count":@32,@"bytes":@128,
                @"dtype":@"float32-le",@"axis":@"output_column",@"direction":@"decoded_value * dequant_scale"}};
    }
    [dispatches addObject:@{@"function":@"pwin_mlp2_tokens",@"grid":@[@1,@1,@1],@"block":@[@128,@1,@1],
        @"constants":@{@"0":@32,@"1":@32,@"2":@32,@"3":@32,@"7":@32,@"14":@128},
        @"buffers":@[Bind(0,@"scratch",0),Bind(1,@"pweights",0),Bind(2,@"pweights",0),Bind(3,@"pweights",0),
            Bind(4,@"scratch",0),Bind(5,@"pweights",0),Bind(6,@"scratch",0),Bind(8,@"pweights",0)],
        @"bytes":@[@{@"index":@7,@"u32":@[@64,@2,@8,@0,@1,@0,@0,@0]}]}];
    NSDictionary* fixture=@{@"library":@"pwin_mpp_all",@"gemm_backend":@"fp8-e4m3",@"mlp_backend":@"mpp-half",
        @"candidate_approximate":@YES,@"reference_exact":@NO,@"requires_ordered_dispatches":@YES,@"fp8_pack_root_sha256":root,
        @"scratch_bytes":@12544,@"fp8_scratch":@{@"legacy_bytes":@4096,@"activation_offset":@4096,@"activation_bytes":@8192,@"scale_offset":@12288,@"scale_bytes":@256},
        @"files":files,@"dispatches":dispatches,@"fp8_gemm_coverage":@{@"dispatches":@3,@"tensor_ids":ids,@"fp16_fallbacks":@[]}};
    auto valid=[&](NSDictionary* q,NSDictionary* l,NSDictionary* t){return [netal_probe::AuditFP8Descriptor(q,@"enc1",l,t,@{},root)[@"validated"] boolValue];};
    assert(valid(fixture,ledger,tensors));
    NSMutableDictionary* q=Clone(fixture);q[@"requires_ordered_dispatches"]=@NO;assert(!valid(q,ledger,tensors));
    q=Clone(fixture);q[@"dispatches"][1][@"function"]=@"pwin_norm64";assert(!valid(q,ledger,tensors));
    q=Clone(fixture);q[@"dispatches"][2][@"bytes"][0][@"u32"][15]=@64;assert(!valid(q,ledger,tensors));
    q=Clone(fixture);q[@"dispatches"][1][@"bytes"][0][@"u32"][3]=@256;assert(!valid(q,ledger,tensors));
    q=Clone(fixture);q[@"dispatches"][2][@"buffers"][1][@"offset"]=@64;assert(!valid(q,ledger,tensors));
    q=Clone(fixture);q[@"dispatches"][2][@"buffers"][0][@"offset"]=@4352;assert(!valid(q,ledger,tensors));
    q=Clone(fixture);q[@"fp8_scratch"][@"scale_offset"]=@8192;assert(!valid(q,ledger,tensors));
    q=Clone(fixture);q[@"fp8_gemm_coverage"][@"dispatches"]=@2;assert(!valid(q,ledger,tensors));
    NSMutableDictionary* l=Clone(ledger);l[@"pipeline/fp8/enc1-weights.e4m3fn.bin"][@"bytes"]=@4096;assert(!valid(fixture,l,tensors));
    NSMutableDictionary* t=Clone(tensors);t[@"enc1.W1"][@"payload"][@"offset"]=@128;assert(!valid(fixture,ledger,t));
    // Generic argument-table corruption, including paths not otherwise transformed by FP8.
    for(NSNumber* index in @[@17,@999,@(-1),@4294967296ULL,@YES]){
        q=Clone(fixture);q[@"dispatches"][0][@"buffers"][0][@"index"]=index;assert(!valid(q,ledger,tensors));
    }
    for(NSNumber* offset in @[@(-2),@4294967296ULL,@4096]){
        q=Clone(fixture);q[@"dispatches"][0][@"buffers"][0][@"offset"]=offset;assert(!valid(q,ledger,tensors));
    }
    q=Clone(fixture);q[@"dispatches"][0][@"buffers"][0][@"name"]=@"unknown";assert(!valid(q,ledger,tensors));
    q=Clone(fixture);q[@"dispatches"][0][@"buffers"][1][@"index"]=@0;assert(!valid(q,ledger,tensors));
    q=Clone(fixture);q[@"dispatches"][0][@"bytes"][0][@"index"]=@0;assert(!valid(q,ledger,tensors));
    q=Clone(fixture);[q[@"dispatches"][0][@"bytes"] addObject:q[@"dispatches"][0][@"bytes"][0]];assert(!valid(q,ledger,tensors));
    q=Clone(fixture);q[@"dispatches"][0][@"bytes"][0][@"u32"][0]=@0.5;assert(!valid(q,ledger,tensors));
    // Full C tile, mode1 bias/Cin, and residual spans. First show each valid mode is admitted.
    q=Clone(fixture);q[@"dispatches"][2][@"buffers"][2][@"offset"]=@65532;assert(!valid(q,ledger,tensors));
    q=Clone(fixture);q[@"dispatches"][2][@"bytes"][0][@"u32"][5]=@1;assert(!valid(q,ledger,tensors));
    q=Clone(fixture);q[@"dispatches"][2][@"bytes"][0][@"u32"][7]=@1;assert(valid(q,ledger,tensors));
    q[@"dispatches"][2][@"buffers"][5][@"offset"]=@65534;assert(!valid(q,ledger,tensors));
    q=Clone(fixture);q[@"dispatches"][2][@"bytes"][0][@"u32"][7]=@1;
    q[@"dispatches"][2][@"buffers"][4][@"offset"]=@65534;assert(!valid(q,ledger,tensors));
    q=Clone(fixture);q[@"dispatches"][2][@"bytes"][0][@"u32"][6]=@3;q[@"dispatches"][2][@"bytes"][0][@"u32"][13]=@32;
    assert(valid(q,ledger,tensors));q[@"dispatches"][2][@"bytes"][0][@"u32"][13]=@31;assert(!valid(q,ledger,tensors));
    q[@"dispatches"][2][@"bytes"][0][@"u32"][13]=@4294967295ULL;assert(!valid(q,ledger,tensors));
    // Required MLP slots, all function-constant strides, Ce's full16 rows, and scaled-X vector.
    q=Clone(fixture);q[@"dispatches"][9][@"buffers"]=@[];assert(!valid(q,ledger,tensors));
    q=Clone(fixture);q[@"dispatches"][9][@"buffers"]=@[Bind(999,@"scratch",0)];assert(!valid(q,ledger,tensors));
    for(NSString* constant in @[@"2",@"3",@"7",@"15"]){
        q=Clone(fixture);q[@"dispatches"][9][@"constants"][constant]=@31;assert(!valid(q,ledger,tensors));
    }
    for(NSUInteger index : {1u,2u,3u,5u,7u}){
        q=Clone(fixture);q[@"dispatches"][9][@"buffers"][index][@"offset"]=@65532;assert(!valid(q,ledger,tensors));
    }
    q=Clone(fixture);q[@"dispatches"][9][@"bytes"][0][@"u32"][0]=@4294967295ULL;
    q[@"dispatches"][9][@"grid"][0]=@67108864;q[@"dispatches"][9][@"constants"][@"2"]=@4294967295ULL;assert(!valid(q,ledger,tensors));
    q=Clone(fixture);q[@"dispatches"][9][@"bytes"][0][@"u32"][2]=@0;assert(!valid(q,ledger,tensors));
    q=Clone(fixture);q[@"dispatches"][9][@"bytes"][0][@"u32"][0]=@1;assert(valid(q,ledger,tensors));
    q[@"dispatches"][9][@"buffers"][2][@"offset"]=@(65536-512);assert(!valid(q,ledger,tensors)); // Ce still reads16 rows with one token.
    // The direct half fallback consumes all N=48 initialization columns although stores stop at40.
    NSDictionary* caps=@{@"pweights":@65536};
    NSMutableDictionary* direct=Clone(@{@"buffers":@[Bind(2,@"pweights",0),Bind(5,@"pweights",0),Bind(6,@"pweights",0)]});
    assert(netal_probe::FP8EpilogueSpans(direct,32,48,40,48,1,0,0,caps));
    direct[@"buffers"][2][@"offset"]=@(65536-80);assert(!netal_probe::FP8EpilogueSpans(direct,32,48,40,48,1,0,0,caps));
    std::uint64_t bytes=0;assert(!netal_probe::FP8MatrixBytes(UINT32_MAX,UINT32_MAX,32,4,&bytes));
}}
