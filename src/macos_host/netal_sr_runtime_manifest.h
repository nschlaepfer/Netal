#pragma once
#import <Foundation/Foundation.h>
#import <CommonCrypto/CommonDigest.h>
#include <cstring>

static NSString* NetalAssetSHA(NSData* data) {
    CC_SHA256_CTX context; CC_SHA256_Init(&context);
    const auto* bytes=static_cast<const unsigned char*>(data.bytes);
    NSUInteger left=data.length;
    while (left) { CC_LONG part=static_cast<CC_LONG>(MIN(left,static_cast<NSUInteger>(1u<<30))); CC_SHA256_Update(&context,bytes,part);bytes+=part;left-=part; }
    unsigned char digest[32];CC_SHA256_Final(digest,&context);
    NSMutableString* result=[NSMutableString string];for(unsigned char value:digest)[result appendFormat:@"%02x",value];return result;
}
static bool NetalSafeAssetPath(NSString* path) {
    if (![path isKindOfClass:NSString.class] || !path.length || [path hasPrefix:@"/"] || [path.lowercaseString containsString:@"expected"] || [path.lowercaseString containsString:@"capture"]) return false;
    NSCharacterSet* allowed=[NSCharacterSet characterSetWithCharactersInString:@"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_./-"];
    if ([path rangeOfCharacterFromSet:allowed.invertedSet].location!=NSNotFound) return false;
    for(NSString* part in [path componentsSeparatedByString:@"/"]) if (!part.length || [part isEqual:@"."] || [part isEqual:@".."]) return false;
    return true;
}
static bool NetalValidateSRInventoryV2(NSURL* root, NSDictionary* manifest, NSString** reason) {
    root=root.URLByResolvingSymlinksInPath.URLByStandardizingPath;
    auto fail=[&](NSString* message){if(reason)*reason=message;return false;};
    if (![manifest isKindOfClass:NSDictionary.class] || ![manifest[@"schema"] isEqual:@"netal.sr.runtime.v2"] || ![manifest[@"model"] isEqual:@"DLSS_SR_Feature1_Quality_K"]) return fail(@"unsupported SR manifest contract");
    NSArray* rows=manifest[@"files"];
    if (![rows isKindOfClass:NSArray.class] || !rows.count) return fail(@"missing asset inventory");
    NSMutableDictionary<NSString*,NSDictionary*>* ledger=[NSMutableDictionary dictionary];
    NSCharacterSet* hex=[NSCharacterSet characterSetWithCharactersInString:@"0123456789abcdef"];
    for(id item in rows) {
        if (![item isKindOfClass:NSDictionary.class]) return fail(@"invalid asset row");
        NSDictionary* row=item;NSString* path=row[@"path"];NSString* hash=row[@"sha256"];NSNumber* size=row[@"bytes"];
        if (!NetalSafeAssetPath(path) || [path isEqual:@"manifest.json"] || ledger[path] || ![hash isKindOfClass:NSString.class] || hash.length!=64 || [hash rangeOfCharacterFromSet:hex.invertedSet].location!=NSNotFound || ![size isKindOfClass:NSNumber.class] || CFGetTypeID((__bridge CFTypeRef)size)==CFBooleanGetTypeID() || ![size.stringValue isEqualToString:[NSString stringWithFormat:@"%llu",size.unsignedLongLongValue]]) return fail(@"invalid or duplicate asset metadata");
        ledger[path]=row;
    }
    NSMutableData* material=[NSMutableData dataWithData:[@"netal.sr.runtime.v2\n" dataUsingEncoding:NSUTF8StringEncoding]];
    const unsigned char zero=0;
    for(NSString* path in [[ledger allKeys] sortedArrayUsingSelector:@selector(compare:)]) {
        NSDictionary* row=ledger[path];
        [material appendData:[path dataUsingEncoding:NSUTF8StringEncoding]];[material appendBytes:&zero length:1];
        [material appendData:[[row[@"bytes"] stringValue] dataUsingEncoding:NSUTF8StringEncoding]];[material appendBytes:&zero length:1];
        [material appendData:[[row[@"sha256"] stringByAppendingString:@"\n"] dataUsingEncoding:NSUTF8StringEncoding]];
    }
    if (![NetalAssetSHA(material) isEqual:manifest[@"asset_root_sha256"]]) return fail(@"asset inventory root mismatch");
    NSFileManager* manager=NSFileManager.defaultManager;
    __block BOOL enumeration_failed=NO;
    NSDirectoryEnumerator* enumerator=[manager enumeratorAtURL:root includingPropertiesForKeys:nil options:0 errorHandler:^BOOL(NSURL*,NSError*) { enumeration_failed=YES; return NO; }];
    if (!enumerator) return fail(@"cannot enumerate runtime");
    NSMutableSet* seen=[NSMutableSet set];
    for(NSURL* url in enumerator) {
        NSDictionary* attributes=[manager attributesOfItemAtPath:url.path error:nil];
        if ([attributes[NSFileType] isEqual:NSFileTypeSymbolicLink]) return fail(@"symlink in runtime");
        if ([attributes[NSFileType] isEqual:NSFileTypeDirectory]) continue;
        if (![attributes[NSFileType] isEqual:NSFileTypeRegular]) return fail(@"nonregular runtime asset");
        NSString* prefix=[root.path stringByAppendingString:@"/"];
        NSString* resolved=url.URLByResolvingSymlinksInPath.URLByStandardizingPath.path;
        if (![resolved hasPrefix:prefix]) return fail(@"runtime path escaped root");
        NSString* relative=[resolved substringFromIndex:prefix.length];if([relative isEqual:@"manifest.json"])continue;
        NSDictionary* row=ledger[relative];NSData* data=[NSData dataWithContentsOfURL:url];
        if (!row || !data || data.length!=[row[@"bytes"] unsignedLongLongValue] || ![NetalAssetSHA(data) isEqual:row[@"sha256"]]) return fail([@"runtime asset mismatch: " stringByAppendingString:relative]);
        NSString* name=relative.lastPathComponent;
        NSArray<NSNumber*>* pointers=nil;NSUInteger expected=0;
        if ([name isEqual:@"hiluma-input-reset-template.bin"]) { expected=248;pointers=@[@168,@176,@184,@192,@200,@208,@216,@224,@240]; }
        if ([name isEqual:@"hiluma-output-reset-template.bin"]) { expected=328;pointers=@[@200,@224,@232,@240,@248,@256,@264,@272,@280,@288,@296,@304,@312]; }
        if ([name isEqual:@"exposure-reset-template.bin"]) { expected=16;pointers=@[@0,@8]; }
        if ([name isEqual:@"enc0-arguments-template.bin"]) { expected=176;pointers=@[@8,@24,@32,@48,@64]; }
        if ([name isEqual:@"arguments-template.bin"] && ([relative hasPrefix:@"pwin/"] || [relative containsString:@"/pwin/"])) {
            NSURL* config_url=[[url URLByDeletingLastPathComponent] URLByAppendingPathComponent:@"replay-config.bin"];
            if ([[manager attributesOfItemAtPath:config_url.path error:nil][NSFileType] isEqual:NSFileTypeSymbolicLink])return fail(@"linked PWin config");
            NSData* config=[NSData dataWithContentsOfURL:config_url];
            if (config.length!=296 || std::memcmp(config.bytes,"NSPWIN1",7)!=0)return fail(@"PWin ABI shape mismatch");
            const auto* raw=static_cast<const unsigned char*>(config.bytes);uint32_t count=0;std::memcpy(&count,raw+164,4);
            if (count>8)return fail(@"PWin relocation count");
            NSMutableArray* slots=[NSMutableArray array];
            for(uint32_t i=0;i<count;i++){uint32_t offset=0;std::memcpy(&offset,raw+168+i*16,4);if(offset>168)return fail(@"PWin pointer offset");[slots addObject:@(offset)];}
            expected=176;pointers=slots;
        }
        if (pointers) {
            if (data.length!=expected) return fail(@"boundary ABI size mismatch");
            const auto* bytes=static_cast<const unsigned char*>(data.bytes);
            for(NSNumber* offset in pointers) for(NSUInteger i=0;i<8;i++) if(bytes[offset.unsignedIntegerValue+i])return fail(@"captured pointer in boundary ABI");
        }
        [seen addObject:relative];
    }
    if (enumeration_failed) return fail(@"cannot enumerate complete runtime");
    if (seen.count!=ledger.count) return fail(@"missing runtime assets");
    return true;
}
static bool NetalValidateSRManifestV2(NSURL* root, NSDictionary* manifest, NSString** reason) {
    if (!NetalValidateSRInventoryV2(root,manifest,reason)) return false;
    auto fail=[&](NSString* message){if(reason)*reason=message;return false;};
    NSMutableDictionary<NSString*,NSDictionary*>* ledger=[NSMutableDictionary dictionary];
    for(NSDictionary* row in manifest[@"files"])ledger[row[@"path"]]=row;
    NSDictionary* models=@{@"model/weights.bin":@[@3545824,@"ddaa59cd82deed53bad9d65b82b454f1439fc4875e4220b3ffafafb6cd0296a1"],@"model/constants.bin":@[@524288,@"305f4c6773d1cb333b577094e28a46beb1867898716d8290121862a2cb06e773"]};
    for(NSString* path in models) if (![ledger[path][@"bytes"] isEqual:models[path][0]] || ![ledger[path][@"sha256"] isEqual:models[path][1]]) return fail(@"unsupported model bytes");
    NSArray* required=@[@"metallib/hiluma_input_poison_a.metallib",@"metallib/hiluma_output_exact_math_safe.metallib",@"metallib/hiluma_input_depthinv_exact.metallib",@"metallib/hiluma_output_depthinv_exact.metallib",@"metallib/netal_copy_exposure.metallib",@"metallib/dlss_sr_live_texture_convert.metallib",@"math/rcp-correction-rom.bin",@"math/rsq-correction-rom.bin",@"math/rcp-f16-to-f16-lut.bin",@"math/ex2-f16x2-lut.bin",@"abi/hiluma-input-reset-template.bin",@"abi/hiluma-output-reset-template.bin",@"abi/exposure-reset-template.bin"];
    for(NSString* name in required)if(!ledger[name])return fail(@"missing required runtime asset");
    for(NSString* name in @[@"metallib/hiluma_input_depthinv.metallib",@"math/sm120-texture-weights.bin",@"math/sm120-mufu/lg2-input-table.bin",@"math/sm120-mufu/lg2-output-table.bin",@"math/sm120-mufu/ex2-input-table.bin",@"math/sm120-mufu/ex2-output-table.bin"])if(!ledger[name])return fail(@"missing scalar/input contract asset");
    for(NSString* stage in @[@"enc0",@"enc1",@"enc2",@"enc3",@"enc4",@"dec5",@"dec4",@"dec3",@"dec2",@"dec1",@"dec0"])if(!ledger[[NSString stringWithFormat:@"metallib/dltss_pwin_%@_layer_exact.metallib",stage]])return fail(@"missing Reference kernel");
    return true;
}
