#import "netal_sr_runtime_manifest.h"
#include <iostream>
int main(int argc,const char** argv) { @autoreleasepool {
    if(argc<2 || argc>3)return 64;
    NSURL* root=[NSURL fileURLWithPath:@(argv[1])];
    NSData* data=[NSData dataWithContentsOfURL:[root URLByAppendingPathComponent:@"manifest.json"]];
    NSDictionary* manifest=data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
    NSString* reason=nil;
    bool ok=argc==3 && std::string(argv[2])=="--inventory-only" ? NetalValidateSRInventoryV2(root,manifest,&reason) : NetalValidateSRManifestV2(root,manifest,&reason);
    std::cout<<(ok?"result=pass":"result=fail")<<" detail="<<(reason?reason.UTF8String:"validated")<<'\n';return ok?0:1;
} }
