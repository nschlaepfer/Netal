// CPU-only selector/layout admission checks. No device, game or model execution.
#define NETAL_NR_OBSERVE_ONLY 1
#include "netal_nr_experimental_bridge.mm"
#include <stdexcept>
@interface LegacyLayoutFixture : NSObject
@property(nonatomic) NSUInteger outputWidth,outputHeight;
@end
@implementation LegacyLayoutFixture
@end
@interface ModernLayoutFixture : LegacyLayoutFixture
@property(nonatomic) NSUInteger outputOffsetX,outputOffsetY;
@property(nonatomic,getter=isOutputResolutionMotionVectorsEnabled) BOOL outputResolutionMotionVectorsEnabled;
@property(nonatomic,getter=isJitteredMotionVectorsEnabled) BOOL jitteredMotionVectorsEnabled;
@end
@implementation ModernLayoutFixture
@end
int main() { @autoreleasepool {
  int tests=0;
  auto need=[&](bool ok,const char* name) {++tests;if(!ok)throw std::runtime_error(name);};
  try {
    auto legacy=[LegacyLayoutFixture new];legacy.outputWidth=1920;legacy.outputHeight=1080;
    need(OptionalUInt(legacy,"outputOffsetX")==-1,"missing offset explicit");
    need(OptionalBool(legacy,"isJitteredMotionVectorsEnabled")==-1,"missing flag explicit");
    need(WholeFrameProfile(legacy,1920,1080),"legacy zero-origin full extent");
    need(!WholeFrameProfile([NSObject new],1920,1080),"missing extent must reject");
    auto modern=[ModernLayoutFixture new];modern.outputWidth=1920;modern.outputHeight=1080;
    need(WholeFrameProfile(modern,1920,1080),"modern whole frame");
    modern.outputOffsetX=1;need(!WholeFrameProfile(modern,1920,1080),"X subrect refused");
    modern.outputOffsetX=std::numeric_limits<NSUInteger>::max();
    need(!WholeFrameProfile(modern,1920,1080),"oversized offset cannot alias absent selector");
    modern.outputOffsetX=0;modern.outputOffsetY=1;need(!WholeFrameProfile(modern,1920,1080),"Y subrect refused");
    modern.outputOffsetY=0;modern.outputHeight=1200;need(!WholeFrameProfile(modern,1920,1080),"1200 refused");
    modern.outputHeight=1080;modern.outputWidth=960;need(!WholeFrameProfile(modern,1920,1080),"partial width refused");
    need(OptionalBool(modern,"isOutputResolutionMotionVectorsEnabled")==0,"false flag");
    modern.outputResolutionMotionVectorsEnabled=YES;
    need(OptionalBool(modern,"isOutputResolutionMotionVectorsEnabled")==1,"true flag");
    modern.jitteredMotionVectorsEnabled=YES;
    need(OptionalBool(modern,"isJitteredMotionVectorsEnabled")==1,"custom jitter getter");
    std::printf("result=PASS cases=%d GPU_execution=0 model_execution=0 game_execution=0\n",tests);return 0;
  }catch(const std::exception& e){std::fprintf(stderr,"result=FAIL detail=%s\n",e.what());return 1;}
} }
