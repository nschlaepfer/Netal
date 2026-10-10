#pragma once

#import <Metal/Metal.h>

extern "C" void FgSubrectSetMacLivePrefixInput(id<MTLBuffer> input);
extern "C" id<MTLBuffer> FgSubrectMacLivePrefixInput(void);
