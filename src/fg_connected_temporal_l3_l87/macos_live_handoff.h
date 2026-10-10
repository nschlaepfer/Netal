#pragma once

#import <Metal/Metal.h>

extern "C" void FgConnectedSetMacLivePrefixInput(id<MTLBuffer> input);
extern "C" id<MTLBuffer> FgConnectedMacLivePrefixInput(void);
extern "C" void FgConnectedSetMacAllocation29(id<MTLTexture> texture);
extern "C" id<MTLTexture> FgConnectedMacAllocation29(void);
