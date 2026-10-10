#pragma once

#import <Metal/Metal.h>

extern "C" void FgConnectedSetMacLivePrefixInput(id<MTLBuffer> input);
extern "C" id<MTLBuffer> FgConnectedMacLivePrefixInput(void);
