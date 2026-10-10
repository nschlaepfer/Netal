#pragma once

#import <Metal/Metal.h>

extern "C" void FgConnectedSetMacLivePrefixInput(id<MTLBuffer> input);
extern "C" id<MTLBuffer> FgConnectedMacLivePrefixInput(void);
extern "C" void FgConnectedSetMacAllocation29(id<MTLTexture> texture);
extern "C" id<MTLTexture> FgConnectedMacAllocation29(void);
extern "C" void FgConnectedSetMacEx2Tables(id<MTLBuffer> unit_table,
                                             id<MTLBuffer> fraction_table);
extern "C" id<MTLBuffer> FgConnectedMacEx2UnitTable(void);
extern "C" id<MTLBuffer> FgConnectedMacEx2FractionTable(void);
