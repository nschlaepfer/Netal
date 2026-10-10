Buffer ExportEncoderNative(Pipelines& p,id<MTLLibrary> lib,Buffer& source,uint32_t width,uint32_t height,uint32_t channels,bool planar,uint8_t poison) {
  const size_t rows=size_t((width+7)/8)*((height+7)/8)*64;
  if(source.bytes!=rows*channels||width%4||height%4||channels%32)throw std::runtime_error("native encoder export extent");
  const uint32_t shape[4]={width,height,channels,planar?1u:0u};const size_t bytes=size_t(width)*height*channels;
  Buffer result=Make(p.device,bytes,poison),counts=Make(p.device,bytes*4,0);const auto before=Sha(Data(source),source.bytes);NSError* error=nil;
  auto state=[p.device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nr_encoder_native_export"] error:&error];if(!state)throw std::runtime_error("export pipeline creation");
  auto command=[p.queue commandBuffer];auto encoder=[command computeCommandEncoder];[encoder setComputePipelineState:state];
  [encoder setBuffer:source.metal offset:Guard atIndex:0];[encoder setBuffer:result.metal offset:Guard atIndex:1];[encoder setBuffer:counts.metal offset:Guard atIndex:2];
  [encoder setBytes:shape length:sizeof(shape) atIndex:3];[encoder dispatchThreads:MTLSizeMake(bytes,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[encoder endEncoding];Complete(command);
  GuardOutput(result,counts);if(before!=Sha(Data(source),source.bytes)||Canary(source))throw std::runtime_error("native export modified source");return result;
}
