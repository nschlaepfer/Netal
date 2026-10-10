// Read-only runtime discovery for ANE-FEAS-001. Never compiles/evaluates a model.
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <objc/runtime.h>
#include <dlfcn.h>
#include <cstdlib>

static NSArray* Methods(Class cls) {
  unsigned count=0;Method* methods=class_copyMethodList(cls,&count);
  NSMutableArray* result=[NSMutableArray array];
  for(unsigned i=0;i<count;++i)[result addObject:NSStringFromSelector(method_getName(methods[i]))];
  free(methods);return [result sortedArrayUsingSelector:@selector(compare:)];
}
static NSDictionary* MethodTypes(Class cls){
  unsigned count=0;Method* methods=class_copyMethodList(cls,&count);NSMutableDictionary* result=[NSMutableDictionary dictionary];
  for(unsigned i=0;i<count;++i)result[NSStringFromSelector(method_getName(methods[i]))]=[NSString stringWithUTF8String:method_getTypeEncoding(methods[i])];
  free(methods);return result;
}
int main(int argc,const char**argv) {@autoreleasepool {
  if(argc!=2){fprintf(stderr,"usage: placement_probe NEW_JSON\n");return 2;}
  NSString* output=[NSString stringWithUTF8String:argv[1]];
  if([[NSFileManager defaultManager] fileExistsAtPath:output]){fprintf(stderr,"preserve output\n");return 3;}
  NSMutableArray* devices=[NSMutableArray array];
  for(id<MLComputeDeviceProtocol> device in [MLModel availableComputeDevices]) {
    NSMutableDictionary* item=[@{@"class":NSStringFromClass([(id)device class])} mutableCopy];
    if([(id)device isKindOfClass:[MLNeuralEngineComputeDevice class]])item[@"cores"]=@([(MLNeuralEngineComputeDevice*)device totalCoreCount]);
    [devices addObject:item];
  }
  void* framework=dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",RTLD_LAZY|RTLD_LOCAL);
  NSMutableDictionary* classes=[NSMutableDictionary dictionary];
  for(NSString* name in @[@"_ANEInMemoryModel",@"_ANEInMemoryModelDescriptor",@"_ANERequest",@"_ANEIOSurfaceObject",@"_ANEPerformanceStats",@"_ANEPerformanceStatsIOSurface"]) {
    Class cls=NSClassFromString(name);
    classes[name]=@{@"present":@(cls!=Nil),@"instance_methods":cls?Methods(cls):@[],@"class_methods":cls?Methods(object_getClass(cls)):@[],@"instance_method_types":cls?MethodTypes(cls):@{},@"class_method_types":cls?MethodTypes(object_getClass(cls)):@{}};
  }
  NSDictionary* result=@{@"schema":@"neuralstudio.nr.ane-runtime-discovery.v1",@"devices":devices,@"private_framework_loaded":@(framework!=nullptr),
    @"classes":classes,@"model_compiled":@NO,@"model_evaluated":@NO,@"ane_execution_proven":@NO,
    @"scope":@"Static runtime availability and selector discovery only. Device presence and API availability do not prove ANE execution."};
  NSError* error=nil;NSData* json=[NSJSONSerialization dataWithJSONObject:result options:NSJSONWritingPrettyPrinted error:&error];
  if(!json || ![json writeToFile:output options:NSDataWritingWithoutOverwriting error:&error]){fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);return 4;}
  return 0;
}}
