#import "netal_ios_sr_capabilities.h"
#include "netal_ios_sr_device_policy.h"

#import <TargetConditionals.h>

namespace {
#if TARGET_OS_IOS && !TARGET_OS_SIMULATOR && !TARGET_OS_MACCATALYST
bool CanAllocateTensor(id<MTLDevice> device, MTLTensorDataType type) API_AVAILABLE(ios(26.0)) {
  MTLTensorDescriptor* descriptor = [[MTLTensorDescriptor alloc] init];
  const NSInteger extent = 32;  // Packed FP8 types require an innermost multiple of 32.
  descriptor.dimensions = [[MTLTensorExtents alloc] initWithRank:1 values:&extent];
  descriptor.dataType = type;
  descriptor.storageMode = MTLStorageModePrivate;
  descriptor.usage = MTLTensorUsageCompute;
  NSError* error = nil;
  id<MTLTensor> tensor = [device newTensorWithDescriptor:descriptor error:&error];
  return tensor != nil && error == nil;
}
#endif

NSString* String(std::string_view value) {
  return [[NSString alloc] initWithBytes:value.data()
                                 length:value.size()
                               encoding:NSUTF8StringEncoding];
}
}  // namespace

@interface NetalIOSSRCapabilities ()
- (instancetype)initWithAdmission:(const netal::ios_sr::Admission&)admission;
@end

@implementation NetalIOSSRCapabilities

- (instancetype)initWithAdmission:(const netal::ios_sr::Admission&)admission {
  if ((self = [super init])) {
    _eligible = admission.eligible;
    _chipName = String(netal::ios_sr::ChipName(admission.chip));
    _reason = String(netal::ios_sr::AdmissionReasonText(admission.reason));
    _supportsNeuralAccelerators = admission.neural_accelerator_candidate;
    _supportsFP8TensorTypes = admission.fp8_tensor_types_available;
  }
  return self;
}

+ (instancetype)capabilitiesForDevice:(id<MTLDevice>)device {
  netal::ios_sr::DeviceFacts facts;
  NSString* name = device.name ?: @"";
  const char* utf8 = name.UTF8String;
  facts.gpu_name = utf8 ? std::string_view(utf8, [name lengthOfBytesUsingEncoding:NSUTF8StringEncoding])
                        : std::string_view();
  facts.metal_device_available = device != nil;
  const NSOperatingSystemVersion os = NSProcessInfo.processInfo.operatingSystemVersion;
  facts.os_major = os.majorVersion >= 0 ? static_cast<unsigned>(os.majorVersion) : 0;
#if TARGET_OS_MACCATALYST
  facts.platform = netal::ios_sr::Platform::Catalyst;
#elif TARGET_OS_IOS && TARGET_OS_SIMULATOR
  facts.platform = netal::ios_sr::Platform::Simulator;
#elif TARGET_OS_IOS
  facts.platform = netal::ios_sr::Platform::PhysicalIOS;
  if (@available(iOS 26.0, *)) {
    facts.supports_metal4 = [device supportsFamily:MTLGPUFamilyMetal4];
    facts.supports_apple10 = [device supportsFamily:MTLGPUFamilyApple10];
#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 270000
    if (@available(iOS 27.0, *)) {
      facts.supports_apple10 = facts.supports_apple10 || [device supportsFamily:MTLGPUFamilyApple11];
    }
#endif
    if (facts.supports_metal4 && facts.supports_apple10 &&
        netal::ios_sr::ClassifyChipName(facts.gpu_name) != netal::ios_sr::Chip::Unknown) {
      facts.fp16_tensor_allocation_succeeded = CanAllocateTensor(device, MTLTensorDataTypeFloat16);
#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 270000
      if (@available(iOS 27.0, *)) {
        if (facts.fp16_tensor_allocation_succeeded) {
          facts.fp8_tensor_allocations_succeeded =
              CanAllocateTensor(device, MTLTensorDataTypeMetalFloat8E4M3) &&
              CanAllocateTensor(device, MTLTensorDataTypeMetalFloat8E5M2);
        }
      }
#endif
    }
  }
#else
  facts.platform = netal::ios_sr::Platform::Other;
#endif
  return [[self alloc] initWithAdmission:netal::ios_sr::EvaluateDevice(facts)];
}

@end
