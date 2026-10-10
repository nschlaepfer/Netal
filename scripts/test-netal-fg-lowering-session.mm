#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "../src/macos_host/dlss_fg_session.h"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc == 3) {
      const bool reference = std::strcmp(argv[2], "reference") == 0;
      if (!reference && std::strcmp(argv[2], "hardware") != 0) return 2;
      unsetenv("NSFG_PREFIX_MODE"); unsetenv("NSFG_BLOCK_MODE");
      setenv("NEURALSTUDIO_FG_EXACT", reference ? "1" : "0", 1);
      NSError* error = nil;
      auto session = [[NeuralStudioDLSSFGSession alloc] initWithDevice:MTLCreateSystemDefaultDevice()
          runtimeBundleURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
      if (!session) { std::cerr << error.localizedDescription.UTF8String << '\n'; return 1; }
      assert(session.exactLowering == reference);
      std::cout << "mode=" << argv[2] << " initialized=true frames_executed=0\n";
      return 0;
    }
    if (argc != 1) return 2;
    // These cases terminate before Metal/model setup. No GPU or model is needed.
    id<MTLDevice> noDevice = nil;
    NSURL* absent = [NSURL fileURLWithPath:@"/nonexistent-netal-lowering-fixture"];
    setenv("NEURALSTUDIO_FG_EXACT", "1", 1);
    for (const char* key : {"NSFG_PREFIX_MODE", "NSFG_BLOCK_MODE"}) {
      unsetenv("NSFG_PREFIX_MODE"); unsetenv("NSFG_BLOCK_MODE");
      setenv(key, "hw", 1);
      NSError* error = nil;
      auto session = [[NeuralStudioDLSSFGSession alloc] initWithDevice:noDevice runtimeBundleURL:absent error:&error];
      assert(session == nil && error.code == 6);
    }
    unsetenv("NSFG_PREFIX_MODE"); unsetenv("NSFG_BLOCK_MODE");
    NSError* error = nil;
    auto reference = [[NeuralStudioDLSSFGSession alloc] initWithDevice:noDevice runtimeBundleURL:absent error:&error];
    assert(reference == nil && error.code == 1); // configuration accepted; bundle absent
    unsetenv("NEURALSTUDIO_FG_EXACT"); error = nil;
    auto accelerated = [[NeuralStudioDLSSFGSession alloc] initWithDevice:noDevice runtimeBundleURL:absent error:&error];
    assert(accelerated == nil && error.code == 1);
    assert(std::strcmp(std::getenv("NSFG_PREFIX_MODE"), "hw") == 0);
    assert(std::strcmp(std::getenv("NSFG_BLOCK_MODE"), "hw") == 0);
    std::cout << "four adapter-constructor mode checks passed; gpu_executed=false\n";
  }
}
