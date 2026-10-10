#!/usr/bin/env python3
"""Generate the retained FG live-tail host from tracked source; no model inputs or execution."""
import pathlib, re, sys
source = pathlib.Path(sys.argv[1]).read_text()
cache = pathlib.Path(sys.argv[3]).read_text()
replacements = {
    '#define main FgAcceptedConnectedTailUnusedMain':
        '#define NSFG_TAIL_AUDIT_HOOK() NSFGTailAuditEnabled()\n'
        '#define Guarded NSFGAcceptedGuarded\n'
        '#define Pipeline NSFGAcceptedPipeline\n'
        '#define TailTexture NSFGAcceptedTailTexture\n'
        '#define Wait NSFGAcceptedWait\n'
        '#define main FgAcceptedConnectedTailUnusedMain',
    '#undef main\n\n#include "../fg_conv/conv_1x1_contract.h"':
        '#undef main\n#undef Wait\n#undef TailTexture\n#undef Pipeline\n#undef Guarded\n\n'
        '#include "../fg_runtime/tail_output_bridge.h"\n'
        '#include "../fg_runtime/span_recorder.h"\n'
        '#include "../fg_production_contract_l3_l87/macos_live_handoff.h"\n'
        '#define Write(...) ((void)0)\n'
        '#include "../fg_conv/conv_1x1_contract.h"',
    'int main(int argc, const char* argv[]) {':
        'int FgRuntimeFullTailMain(int argc, const char* argv[]) {',
    '      auto conv3 = Pipeline(device, libs / "conv_3x3.metallib",':
        '      auto conv3 = Pipeline(device, libs / NSFGPrefixLibName("conv_3x3.metallib"),',
    '      auto conv1 = Pipeline(device, libs / "conv_1x1.metallib",':
        '      auto conv1 = Pipeline(device, libs / NSFGPrefixLibName("conv_1x1.metallib"),',
    # Launch 84's inpainting mask (d3d-resource-000003, R8 640x360) is a DLSS-G-owned resource with no
    # producer among the selected launches; in the core feature mode it keeps its created contents,
    # zero. The reference tail feeds it the two-rectangle "islands" fixture, which put blurred
    # rectangles into every generated frame in the game. Live frames get the zero mask.
    '      const auto tail_mask_before = TailMask();':
        '      static const std::vector<std::uint8_t> tail_mask_before = NSFGTailUsesLiveAuxiliaries()\n'
        '          ? std::vector<std::uint8_t>(kTailMaskBytes, 0) : TailMask();',
    '      const auto prefix_input_before = PrefixInput();\n'
    '      id<MTLBuffer> prefix_input =\n'
    '          Guarded(device, &prefix_input_before, kInputBytes, 0);':
        '      id<MTLBuffer> prefix_input = FgConnectedMacLivePrefixInput();\n'
        '      if (prefix_input == nil)\n'
        '        throw std::runtime_error("live allocation37 binding missing");\n'
        '      std::vector<std::uint8_t> prefix_input_before;\n'
        '      if (NSFGTailAuditEnabled())\n'
        '        prefix_input_before = Payload(prefix_input, kInputBytes);',
    '      auto l52_confidence = ConnectedTexture(\n'
    '          device, MTLPixelFormatR16Float, 320, 180,\n'
    '          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,\n'
    '          std::vector<std::uint8_t>(320u * 180u * 2u, poison), 2);':
        '      id<MTLTexture> l52_confidence = FgConnectedMacAllocation29();\n'
        '      if (l52_confidence == nil)\n'
        '        throw std::runtime_error("live allocation29 binding missing");',
    '      [l52_encoder setBuffer:l52_params_buffer offset:0 atIndex:2];':
        '      [l52_encoder setBuffer:l52_params_buffer offset:0 atIndex:2];\n'
        '      [l52_encoder setBuffer:FgConnectedMacEx2UnitTable() offset:kGuardBytes atIndex:3];\n'
        '      [l52_encoder setBuffer:FgConnectedMacEx2FractionTable() offset:kGuardBytes atIndex:4];',
    '      [l54_encoder setBytes:&l54_params length:sizeof(l54_params) atIndex:3];':
        '      [l54_encoder setBytes:&l54_params length:sizeof(l54_params) atIndex:3];\n'
        '      [l54_encoder setBuffer:FgConnectedMacEx2UnitTable() offset:kGuardBytes atIndex:4];\n'
        '      [l54_encoder setBuffer:FgConnectedMacEx2FractionTable() offset:kGuardBytes atIndex:5];',
    '      const bool pass = mismatches == 0 && canaries == 0;':
        '      const bool pass = mismatches == 0 && canaries == 0;\n'
        '      for (std::size_t index = 0; index < checkpoints.size(); ++index)\n'
        '        NSFGRecordTailCheckpoint(index, checkpoints[index].c_str());\n'
        '      NSFGRecordTailResourceAllocations(NSFGTailPassResourceAllocations());\n'
        '      if (pass) NSFGCopyTailOutput(queue, tail_color);',
    '      id<MTLDevice> device = MTLCreateSystemDefaultDevice();':
        '      id<MTLDevice> device = NSFGTailDevice();\n'
        '      NSFGTailBeginResourcePass(device);\n'
        '      const bool use_live_auxiliaries = NSFGTailUsesLiveAuxiliaries();\n'
        '      const NSFGTailLiveResources* live = NSFGTailGetLiveResources();\n'
        '      const bool runtime_audit = NSFGTailAuditEnabled();\n'
        '      if (use_live_auxiliaries && live == nullptr)\n'
        '        throw std::runtime_error("live auxiliary resources are not bound");',
    '      id<MTLCommandQueue> queue = [device newCommandQueue];':
        '      id<MTLCommandQueue> queue = NSFGTailCommandQueue();',
    '      std::filesystem::create_directories(output);':
        '      (void)output;',
    '      std::ofstream receipt(output / "run-receipt.json");':
        '      std::ostringstream receipt;',
    '      Wait(core_command);\n':
        '      Wait(core_command);\n'
        '      NSFGRecordTailExports(l83, G, l53_a, l53_b, kGuardBytes, l54_blend, tail_sigmoid, G);\n'
        '      if (std::getenv("NSFG_BLOCK_TIMING") != nullptr && std::getenv("NSFG_SCALAR_BLOCK") == nullptr) nsfg_guarded::TheBlock().ReportTimings(device);\n',
    '      Wait(l87_command);':
        '      Wait(l87_command);\n'
        '      if (!NSFGTailAuditEnabled()) {\n'
        '        NSFGRecordTailResourceAllocations(NSFGTailPassResourceAllocations());\n'
        '        NSFGCopyTailOutput(queue, tail_color);\n'
        '        return 0;\n'
        '      }',
    '}  // namespace\n\nint FgRuntimeFullTailMain(int argc, const char* argv[]) {':
        '}  // namespace\n\n'
        'void NSFGTailReleasePersistentResources(void) {\n'
        '  NSFGTailReleasePersistentResourcesImpl();\n'
        '}\n\n'
        'int FgRuntimeFullTailMain(int argc, const char* argv[]) {',
}
for old, new in replacements.items():
    if source.count(old) != 1:
        raise SystemExit(f'runtime tail patch anchor drift: {old[:60]!r}')
    source = source.replace(old, new)
live_replacements = [
('''      const auto depth_before = BinaryDepthMac();
      auto depth = ConnectedTexture(device, MTLPixelFormatR8Unorm, 320, 180,
                                    MTLTextureUsageShaderRead, depth_before, 1);''',
'''      std::vector<std::uint8_t> depth_before;
      id<MTLTexture> depth = nil;
      if (use_live_auxiliaries) {
        depth = live->auxiliary_mask;
        if (depth == nil || depth.pixelFormat != MTLPixelFormatR8Unorm ||
            depth.width != 320 || depth.height != 180)
          throw std::runtime_error("live L52 auxiliary-mask contract");
        if (NSFGTailAuditEnabled()) depth_before = ConnectedReadTexture(depth, 1);
      } else {
        depth_before = BinaryDepthMac();
        depth = ConnectedTexture(device, MTLPixelFormatR8Unorm, 320, 180,
                                 MTLTextureUsageShaderRead, depth_before, 1);
      }'''),
('''      const std::array<std::vector<std::uint8_t>, 4> l53_input_before = {
          m53::CurrToPrevFlow(), m53::PrevToCurrFlow(false),
          m53::CurrDepth(), m53::PrevDepth()};
      auto l53_flow_a = ConnectedTexture(
          device, MTLPixelFormatRG32Float, 320, 180,
          MTLTextureUsageShaderRead, l53_input_before[0], 8);
      auto l53_flow_b = ConnectedTexture(
          device, MTLPixelFormatRG16Float, 320, 180,
          MTLTextureUsageShaderRead, l53_input_before[1], 4);
      auto l53_depth_a = ConnectedTexture(
          device, MTLPixelFormatR32Uint, 320, 180,
          MTLTextureUsageShaderRead, l53_input_before[2], 4);
      auto l53_depth_b = ConnectedTexture(
          device, MTLPixelFormatR32Uint, 320, 180,
          MTLTextureUsageShaderRead, l53_input_before[3], 4);''',
'''      std::array<std::vector<std::uint8_t>, 4> l53_input_before;
      id<MTLTexture> l53_flow_a = nil, l53_flow_b = nil;
      id<MTLTexture> l53_depth_a = nil, l53_depth_b = nil;
      if (use_live_auxiliaries) {
        l53_flow_a = live->flow;
        l53_flow_b = live->working_vector;
        l53_depth_a = live->current_depth;
        l53_depth_b = live->previous_depth;
        if (l53_flow_a == nil || l53_flow_a.pixelFormat != MTLPixelFormatRG32Float ||
            l53_flow_b == nil || l53_flow_b.pixelFormat != MTLPixelFormatRG16Float ||
            l53_depth_a == nil || l53_depth_a.pixelFormat != MTLPixelFormatR32Uint ||
            l53_depth_b == nil || l53_depth_b.pixelFormat != MTLPixelFormatR32Uint)
          throw std::runtime_error("live L53 resource contract");
        if (NSFGTailAuditEnabled()) {
          l53_input_before = {ConnectedReadTexture(l53_flow_a, 8),
                              ConnectedReadTexture(l53_flow_b, 4),
                              ConnectedReadTexture(l53_depth_a, 4),
                              ConnectedReadTexture(l53_depth_b, 4)};
        }
      } else {
        l53_input_before = {m53::CurrToPrevFlow(), m53::PrevToCurrFlow(false),
                            m53::CurrDepth(), m53::PrevDepth()};
        l53_flow_a = ConnectedTexture(device, MTLPixelFormatRG32Float, 320, 180,
            MTLTextureUsageShaderRead, l53_input_before[0], 8);
        l53_flow_b = ConnectedTexture(device, MTLPixelFormatRG16Float, 320, 180,
            MTLTextureUsageShaderRead, l53_input_before[1], 4);
        l53_depth_a = ConnectedTexture(device, MTLPixelFormatR32Uint, 320, 180,
            MTLTextureUsageShaderRead, l53_input_before[2], 4);
        l53_depth_b = ConnectedTexture(device, MTLPixelFormatR32Uint, 320, 180,
            MTLTextureUsageShaderRead, l53_input_before[3], 4);
      }'''),
('''      auto l54_blend = ConnectedTexture(
          device, MTLPixelFormatRG16Float, 320, 180,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(fg_mid_l54::kOutputBytes, poison), 4);''',
'''      id<MTLTexture> l54_blend = use_live_auxiliaries
          ? live->working_vector
          : ConnectedTexture(device, MTLPixelFormatRG16Float, 320, 180,
                MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
                std::vector<std::uint8_t>(fg_mid_l54::kOutputBytes, poison), 4);'''),
('''      const m55::GeneratedInputs l55_input;
      auto rgba_a = ConnectedTexture(
          device, MTLPixelFormatRGBA16Float, m55::kActiveWidth,
          m55::kActiveHeight, MTLTextureUsageShaderRead,
          l55_input.rgba_a, 8);
      auto rgba_b = ConnectedTexture(
          device, MTLPixelFormatRGBA16Float, m55::kActiveWidth,
          m55::kActiveHeight, MTLTextureUsageShaderRead,
          l55_input.rgba_b, 8);
      auto motion = ConnectedTexture(
          device, MTLPixelFormatRG16Float, m55::kCandidateWidth,
          m55::kCandidateHeight, MTLTextureUsageShaderRead,
          l55_input.motion, 4);''',
'''      std::array<std::vector<std::uint8_t>, 3> l55_input_before;
      id<MTLTexture> rgba_a = nil, rgba_b = nil, motion = nil;
      if (use_live_auxiliaries) {
        rgba_a = live->previous_history;
        rgba_b = live->next_history;
        motion = live->current_projection;
        if (rgba_a == nil || rgba_a.pixelFormat != MTLPixelFormatRGBA16Float ||
            rgba_b == nil || rgba_b.pixelFormat != MTLPixelFormatRGBA16Float ||
            motion == nil || motion.pixelFormat != MTLPixelFormatRG16Float)
          throw std::runtime_error("live L55 resource contract");
        if (NSFGTailAuditEnabled()) {
          l55_input_before = {ConnectedReadTexture(rgba_a, 8),
                              ConnectedReadTexture(rgba_b, 8),
                              ConnectedReadTexture(motion, 4)};
        }
      } else {
        const m55::GeneratedInputs generated;
        l55_input_before = {generated.rgba_a, generated.rgba_b, generated.motion};
        rgba_a = ConnectedTexture(device, MTLPixelFormatRGBA16Float,
            m55::kActiveWidth, m55::kActiveHeight, MTLTextureUsageShaderRead,
            l55_input_before[0], 8);
        rgba_b = ConnectedTexture(device, MTLPixelFormatRGBA16Float,
            m55::kActiveWidth, m55::kActiveHeight, MTLTextureUsageShaderRead,
            l55_input_before[1], 8);
        motion = ConnectedTexture(device, MTLPixelFormatRG16Float,
            m55::kCandidateWidth, m55::kCandidateHeight,
            MTLTextureUsageShaderRead, l55_input_before[2], 4);
      }'''),
('''      mismatches += Mismatches(l55_input.rgba_a,
                               ConnectedReadTexture(rgba_a, 8));
      mismatches += Mismatches(l55_input.rgba_b,
                               ConnectedReadTexture(rgba_b, 8));
      mismatches += Mismatches(l55_input.motion,
                               ConnectedReadTexture(motion, 4));''',
'''      mismatches += Mismatches(l55_input_before[0],
                               ConnectedReadTexture(rgba_a, 8));
      mismatches += Mismatches(l55_input_before[1],
                               ConnectedReadTexture(rgba_b, 8));
      mismatches += Mismatches(l55_input_before[2],
                               ConnectedReadTexture(motion, 4));'''),
('''      for (std::size_t index = 0; index < l53_input_before.size(); ++index) {
        const std::array<id<MTLTexture>,4> textures = {
            l53_flow_a,l53_flow_b,l53_depth_a,l53_depth_b};
        const std::array<NSUInteger,4> bytes_per_pixel = {8,4,4,4};
        mismatches += Mismatches(
            l53_input_before[index],
            ConnectedReadTexture(textures[index], bytes_per_pixel[index]));
      }''',
'''      for (std::size_t index = 0; index < l53_input_before.size(); ++index) {
        // In the live graph L54 intentionally overwrites working_vector_ only
        // after L53 has consumed it. The other L53 inputs remain read-only.
        if (use_live_auxiliaries && index == 1u) continue;
        const std::array<id<MTLTexture>,4> textures = {
            l53_flow_a,l53_flow_b,l53_depth_a,l53_depth_b};
        const std::array<NSUInteger,4> bytes_per_pixel = {8,4,4,4};
        mismatches += Mismatches(
            l53_input_before[index],
            ConnectedReadTexture(textures[index], bytes_per_pixel[index]));
      }'''),
]
for old, new in live_replacements:
    if source.count(old) != 1:
        raise SystemExit(f'runtime live-resource anchor drift: {old[:64]!r}')
    source = source.replace(old, new)
hot_path_replacements = [
# Entries for the model reads/hashes and the L55/L84/L83 audit read-backs moved into the source
# (CachedRead/CachedSha256/AuditOnly in macos_connected_l20_l87.mm); the remaining entries keep the
# L84-L86 checkpoint read-backs audit-only.
('''      const auto color84=TailReadTexture(tail_color,4);
      const auto mask84=TailReadTexture(tail_mask_output,1);
      const std::array<std::vector<std::uint8_t>,4> pyramid84 = {
          Payload(pyramid[0],kPyramidBytes[0]),
          Payload(pyramid[1],kPyramidBytes[1]),
          Payload(pyramid[2],kPyramidBytes[2]),
          Payload(pyramid[3],kPyramidBytes[3])};
      const auto checkpoint84=JoinVec({color84,mask84,pyramid84[0],
                                       pyramid84[1],pyramid84[2],
                                       pyramid84[3]});''',
'''      std::vector<std::uint8_t> color84,mask84,checkpoint84;
      std::array<std::vector<std::uint8_t>,4> pyramid84;
      if (runtime_audit) {
        color84=TailReadTexture(tail_color,4);
        mask84=TailReadTexture(tail_mask_output,1);
        pyramid84={Payload(pyramid[0],kPyramidBytes[0]),
                   Payload(pyramid[1],kPyramidBytes[1]),
                   Payload(pyramid[2],kPyramidBytes[2]),
                   Payload(pyramid[3],kPyramidBytes[3])};
        checkpoint84=JoinVec({color84,mask84,pyramid84[0],pyramid84[1],
                              pyramid84[2],pyramid84[3]});
      }'''),
('''      const auto checkpoint85=Payload(l85_output[0],l85_sizes[0]);''',
'''      std::vector<std::uint8_t> checkpoint85;
      if (runtime_audit) checkpoint85=Payload(l85_output[0],l85_sizes[0]);'''),
('''      const std::array<std::vector<std::uint8_t>,5> pyramid86 = {
          Payload(pyramid[0],kPyramidBytes[0]),
          Payload(pyramid[1],kPyramidBytes[1]),
          Payload(pyramid[2],kPyramidBytes[2]),
          Payload(pyramid[3],kPyramidBytes[3]),
          Payload(l85_output[0],l85_sizes[0])};
      const auto checkpoint86=JoinVec({pyramid86[0],pyramid86[1],
                                       pyramid86[2],pyramid86[3],
                                       pyramid86[4]});''',
'''      std::array<std::vector<std::uint8_t>,5> pyramid86;
      std::vector<std::uint8_t> checkpoint86;
      if (runtime_audit) {
        pyramid86={Payload(pyramid[0],kPyramidBytes[0]),
                   Payload(pyramid[1],kPyramidBytes[1]),
                   Payload(pyramid[2],kPyramidBytes[2]),
                   Payload(pyramid[3],kPyramidBytes[3]),
                   Payload(l85_output[0],l85_sizes[0])};
        checkpoint86=JoinVec({pyramid86[0],pyramid86[1],pyramid86[2],
                              pyramid86[3],pyramid86[4]});
      }'''),
]
for old, new in hot_path_replacements:
    if source.count(old) != 1:
        raise SystemExit(f'runtime hot-path anchor drift: {old[:64]!r}')
    source = source.replace(old, new)
needle = 'id<MTLTexture> ConnectedTexture('
if source.count(needle) != 1:
    raise SystemExit('runtime connected texture definition drift')
source = source.replace(needle, 'id<MTLTexture> NSFGAcceptedConnectedTexture(', 1)
anchor = 'void PrefixEncode3x3(id<MTLCommandBuffer> command,'
if source.count(anchor) != 1:
    raise SystemExit('runtime tail cache insertion anchor drift')
source = source.replace(anchor, cache + '\n\n' + anchor, 1)
pattern = re.compile(
    r'\[device newBufferWithBytes:&([A-Za-z0-9_]+)\s+'
    r'length:sizeof\(\1\)\s+options:MTLResourceStorageModeShared\]')
source, raw_count = pattern.subn(
    lambda match: f'NSFGCachedRawBuffer(device,&{match.group(1)},sizeof({match.group(1)}))',
    source)
if raw_count != 12:
    raise SystemExit(f'runtime raw parameter cache count drift: {raw_count}')
pathlib.Path(sys.argv[2]).write_text(source)
