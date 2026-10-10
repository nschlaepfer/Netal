#!/usr/bin/env python3
"""Connect the exact selected L3-L17 RTX prefix to the live L18-L83 tail."""

from __future__ import annotations

import argparse
from pathlib import Path


def once(source: str, before: str, after: str, label: str) -> str:
    count = source.count(before)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return source.replace(before, after)


TAIL_CALL = r'''    const fs::path tail_output=output_root/"l18-l83";
    FgSubrectSetRtxLivePrefixInput(static_cast<std::uint64_t>(
        model_input_shared.cuda_ptr+kGuardBytes));
    std::string modules_argument=modules.string();
    std::string model_argument=model.string();
    std::string tail_output_argument=tail_output.string();
    std::string poison_argument=argv[5];
    std::array<char*,6> tail_argv={
        argv[0],modules_argument.data(),model_argument.data(),
        tail_output_argument.data(),const_cast<char*>(run_id.data()),
        poison_argument.data()};
    const int tail_status=FgSubrectTailMain(
        static_cast<int>(tail_argv.size()),tail_argv.data());
    FgSubrectSetRtxLivePrefixInput(0);
    if(tail_status!=0)
      throw std::runtime_error("connected subrect L18-L83 tail failed");
    const auto model_input_after_tail_guarded=ReadBuffer(gpu,model_input);
    const auto model_input_after_tail=Payload(
        model_input_after_tail_guarded,connected::kModelInputBytes);
'''


def generate(source: str) -> str:
    source = once(
        source,
        '#include "contract.h"',
        '#include "contract.h"\n'
        '#include "../fg_connected_l20_l87/rtx_shared_cuda_buffer.h"\n'
        '#include "rtx_subrect_live_handoff.h"\n'
        'int FgSubrectTailMain(int argc, char** argv);',
        'connected includes',
    )
    source = once(
        source,
        '    (void)model;',
        '',
        'model usage',
    )
    source = once(
        source,
        '    auto model_input=UploadBuffer(gpu,Guarded(\n'
        '        std::vector<std::uint8_t>(connected::kModelInputBytes,0u)));',
        '    auto model_input_shared=\n'
        '        fg_connected_l20_l87::UploadSharedCudaBuffer(\n'
        '            gpu,Guarded(std::vector<std::uint8_t>(\n'
        '                connected::kModelInputBytes,0u)));\n'
        '    GpuBuffer& model_input=model_input_shared.d3d;',
        'shared allocation37',
    )
    source = once(
        source,
        '    std::size_t input_mismatches=0;',
        TAIL_CALL + '\n    std::size_t input_mismatches=0;',
        'tail call',
    )
    source = once(
        source,
        '    input_mismatches+=Mismatches(connected::InitialAllocation29(),\n'
        '                                 ReadTexture(gpu,allocation29));',
        '    input_mismatches+=Mismatches(connected::InitialAllocation29(),\n'
        '                                 ReadTexture(gpu,allocation29));\n'
        '    input_mismatches+=Mismatches(model_input_bytes,\n'
        '                                 model_input_after_tail);',
        'live input immutability',
    )
    source = once(
        source,
        '    std::size_t canary_mismatches=CanaryMismatches(\n'
        '        model_guarded,connected::kModelInputBytes);',
        '    std::size_t canary_mismatches=CanaryMismatches(\n'
        '        model_guarded,connected::kModelInputBytes);\n'
        '    canary_mismatches+=CanaryMismatches(\n'
        '        model_input_after_tail_guarded,connected::kModelInputBytes);',
        'live input canary',
    )
    source = once(
        source,
        '           <<"  \\"address_fields_recorded\\": false,\\n"\n'
        '           <<"  \\"result\\": \\""<<(pass?"pass":"fail")<<"\\"\\n}\\n";',
        '           <<"  \\"address_fields_recorded\\": false,\\n"\n'
        '           <<"  \\"connected_tail_receipt\\": '
        '\\"l18-l83/run-receipt.json\\",\\n"\n'
        '           <<"  \\"result\\": \\""<<(pass?"pass":"fail")<<"\\"\\n}\\n";',
        'tail receipt pointer',
    )
    source = once(
        source,
        '    std::cout<<"allocation37_sha256="<<Sha256(model_input_bytes)<<\'\\n\'\n'
        '             <<"checkpoint_count="<<checkpoints.size()<<\'\\n\'\n'
        '             <<"result="<<(pass?"pass":"fail")<<\'\\n\';',
        '    model_input_shared.Release();\n'
        '    std::cout<<"allocation37_sha256="<<Sha256(model_input_bytes)<<\'\\n\'\n'
        '             <<"checkpoint_count="<<checkpoints.size()<<\'\\n\'\n'
        '             <<"result="<<(pass?"pass":"fail")<<\'\\n\';',
        'shared release',
    )
    return '// Generated connected subrect RTX early host.\n' + source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f'preserve existing output: {args.output}')
    result = generate(args.source.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result)
    print(f'generated={args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
