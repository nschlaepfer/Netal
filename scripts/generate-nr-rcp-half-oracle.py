#!/usr/bin/env python3
"""Generate an RCP-domain probe from the frozen RSQ-domain harness."""
import argparse
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--mac-output',type=Path);a=p.parse_args()
    if a.output.exists():raise ValueError('preserve prior generated source')
    repo=Path(__file__).resolve().parents[1]
    source=(repo/'src/nr_swin/rsq_half_oracle.cu').read_text()
    if source.count('rsqrt.approx.ftz.f32')!=2:raise ValueError('RSQ harness changed')
    source=source.replace('rsqrt.approx.ftz.f32','rcp.approx.ftz.f32').replace('rsq-half-rtx.v1','rcp-half-rtx.v1')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(source)
    if a.mac_output:
        if a.mac_output.exists():raise ValueError('preserve generated Mac source')
        mac=(repo/'src/nr_swin/rsq_half_probe.mm').read_text()
        if mac.count('rsq-half-metal.v1')!=1:raise ValueError('Mac probe schema anchor changed')
        a.mac_output.parent.mkdir(parents=True,exist_ok=True)
        a.mac_output.write_text(mac.replace('rsq-half-metal.v1','rcp-half-metal.v1'))


if __name__=='__main__':main()
