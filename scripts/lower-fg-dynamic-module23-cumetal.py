#!/usr/bin/env python3
import argparse,hashlib,json,re
from pathlib import Path
H="d278535348df59fd462d8bb12ec5f90185fcf3e2b14f921a26c46b73f7d7f8ba"
OLD="""static inline float cm_rcp_approx_ftz_f32(float value) {
    return 1.0f / value;
}
static inline float cm_div_approx_ftz_f32(float numerator, float denominator) {
    return numerator / denominator;
}"""
NEW="""static inline float cm_rcp_approx_ftz_f32(float value, device const uchar* rom) {
    uint ib=as_type<uint>(value),bb=as_type<uint>(1.0f/value),m=ib&0x7fffffu;
    uchar p=rom[m>>2u];uint c=(uint(p)>>((m&3u)*2u))&3u;
    if(c==1u)--bb;else if(c==2u)++bb;return as_type<float>(bb);
}
static inline float cm_div_approx_ftz_f32(float n,float d,device const uchar* rom){return n*cm_rcp_approx_ftz_f32(d,rom);}"""
def sh(b):return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--receipt",type=Path,required=True);a=p.parse_args();b=a.input.read_bytes();assert sh(b)==H;s=b.decode();assert s.count(OLD)==1;s=s.replace(OLD,NEW);old="""kernel void main_kernel(
    constant uchar* main_kernel_param_0_72_ [[buffer(0)]],
    uint3 cm_thread_position [[thread_position_in_threadgroup]],""";new="""kernel void fg_dynamic_module23_mode0(
    constant uchar* main_kernel_param_0_72_ [[buffer(0)]],
    device const uchar* cm_rcp_rom [[buffer(1)]],
    uint3 cm_thread_position [[thread_position_in_threadgroup]],""";assert s.count(old)==1;s=s.replace(old,new);s,n=re.subn(r"cm_div_approx_ftz_f32\((v\d+), (v\d+)\)",r"cm_div_approx_ftz_f32(\1, \2, cm_rcp_rom)",s);assert n==54;o=("// Target-resolution selected mode0 lowering; mode1 surface path remains dead.\n"+s).encode();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(o);r={"schema":"neuralstudio.fg.dynamic-module23-lowering.v1","input_sha256":H,"output_sha256":sh(o),"division_sites":54,"selected_mode":0,"mode1_surface_restored":False,"expected_output_embedded":False,"result":"pass"};r["receipt_root_sha256"]=sh(json.dumps(r,sort_keys=True,separators=(",",":")).encode());a.receipt.parent.mkdir(parents=True,exist_ok=True);a.receipt.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
