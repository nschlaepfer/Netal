# Pool fusion transformer: takes a pooled family's pipeline .metal and harness .mm (design six layout) and
# produces the "_pf" variants: 4x2 pixel blocks, shuffle pool in the fast-pass epilogue, pass-3 re-pool over
# list2, pool dispatch removed. Reusable on later versions of the same files.
import io, re, sys
metal_in, mm_in, prefix, W, H, CO, tag_in, tag_out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7], sys.argv[8]
s = io.open(metal_in).read().replace(tag_in, tag_out)
def sub(a, b):
    global s; assert a in s, "metal: " + a[:80]; s = s.replace(a, b)
nbx, nby, OW = W // 4, H // 2, W // 2
sub(f"  const uint pixel = group.x * 8u + mr, oc0 = group.y * 16u + mc;\n  if (pixel >= {W}u * {H}u || oc0 >= {CO}u) return;\n  const uint px = pixel % {W}u, py = pixel / {W}u;",
    f"  // 4x2 pixel block per simdgroup so every 2x2 pool window is complete inside it\n  const uint bx = group.x % {nbx}u, by = group.x / {nbx}u, oc0 = group.y * 16u + mc;\n  if (by >= {nby}u || oc0 >= {CO}u) return;\n  const uint px = bx * 4u + (mr & 3u), py = by * 2u + (mr >> 2u);")
# out buffer on the bounded kernel: after after2 (index 10)
m = re.search(r"(device half\* after2 \[\[buffer\(10\)\]\],\n)(\s*uint lane \[\[thread_index_in_simdgroup\]\])", s); assert m, "bounded signature"
s = s[:m.start()] + m.group(1) + "    device half* out [[buffer(11)]],\n" + m.group(2) + s[m.end():]
old = f"    const uint index = (py * {W}u + px) * {CO}u + oc0 + 8u * g + h;\n    inter[index] = activate(half(acc[g].thread_elements()[h]));"
sub(old, f"""    const uint index = (py * {W}u + px) * {CO}u + oc0 + 8u * g + h;
    const half v = activate(half(acc[g].thread_elements()[h]));
    inter[index] = v;
    // 2x2 pool in the oracle's order; the window's other values sit in lanes ^2, ^16, ^18
    const ushort vb = as_type<ushort>(v);
    const half pb = as_type<half>(simd_shuffle_xor(vb, 2u)), pc = as_type<half>(simd_shuffle_xor(vb, 16u)), pd = as_type<half>(simd_shuffle_xor(vb, 18u));
    if ((mr & 1u) == 0u && (mr >> 2u) == 0u) {{
      half p = half(v + pb); p = half(p + pc); p = half(p + pd);
      out[((py >> 1u) * {OW}u + (px >> 1u)) * {CO}u + oc0 + 8u * g + h] = half(p * half(0.25f));
    }}""")
s += f'''
kernel void {prefix}_repool{tag_out}(device const half* inter [[buffer(0)]], device half* out [[buffer(1)]],
    device const uint* list2 [[buffer(2)]], device const uint* counter2 [[buffer(3)]], uint tid [[thread_position_in_grid]]) {{
  if (tid >= counter2[0]) return;
  const uint linear = list2[tid], oc = linear % {CO}u, pixel = linear / {CO}u, x = (pixel % {W}u) & ~1u, y = (pixel / {W}u) & ~1u;
  const half a = inter[(y * {W}u + x) * {CO}u + oc], bb = inter[(y * {W}u + x + 1u) * {CO}u + oc];
  const half c = inter[((y + 1u) * {W}u + x) * {CO}u + oc], d = inter[((y + 1u) * {W}u + x + 1u) * {CO}u + oc];
  half p = half(a + bb); p = half(p + c); p = half(p + d);
  out[((y >> 1u) * {OW}u + (x >> 1u)) * {CO}u + oc] = half(p * half(0.25f));
}}
'''
io.open(metal_in.replace(tag_in, tag_out), "w").write(s)
h = io.open(mm_in).read().replace(tag_in, tag_out)
m = re.search(r"\[e1 setBuffer:(\w+) offset:0 atIndex:10\];", h); assert m, "harness: bounded index 10"
outbuf = re.search(r"\[e6 setBuffer:(\w+) offset:0 atIndex:1\];", h); interbuf = re.search(r"\[e6 setBuffer:(\w+) offset:0 atIndex:0\];", h); assert outbuf and interbuf, "harness: pool bindings"
h = h.replace(m.group(0), m.group(0) + f" [e1 setBuffer:{outbuf.group(1)} offset:0 atIndex:11];")
i = h.find("id<MTLComputeCommandEncoder> e6=[cb computeCommandEncoder]; [e6 setComputePipelineState:pool];"); assert i > 0, "harness: pool dispatch"
j = h.find("[e6 endEncoding];", i) + len("[e6 endEncoding];")
h = h[:i] + f"id<MTLComputeCommandEncoder> e6=[cb computeCommandEncoder]; [e6 setComputePipelineState:repool]; [e6 setBuffer:{interbuf.group(1)} offset:0 atIndex:0]; [e6 setBuffer:{outbuf.group(1)} offset:0 atIndex:1]; [e6 setBuffer:blist2 offset:0 atIndex:2]; [e6 setBuffer:bcnt2 offset:0 atIndex:3];\n        [e6 dispatchThreadgroupsWithIndirectBuffer:bind2 indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e6 endEncoding];" + h[j:]
pm = re.search(r'pool=Pipe\(dev,L,@"(\w+)"\),', h); assert pm, "harness: pool pipe"
h = h.replace(pm.group(0), pm.group(0) + f' repool=Pipe(dev,L,@"{prefix}_repool{tag_out}"),')
io.open(mm_in.replace(tag_in, tag_out), "w").write(h)
print("ok", metal_in.replace(tag_in, tag_out), mm_in.replace(tag_in, tag_out))
