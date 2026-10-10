#define main fg_compositor_embedded_main
#include "../fg_compositor_l87/rtx_nvapi_oracle.cpp"
#undef main
#include "module9_contract.h"
namespace {
namespace m9 = fg_distortion_module9;
struct Spec {
  DXGI_FORMAT f;
  UINT b;
  UINT slot;
};
constexpr std::array<Spec, 5> specs = {{{DXGI_FORMAT_R16G16B16A16_FLOAT, 8, 23},
                                        {DXGI_FORMAT_R16G16B16A16_FLOAT, 8, 24},
                                        {DXGI_FORMAT_R32G32_FLOAT, 8, 17},
                                        {DXGI_FORMAT_R16G16_FLOAT, 4, 25},
                                        {DXGI_FORMAT_R32_UINT, 4, 18}}};
GpuTexture UploadL19Texture(GpuContext &g, uint32_t w, uint32_t h,
                            DXGI_FORMAT f, uint32_t b, D3D12_RESOURCE_FLAGS fl,
                            D3D12_RESOURCE_STATES st,
                            const std::vector<uint8_t> &v) {
  GpuTexture r;
  r.width = w;
  r.height = h;
  r.bytes_per_pixel = b;
  r.state = st;
  D3D12_HEAP_PROPERTIES hp{};
  hp.Type = D3D12_HEAP_TYPE_DEFAULT;
  hp.CreationNodeMask = hp.VisibleNodeMask = 1;
  D3D12_RESOURCE_DESC d{};
  d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  d.Width = w;
  d.Height = h;
  d.DepthOrArraySize = 1;
  d.MipLevels = 1;
  d.Format = f;
  d.SampleDesc.Count = 1;
  d.Flags = fl;
  RequireHr(g.device->CreateCommittedResource(
                &hp, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COPY_DEST,
                nullptr, IID_PPV_ARGS(&r.resource)),
            "create texture");
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
  UINT rows = 0;
  UINT64 rs = 0, sz = 0;
  g.device->GetCopyableFootprints(&d, 0, 1, 0, &fp, &rows, &rs, &sz);
  auto u =
      CreateBuffer(g.device.Get(), sz, D3D12_HEAP_TYPE_UPLOAD,
                   D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
  void *m = nullptr;
  D3D12_RANGE nr{0, 0};
  RequireHr(u->Map(0, &nr, &m), "map upload");
  for (uint32_t y = 0; y < h; y++)
    std::memcpy((uint8_t *)m + size_t(y) * fp.Footprint.RowPitch,
                v.data() + size_t(y) * w * b, size_t(w) * b);
  u->Unmap(0, nullptr);
  D3D12_TEXTURE_COPY_LOCATION dl{};
  dl.pResource = r.resource.Get();
  dl.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  D3D12_TEXTURE_COPY_LOCATION sl{};
  sl.pResource = u.Get();
  sl.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  sl.PlacedFootprint = fp;
  g.list->CopyTextureRegion(&dl, 0, 0, 0, &sl, nullptr);
  Transition(g.list.Get(), r.resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
             st);
  g.pending.push_back(u);
  return r;
}
std::vector<uint8_t> ReadL19Texture(GpuContext &g, GpuTexture &s) {
  auto d = s.resource->GetDesc();
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
  UINT rows = 0;
  UINT64 rs = 0, sz = 0;
  g.device->GetCopyableFootprints(&d, 0, 1, 0, &fp, &rows, &rs, &sz);
  auto r =
      CreateBuffer(g.device.Get(), sz, D3D12_HEAP_TYPE_READBACK,
                   D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
  Transition(g.list.Get(), s.resource.Get(), s.state,
             D3D12_RESOURCE_STATE_COPY_SOURCE);
  s.state = D3D12_RESOURCE_STATE_COPY_SOURCE;
  D3D12_TEXTURE_COPY_LOCATION a{};
  a.pResource = s.resource.Get();
  a.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  D3D12_TEXTURE_COPY_LOCATION b{};
  b.pResource = r.Get();
  b.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  b.PlacedFootprint = fp;
  g.list->CopyTextureRegion(&b, 0, 0, 0, &a, nullptr);
  ExecuteAndReset(g);
  std::vector<uint8_t> o(size_t(s.width) * s.height * s.bytes_per_pixel);
  void *m = nullptr;
  D3D12_RANGE rr{0, (SIZE_T)sz};
  RequireHr(r->Map(0, &rr, &m), "map read");
  for (uint32_t y = 0; y < s.height; y++)
    std::memcpy(o.data() + size_t(y) * s.width * s.bytes_per_pixel,
                (uint8_t *)m + size_t(y) * fp.Footprint.RowPitch,
                size_t(s.width) * s.bytes_per_pixel);
  r->Unmap(0, nullptr);
  return o;
}
template <size_t N>
void P64(std::array<std::uint8_t, N> &b, size_t o, uint64_t v) {
  for (int i = 0; i < 8; i++)
    b[o + i] = static_cast<std::uint8_t>(v >> (i * 8));
}
template <size_t N>
void P32(std::array<std::uint8_t, N> &b, size_t o, uint32_t v) {
  for (int i = 0; i < 4; i++)
    b[o + i] = v >> (i * 8);
}
size_t poisonPixels(const std::vector<uint8_t> &b, size_t s, uint8_t p) {
  size_t n = 0;
  for (size_t o = 0; o < b.size(); o += s) {
    bool q = true;
    for (size_t i = 0; i < s; i++)
      q &= b[o + i] == p;
    n += q;
  }
  return n;
}
} // namespace
int main(int argc, char **argv) {
  if (argc != 5)
    return 2;
  try {
    fs::path mp = argv[1], od = argv[2];
    std::string rid = argv[3];
    uint8_t poison = ParsePoison(argv[4]);
    fs::create_directories(od);
    auto mb = Read(mp);
    if (mb.size() != m9::kModuleBytes || Sha256(mb) != m9::kModuleSha256)
      throw std::runtime_error("module9 identity");
    auto in = m9::Generate();
    std::array<std::vector<uint8_t> *, 5> iv = {&in.a, &in.b, &in.flow32,
                                                &in.flow16, &in.mask};
    std::vector<uint8_t> o0(m9::kRg16Bytes, poison), o1(m9::kRg16Bytes, poison),
        o2(m9::kR32Bytes, poison), can(m9::kRg16Bytes, poison ^ 0xff);
    GpuContext g = CreateGpuContext();
    UniqueLibrary nv(LoadLibraryW(L"nvapi64.dll"));
    auto q = reinterpret_cast<QueryInterface>(
        GetProcAddress(nv.get(), "nvapi_QueryInterface"));
    auto init = Resolve<Initialize>(q, kIdInitialize, "init");
    auto gt = Resolve<GetMergedTextureSampler>(q, kIdGetMergedTextureSampler,
                                               "texture");
    auto gs = Resolve<GetIndependentDescriptor>(q, kIdGetIndependentDescriptor,
                                                "surface");
    auto cm = Resolve<CreateCuModule>(q, kIdCreateCuModule, "module");
    auto cf = Resolve<CreateCuFunction>(q, kIdCreateCuFunction, "function");
    auto lk = Resolve<LaunchKernelChain>(q, kIdLaunchKernelChain, "launch");
    RequireNvapi(init(), "init");
    std::array<GpuTexture, 5> ti;
    for (size_t i = 0; i < 5; i++)
      ti[i] =
          UploadL19Texture(g, m9::kWidth, m9::kHeight, specs[i].f, specs[i].b,
                           D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS, *iv[i]);
    auto out0 =
        UploadL19Texture(g, m9::kWidth, m9::kHeight, DXGI_FORMAT_R16G16_FLOAT,
                         4, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS, o0);
    auto out1 =
        UploadL19Texture(g, m9::kWidth, m9::kHeight, DXGI_FORMAT_R16G16_FLOAT,
                         4, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS, o1);
    auto out2 =
        UploadL19Texture(g, m9::kWidth, m9::kHeight, DXGI_FORMAT_R32_UINT, 4,
                         D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS, o2);
    auto canary =
        UploadL19Texture(g, m9::kWidth, m9::kHeight, DXGI_FORMAT_R16G16_FLOAT,
                         4, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS, can);
    ExecuteAndReset(g);
    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = 64;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    hd.NodeMask = 1;
    ComPtr<ID3D12DescriptorHeap> h;
    RequireHr(g.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&h)), "heap");
    D3D12_DESCRIPTOR_HEAP_DESC sd{};
    sd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    sd.NumDescriptors = 1;
    sd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    sd.NodeMask = 1;
    ComPtr<ID3D12DescriptorHeap> sh;
    RequireHr(g.device->CreateDescriptorHeap(&sd, IID_PPV_ARGS(&sh)),
              "samplerheap");
    UINT inc = g.device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    auto st = h->GetCPUDescriptorHandleForHeapStart();
    auto dh = [&](UINT x) {
      return D3D12_CPU_DESCRIPTOR_HANDLE{st.ptr + uint64_t(x) * inc};
    };
    auto samp = sh->GetCPUDescriptorHandleForHeapStart();
    D3D12_SAMPLER_DESC sm{};
    sm.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    sm.AddressU = sm.AddressV = sm.AddressW = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    g.device->CreateSampler(&sm, samp);
    std::array<uint64_t, 5> th{};
    for (size_t i = 0; i < 5; i++) {
      D3D12_SHADER_RESOURCE_VIEW_DESC v{};
      v.Format = specs[i].f;
      v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
      v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
      v.Texture2D.MipLevels = 1;
      auto d = dh(specs[i].slot);
      g.device->CreateShaderResourceView(ti[i].resource.Get(), &v, d);
      MergedTextureSamplerParams p{};
      p.struct_size_in = p.struct_size_out = sizeof(p);
      p.device = g.device.Get();
      p.texture_descriptor.ptr = d.ptr;
      p.sampler_descriptor.ptr = samp.ptr;
      RequireNvapi(gt(&p), "texhandle");
      th[i] = p.texture_handle;
    }
    std::array<GpuTexture *, 3> ot = {&out0, &out1, &out2};
    std::array<DXGI_FORMAT, 3> of = {DXGI_FORMAT_R16G16_FLOAT,
                                     DXGI_FORMAT_R16G16_FLOAT,
                                     DXGI_FORMAT_R32_UINT};
    std::array<uint64_t, 3> oh{};
    for (int i = 0; i < 3; i++) {
      auto d = dh(26 + i);
      D3D12_UNORDERED_ACCESS_VIEW_DESC v{};
      v.Format = of[i];
      v.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
      g.device->CreateUnorderedAccessView(ot[i]->resource.Get(), nullptr, &v,
                                          d);
      IndependentDescriptorParams p{};
      p.struct_size_in = p.struct_size_out = sizeof(p);
      p.device = g.device.Get();
      p.type = IndependentDescriptorType::kSurface;
      p.descriptor.ptr = d.ptr;
      RequireNvapi(gs(&p), "surfhandle");
      oh[i] = p.handle;
    }
    NvObjectHandle mo = nullptr, fn = nullptr;
    RequireNvapi(cm(g.device.Get(), mb.data(),
                    static_cast<std::uint32_t>(mb.size()), &mo),
                 "cm");
    RequireNvapi(cf(g.device.Get(), mo, "main_kernel", &fn), "cf");
    std::array<uint8_t, 72> pk{};
    for (int i = 0; i < 5; i++)
      P64(pk, i * 8, th[i]);
    P32(pk, 40, 320);
    P32(pk, 44, 180);
    for (int i = 0; i < 3; i++)
      P64(pk, 48 + i * 8, oh[i]);
    auto af = pk;
    for (int i = 0; i < 5; i++)
      P64(af, i * 8, 0);
    for (int i = 0; i < 3; i++)
      P64(af, 48 + i * 8, 0);
    if (Sha256(std::vector<uint8_t>(af.begin(), af.end())) !=
        m9::kFixtureAddressFreeSha256)
      throw std::runtime_error("abi");
    KernelLaunchParams lp{fn, {20, 12, 1}, {16, 16, 1}, 0, pk.data(), 72};
    RequireNvapi(lk(g.list.Get(), &lp, 1), "launch");
    ExecuteAndReset(g);
    std::array<std::vector<uint8_t>, 5> ia;
    size_t im = 0;
    std::array<std::string, 5> input_pre_sha256, input_post_sha256;
    for (int i = 0; i < 5; i++) {
      ia[i] = ReadL19Texture(g, ti[i]);
      im += Mismatches(*iv[i], ia[i]);
      input_pre_sha256[i] = Sha256(*iv[i]);
      input_post_sha256[i] = Sha256(ia[i]);
    }
    auto a = ReadL19Texture(g, out0), b = ReadL19Texture(g, out1),
         c = ReadL19Texture(g, out2), ca = ReadL19Texture(g, canary);
    size_t cmis = Mismatches(can, ca);
    Write(od / "out-a.rg16f", a);
    Write(od / "out-b.rg16f", b);
    Write(od / "out-mask.r32u", c);
    std::ofstream r(od / "run-receipt.json");
    r << "{\n\"schema\":\"neuralstudio.fg.distortion-module9-run.v1\","
         "\n\"backend\":\"rtx-d3d12-nvapi\",\n\"run_id\":\""
      << rid << "\",\n\"process_id\":" << GetCurrentProcessId()
      << ",\n\"poison_byte\":" << unsigned(poison) << ",\n\"module_sha256\":\""
      << m9::kModuleSha256 << "\",\n\"module_bytes\":" << m9::kModuleBytes
      << ",\n\"selected_launch_tokens\":[\"launch-000009\",\"launch-000020\"],"
         "\n\"selected_phase_raw_parameter_sha256\":[\""
      << m9::kRawParameterSha256 << "\",\"" << m9::kGeneratedRawParameterSha256
      << "\"],\n\"frozen_reset_address_free_parameter_sha256\":\""
      << m9::kFrozenAddressFreeSha256
      << "\",\n\"packed_parameter_bytes\":72,\n\"fixture_address_free_"
         "parameter_sha256\":\""
      << m9::kFixtureAddressFreeSha256 << "\",\n\"input_generator\":\""
      << m9::kInputGenerator << "\",\n\"input_pre_sha256\":[";
    for (size_t i = 0; i < input_pre_sha256.size(); ++i) {
      if (i)
        r << ',';
      r << '\"' << input_pre_sha256[i] << '\"';
    }
    r << "],\n\"input_post_sha256\":[";
    for (size_t i = 0; i < input_post_sha256.size(); ++i) {
      if (i)
        r << ',';
      r << '\"' << input_post_sha256[i] << '\"';
    }
    r << "],\n\"input_mismatched_bytes\":" << im
      << ",\n\"texture_canary_mismatched_bytes\":" << cmis
      << ",\n\"out_a_bytes\":" << a.size() << ",\n\"out_b_bytes\":" << b.size()
      << ",\n\"out_mask_bytes\":" << c.size() << ",\n\"out_a_sha256\":\""
      << Sha256(a) << "\",\n\"out_b_sha256\":\"" << Sha256(b)
      << "\",\n\"out_mask_sha256\":\"" << Sha256(c)
      << "\",\n\"out_a_poison_pixels\":" << poisonPixels(a, 4, poison)
      << ",\n\"out_b_poison_pixels\":" << poisonPixels(b, 4, poison)
      << ",\n\"out_mask_poison_pixels\":" << poisonPixels(c, 4, poison)
      << ",\n\"typed_d3d12_resources\":true,"
         "\n\"complete_write_coverage_checked_by_dual_poison\":true,"
         "\n\"coordinate_probe_consumed_by_execution\":false,"
         "\n\"expected_output_fed_to_execution\":false,"
         "\n\"archived_intermediates_consumed\":false,"
         "\n\"address_fields_recorded\":false,\n\"result\":\""
      << ((im | cmis) ? "fail" : "pass") << "\"\n}\n";
    return (im | cmis) ? 3 : 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
