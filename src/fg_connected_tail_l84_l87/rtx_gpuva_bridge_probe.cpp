#include <windows.h>
#include <cuda.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <cstdint>
#include <iostream>
#include <stdexcept>
using Microsoft::WRL::ComPtr;
void Hr(HRESULT h,const char*n){if(FAILED(h))throw std::runtime_error(n);}
void Cu(CUresult r,const char*n){if(r!=CUDA_SUCCESS)throw std::runtime_error(n);}
int main(){try{
  ComPtr<IDXGIFactory6>factory;Hr(CreateDXGIFactory2(0,IID_PPV_ARGS(&factory)),"factory");
  ComPtr<IDXGIAdapter1>adapter;for(UINT i=0;;++i){ComPtr<IDXGIAdapter1>a;if(factory->EnumAdapterByGpuPreference(i,DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,IID_PPV_ARGS(&a))==DXGI_ERROR_NOT_FOUND)break;DXGI_ADAPTER_DESC1 d{};a->GetDesc1(&d);if(!(d.Flags&DXGI_ADAPTER_FLAG_SOFTWARE)&&SUCCEEDED(D3D12CreateDevice(a.Get(),D3D_FEATURE_LEVEL_12_0,__uuidof(ID3D12Device),nullptr))){adapter=a;break;}}if(!adapter)throw std::runtime_error("adapter");
  ComPtr<ID3D12Device>device;Hr(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&device)),"device");
  D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_DEFAULT;D3D12_RESOURCE_DESC desc{};desc.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;desc.Width=4096;desc.Height=1;desc.DepthOrArraySize=1;desc.MipLevels=1;desc.SampleDesc.Count=1;desc.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;ComPtr<ID3D12Resource>buffer;Hr(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&buffer)),"buffer");
  Cu(cuInit(0),"cuInit");CUdevice cd;Cu(cuDeviceGet(&cd,0),"cuDeviceGet");CUcontext ctx;Cu(cuDevicePrimaryCtxRetain(&ctx,cd),"retain");Cu(cuCtxSetCurrent(ctx),"set");
  const char*ptx=".version 8.0\n.target sm_75\n.address_size 64\n.visible .entry k(.param .u64 p){.reg .u64 %rd;.reg .u32 %r;ld.param.u64 %rd,[p];mov.u32 %r,0x1234abcd;st.global.u32 [%rd],%r;ret;}\n";CUmodule mod;Cu(cuModuleLoadData(&mod,ptx),"module");CUfunction fn;Cu(cuModuleGetFunction(&fn,mod,"k"),"function");CUdeviceptr p=buffer->GetGPUVirtualAddress();void*args[]={&p};Cu(cuLaunchKernel(fn,1,1,1,1,1,1,0,nullptr,args,nullptr),"launch");Cu(cuCtxSynchronize(),"sync");
  D3D12_HEAP_PROPERTIES rh{};rh.Type=D3D12_HEAP_TYPE_READBACK;D3D12_RESOURCE_DESC rd=desc;rd.Flags=D3D12_RESOURCE_FLAG_NONE;ComPtr<ID3D12Resource>read;Hr(device->CreateCommittedResource(&rh,D3D12_HEAP_FLAG_NONE,&rd,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&read)),"read");D3D12_COMMAND_QUEUE_DESC qd{};ComPtr<ID3D12CommandQueue>q;Hr(device->CreateCommandQueue(&qd,IID_PPV_ARGS(&q)),"queue");ComPtr<ID3D12CommandAllocator>al;Hr(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&al)),"allocator");ComPtr<ID3D12GraphicsCommandList>cl;Hr(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,al.Get(),nullptr,IID_PPV_ARGS(&cl)),"list");D3D12_RESOURCE_BARRIER bar{};bar.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;bar.Transition.pResource=buffer.Get();bar.Transition.StateBefore=D3D12_RESOURCE_STATE_COMMON;bar.Transition.StateAfter=D3D12_RESOURCE_STATE_COPY_SOURCE;bar.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;cl->ResourceBarrier(1,&bar);cl->CopyResource(read.Get(),buffer.Get());Hr(cl->Close(),"close");ID3D12CommandList*lists[]={cl.Get()};q->ExecuteCommandLists(1,lists);ComPtr<ID3D12Fence>f;Hr(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&f)),"fence");HANDLE event=CreateEvent(nullptr,FALSE,FALSE,nullptr);Hr(q->Signal(f.Get(),1),"signal");Hr(f->SetEventOnCompletion(1,event),"event");WaitForSingleObject(event,INFINITE);CloseHandle(event);void*m=nullptr;D3D12_RANGE range{0,4};Hr(read->Map(0,&range,&m),"map");std::uint32_t value=*static_cast<std::uint32_t*>(m);read->Unmap(0,nullptr);std::cout<<"gpu_va_cuda_write=0x"<<std::hex<<value<<"\nresult="<<(value==0x1234abcdu?"pass":"fail")<<"\n";return value==0x1234abcdu?0:2;
}catch(const std::exception&e){std::cerr<<"error="<<e.what()<<"\n";return 1;}}
