#pragma once

#include <windows.h>

#include <nvsdk_ngx.h>
#include <nvsdk_ngx_params.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <unordered_map>

namespace neuralstudio::fg {

// The signed feature snippet consumes NVSDK_NGX_Parameter through its MSVC
// overload-sorted vtable. Keep the layout explicit, as the SR oracle does, so
// this host remains ABI-stable if it is later cross-compiled with MinGW.
class ParameterStore final {
 public:
  ParameterStore() : vtable_(MsvcVtable()) {}

  NVSDK_NGX_Parameter* AsNgx() {
    return reinterpret_cast<NVSDK_NGX_Parameter*>(this);
  }
  const NVSDK_NGX_Parameter* AsNgx() const {
    return reinterpret_cast<const NVSDK_NGX_Parameter*>(this);
  }

  void SetU64(const char* name, unsigned long long value) {
    SetUnsignedLongLong(this, name, value);
  }
  void SetU32(const char* name, unsigned int value) {
    SetUnsigned(this, name, value);
  }
  void SetI32(const char* name, int value) { SetInt(this, name, value); }
  void SetF32(const char* name, float value) { SetFloat(this, name, value); }
  void SetPointerValue(const char* name, void* value) {
    SetPointer(this, name, value);
  }
  void SetResource(const char* name, ID3D12Resource* value) {
    SetD3D12(this, name, value);
  }

  NVSDK_NGX_Result GetU64(const char* name,
                          unsigned long long* output) const {
    return GetUnsignedLongLong(const_cast<ParameterStore*>(this), name,
                               output);
  }
  NVSDK_NGX_Result GetU32(const char* name, unsigned int* output) const {
    return GetUnsigned(const_cast<ParameterStore*>(this), name, output);
  }
  NVSDK_NGX_Result GetI32(const char* name, int* output) const {
    return GetInt(const_cast<ParameterStore*>(this), name, output);
  }
  NVSDK_NGX_Result GetF32(const char* name, float* output) const {
    return GetFloat(const_cast<ParameterStore*>(this), name, output);
  }
  NVSDK_NGX_Result GetPointerValue(const char* name, void** output) const {
    return GetPointer(const_cast<ParameterStore*>(this), name, output);
  }

 private:
  using Slot = void*;

  template <typename Function>
  static Slot SlotFor(Function function) {
    Slot slot = nullptr;
    static_assert(sizeof(slot) == sizeof(function));
    std::memcpy(&slot, &function, sizeof(slot));
    return slot;
  }

  static Slot* MsvcVtable() {
    static Slot slots[] = {
        SlotFor(&SetPointer),          SlotFor(&SetD3D12),
        SlotFor(&SetD3D11),           SlotFor(&SetInt),
        SlotFor(&SetUnsigned),        SlotFor(&SetDouble),
        SlotFor(&SetFloat),           SlotFor(&SetUnsignedLongLong),
        SlotFor(&GetPointer),         SlotFor(&GetD3D12),
        SlotFor(&GetD3D11),           SlotFor(&GetInt),
        SlotFor(&GetUnsigned),        SlotFor(&GetDouble),
        SlotFor(&GetFloat),           SlotFor(&GetUnsignedLongLong),
        SlotFor(&Reset),
    };
    return slots;
  }

  static void WINAPI SetPointer(ParameterStore* self, const char* name,
                                void* value) {
    self->pointers_[SafeName(name)] = value;
    LogSetValue(name, "pointer", value);
  }
  static void WINAPI SetD3D12(ParameterStore* self, const char* name,
                              ID3D12Resource* value) {
    self->d3d12_[SafeName(name)] = value;
    LogSetValue(name, "d3d12", value);
  }
  static void WINAPI SetD3D11(ParameterStore* self, const char* name,
                              ID3D11Resource* value) {
    self->d3d11_[SafeName(name)] = value;
    LogSetValue(name, "d3d11", value);
  }
  static void WINAPI SetInt(ParameterStore* self, const char* name, int value) {
    self->i32_[SafeName(name)] = value;
    LogSetValue(name, "i32", value);
  }
  static void WINAPI SetUnsigned(ParameterStore* self, const char* name,
                                 unsigned int value) {
    self->u32_[SafeName(name)] = value;
    LogSetValue(name, "u32", value);
  }
  static void WINAPI SetDouble(ParameterStore* self, const char* name,
                               double value) {
    self->f64_[SafeName(name)] = value;
    LogSetValue(name, "f64", value);
  }
  static void WINAPI SetFloat(ParameterStore* self, const char* name,
                              float value) {
    self->f32_[SafeName(name)] = value;
    LogSetValue(name, "f32", value);
  }
  static void WINAPI SetUnsignedLongLong(ParameterStore* self,
                                         const char* name,
                                         unsigned long long value) {
    self->u64_[SafeName(name)] = value;
    LogSetValue(name, "u64", value);
  }

  static NVSDK_NGX_Result WINAPI GetPointer(ParameterStore* self,
                                             const char* name, void** output) {
    if (name == nullptr || output == nullptr) {
      return NVSDK_NGX_Result_FAIL_InvalidParameter;
    }
    const auto pointer = self->pointers_.find(name);
    if (pointer != self->pointers_.end()) {
      *output = pointer->second;
      std::cout << "parameter.get name=\"" << name
                << "\" type=pointer hit=1 source=pointer ";
      PrintValue(*output);
      std::cout << '\n';
      return NVSDK_NGX_Result_Success;
    }
    const auto resource = self->d3d12_.find(name);
    if (resource != self->d3d12_.end()) {
      *output = static_cast<void*>(resource->second);
      std::cout << "parameter.get name=\"" << name
                << "\" type=pointer hit=1 source=d3d12_cross_type ";
      PrintValue(*output);
      std::cout << '\n';
      return NVSDK_NGX_Result_Success;
    }
    *output = nullptr;
    std::cout << "parameter.get name=\"" << name
              << "\" type=pointer hit=0 source=none value=null\n";
    return NVSDK_NGX_Result_FAIL_MissingInput;
  }
  static NVSDK_NGX_Result WINAPI GetD3D12(ParameterStore* self,
                                           const char* name,
                                           ID3D12Resource** output) {
    return GetValue(self->d3d12_, name, output, "d3d12");
  }
  static NVSDK_NGX_Result WINAPI GetD3D11(ParameterStore* self,
                                           const char* name,
                                           ID3D11Resource** output) {
    return GetValue(self->d3d11_, name, output, "d3d11");
  }
  static NVSDK_NGX_Result WINAPI GetInt(ParameterStore* self,
                                         const char* name, int* output) {
    if (name == nullptr || output == nullptr) {
      return NVSDK_NGX_Result_FAIL_InvalidParameter;
    }
    const auto signed_value = self->i32_.find(name);
    if (signed_value != self->i32_.end()) {
      *output = signed_value->second;
      std::cout << "parameter.get name=\"" << name
                << "\" type=i32 hit=1 source=i32 value=" << *output << '\n';
      return NVSDK_NGX_Result_Success;
    }
    const auto unsigned_value = self->u32_.find(name);
    if (unsigned_value != self->u32_.end() &&
        unsigned_value->second <=
            static_cast<unsigned int>((std::numeric_limits<int>::max)())) {
      *output = static_cast<int>(unsigned_value->second);
      std::cout << "parameter.get name=\"" << name
                << "\" type=i32 hit=1 source=u32_cross_type value=" << *output
                << '\n';
      return NVSDK_NGX_Result_Success;
    }
    *output = 0;
    std::cout << "parameter.get name=\"" << name
              << "\" type=i32 hit=0 source="
              << (unsigned_value != self->u32_.end() ? "u32_out_of_range"
                                                     : "none")
              << " value=0\n";
    return NVSDK_NGX_Result_FAIL_MissingInput;
  }
  static NVSDK_NGX_Result WINAPI GetUnsigned(ParameterStore* self,
                                              const char* name,
                                              unsigned int* output) {
    if (name == nullptr || output == nullptr) {
      return NVSDK_NGX_Result_FAIL_InvalidParameter;
    }
    const auto unsigned_value = self->u32_.find(name);
    if (unsigned_value != self->u32_.end()) {
      *output = unsigned_value->second;
      std::cout << "parameter.get name=\"" << name
                << "\" type=u32 hit=1 source=u32 value=" << *output << '\n';
      return NVSDK_NGX_Result_Success;
    }
    const auto signed_value = self->i32_.find(name);
    if (signed_value != self->i32_.end() && signed_value->second >= 0) {
      *output = static_cast<unsigned int>(signed_value->second);
      std::cout << "parameter.get name=\"" << name
                << "\" type=u32 hit=1 source=i32_cross_type value=" << *output
                << '\n';
      return NVSDK_NGX_Result_Success;
    }
    *output = 0;
    std::cout << "parameter.get name=\"" << name
              << "\" type=u32 hit=0 source="
              << (signed_value != self->i32_.end() ? "i32_out_of_range"
                                                   : "none")
              << " value=0\n";
    return NVSDK_NGX_Result_FAIL_MissingInput;
  }
  static NVSDK_NGX_Result WINAPI GetDouble(ParameterStore* self,
                                            const char* name,
                                            double* output) {
    return GetValue(self->f64_, name, output, "f64");
  }
  static NVSDK_NGX_Result WINAPI GetFloat(ParameterStore* self,
                                           const char* name, float* output) {
    return GetValue(self->f32_, name, output, "f32");
  }
  static NVSDK_NGX_Result WINAPI GetUnsignedLongLong(
      ParameterStore* self, const char* name, unsigned long long* output) {
    return GetValue(self->u64_, name, output, "u64");
  }

  static void WINAPI Reset(ParameterStore* self) {
    self->u64_.clear();
    self->f32_.clear();
    self->f64_.clear();
    self->u32_.clear();
    self->i32_.clear();
    self->d3d11_.clear();
    self->d3d12_.clear();
    self->pointers_.clear();
  }

  static std::string SafeName(const char* name) {
    return name != nullptr ? name : "";
  }

  template <typename Map, typename Value>
  static NVSDK_NGX_Result GetValue(const Map& values, const char* name,
                                   Value* output, const char* type) {
    if (name == nullptr || output == nullptr) {
      return NVSDK_NGX_Result_FAIL_InvalidParameter;
    }
    const auto found = values.find(name);
    if (found == values.end()) {
      *output = Value{};
      std::cout << "parameter.get name=\"" << name << "\" type=" << type
                << " hit=0\n";
      return NVSDK_NGX_Result_FAIL_MissingInput;
    }
    *output = found->second;
    std::cout << "parameter.get name=\"" << name << "\" type=" << type
              << " hit=1 ";
    PrintValue(*output);
    std::cout << '\n';
    return NVSDK_NGX_Result_Success;
  }

  template <typename Value>
  static void LogSetValue(const char* name, const char* type, Value value) {
    std::cout << "parameter.set name=\"" << SafeName(name) << "\" type="
              << type << ' ';
    PrintValue(value);
    std::cout << '\n';
  }

  template <typename Value>
  static void PrintValue(Value value) {
    if constexpr (std::is_pointer_v<Value>) {
      std::cout << "value=" << (value != nullptr ? "present" : "null");
    } else if constexpr (std::is_same_v<Value, float>) {
      std::uint32_t bits = 0;
      std::memcpy(&bits, &value, sizeof(bits));
      std::cout << "value_bits=0x" << std::hex << bits << std::dec;
    } else if constexpr (std::is_same_v<Value, double>) {
      std::uint64_t bits = 0;
      std::memcpy(&bits, &value, sizeof(bits));
      std::cout << "value_bits=0x" << std::hex << bits << std::dec;
    } else {
      std::cout << "value=" << value;
    }
  }

  Slot* vtable_;
  std::unordered_map<std::string, unsigned long long> u64_;
  std::unordered_map<std::string, float> f32_;
  std::unordered_map<std::string, double> f64_;
  std::unordered_map<std::string, unsigned int> u32_;
  std::unordered_map<std::string, int> i32_;
  std::unordered_map<std::string, ID3D11Resource*> d3d11_;
  std::unordered_map<std::string, ID3D12Resource*> d3d12_;
  std::unordered_map<std::string, void*> pointers_;
};

}  // namespace neuralstudio::fg
