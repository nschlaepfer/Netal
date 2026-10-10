/* PE entry points for netal_wine_host_v1.dll/.so.
 * Keep the native table ABI stable within this filename version; use a new name for breaking changes.
 * The v1 name prevents pairing this two-entry protocol with historical one-entry helpers. */
typedef unsigned long DWORD;
typedef long NTSTATUS;
typedef int BOOL;
typedef unsigned long long SIZE_T;
typedef unsigned long long unixlib_handle_t;
typedef void *HANDLE;
typedef void *HMODULE;
typedef void *LPVOID;
#define WINAPI __stdcall
#define DLLIMPORT __declspec(dllimport)
#define DLLEXPORT __declspec(dllexport)
#define DLL_PROCESS_ATTACH 1
#define MemoryWineUnixFuncs 1000
DLLIMPORT BOOL WINAPI DisableThreadLibraryCalls(HMODULE);
DLLIMPORT NTSTATUS WINAPI NtQueryVirtualMemory(HANDLE, const void *, int, void *, SIZE_T, SIZE_T *);
DLLIMPORT NTSTATUS WINAPI __wine_unix_call(unixlib_handle_t, unsigned int, void *);
static HMODULE self_module;
static NTSTATUS call_native(unsigned int entry) {
    unixlib_handle_t handle = 0;
    NTSTATUS status = NtQueryVirtualMemory((HANDLE)-1, self_module, MemoryWineUnixFuncs,
                                          &handle, sizeof(handle), 0);
    if (status || !handle) return (NTSTATUS)0xc0000135;
    return __wine_unix_call(handle, entry, 0);
}
/* Discovery does not read launch configuration, arm features, or load the Metal bridge. */
DLLEXPORT DWORD WINAPI NetalMacPidV1(LPVOID ignored) {
    NTSTATUS value;
    (void)ignored;
    value = call_native(1);
    return value > 0 ? (DWORD)value : 0;
}
DLLEXPORT DWORD WINAPI NetalStartBridgeV1(LPVOID ignored) {
    (void)ignored;
    return (DWORD)call_native(0);
}
BOOL WINAPI DllMain(HMODULE module, DWORD reason, LPVOID reserved) {
    (void)reserved;
    if (reason == DLL_PROCESS_ATTACH) {
        self_module = module;
        DisableThreadLibraryCalls(module);
    }
    return 1;
}
