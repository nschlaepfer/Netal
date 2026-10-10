/* Minimal no-CRT Wine process injector for the RE9-only bridge. */

typedef unsigned long DWORD;
typedef long LONG;
typedef int BOOL;
typedef unsigned short WCHAR;
typedef unsigned long long ULONG_PTR;
typedef unsigned long long SIZE_T;
typedef void *HANDLE;
typedef void *HMODULE;
typedef void *LPVOID;
typedef const WCHAR *LPCWSTR;
typedef const char *LPCSTR;

#define WINAPI __stdcall
#define DLLIMPORT __declspec(dllimport)
#define INVALID_HANDLE_VALUE ((HANDLE)(LONG)-1)
#define STD_OUTPUT_HANDLE ((DWORD)-11)
#define TH32CS_SNAPPROCESS 0x00000002
#define TH32CS_SNAPMODULE 0x00000008
#define PROCESS_CREATE_THREAD 0x0002
#define PROCESS_VM_OPERATION 0x0008
#define PROCESS_VM_READ 0x0010
#define PROCESS_VM_WRITE 0x0020
#define PROCESS_QUERY_INFORMATION 0x0400
#define MEM_COMMIT 0x1000
#define MEM_RESERVE 0x2000
#define MEM_RELEASE 0x8000
#define PAGE_READWRITE 0x04
#define INFINITE 0xffffffff

typedef DWORD(WINAPI *THREAD_START_ROUTINE)(LPVOID);

typedef struct process_entry32w
{
    DWORD size;
    DWORD usage;
    DWORD process_id;
    ULONG_PTR default_heap_id;
    DWORD module_id;
    DWORD thread_count;
    DWORD parent_process_id;
    LONG base_priority;
    DWORD flags;
    WCHAR executable[260];
} PROCESSENTRY32W;

DLLIMPORT HANDLE WINAPI CreateToolhelp32Snapshot(DWORD, DWORD);
DLLIMPORT BOOL WINAPI Process32FirstW(HANDLE, PROCESSENTRY32W *);
DLLIMPORT BOOL WINAPI Process32NextW(HANDLE, PROCESSENTRY32W *);
typedef struct module_entry32w {
    DWORD size, module_id, process_id, global_usage, process_usage;
    unsigned char *base;
    DWORD image_size;
    HMODULE module;
    WCHAR name[256], path[260];
} MODULEENTRY32W;
DLLIMPORT BOOL WINAPI Module32FirstW(HANDLE, MODULEENTRY32W *);
DLLIMPORT BOOL WINAPI Module32NextW(HANDLE, MODULEENTRY32W *);
DLLIMPORT BOOL WINAPI ReadProcessMemory(HANDLE, const void *, void *, SIZE_T, SIZE_T *);
DLLIMPORT HANDLE WINAPI OpenProcess(DWORD, BOOL, DWORD);
DLLIMPORT LPVOID WINAPI VirtualAllocEx(HANDLE, LPVOID, SIZE_T, DWORD, DWORD);
DLLIMPORT BOOL WINAPI VirtualFreeEx(HANDLE, LPVOID, SIZE_T, DWORD);
DLLIMPORT BOOL WINAPI WriteProcessMemory(HANDLE, LPVOID, const void *, SIZE_T, SIZE_T *);
DLLIMPORT HMODULE WINAPI GetModuleHandleW(LPCWSTR);
DLLIMPORT LPVOID WINAPI GetProcAddress(HMODULE, LPCSTR);
DLLIMPORT HANDLE WINAPI CreateRemoteThread(HANDLE, LPVOID, SIZE_T, THREAD_START_ROUTINE, LPVOID,
                                            DWORD, DWORD *);
DLLIMPORT DWORD WINAPI WaitForSingleObject(HANDLE, DWORD);
DLLIMPORT BOOL WINAPI GetExitCodeThread(HANDLE, DWORD *);
DLLIMPORT BOOL WINAPI CloseHandle(HANDLE);
DLLIMPORT DWORD WINAPI GetLastError(void);
DLLIMPORT HANDLE WINAPI GetStdHandle(DWORD);
DLLIMPORT BOOL WINAPI WriteFile(HANDLE, const void *, DWORD, DWORD *, LPVOID);
DLLIMPORT void WINAPI ExitProcess(DWORD);
DLLIMPORT WCHAR * WINAPI GetCommandLineW(void);

/* Arguments: [bare executable name [absolute Windows hook DLL path]].
 * The path enables portable loading when the target's startup WINEDLLPATH includes the
 * helper directory. Omitting it preserves the historical builtin-name route.
 * Comparison is ASCII case-folded; multiple matching processes are rejected. */
static int query_mode; /* 0=start, 1=identify native process, 2=find Windows process */
static int ambiguous_target;
static WCHAR target_name[260] = L"re9.exe";
static WCHAR hook_name[4096] = L"netal_wine_host_v1.dll";

static int equal_ascii_folded(const WCHAR *left, const WCHAR *right)
{
    unsigned int i = 0;
    for (;; i++)
    {
        WCHAR a = left[i];
        WCHAR b = right[i];
        if (a >= 'A' && a <= 'Z') a += 'a' - 'A';
        if (b >= 'A' && b <= 'Z') b += 'a' - 'A';
        if (a != b) return 0;
        if (!a) return 1;
    }
}

static DWORD wide_bytes(const WCHAR *text)
{
    DWORD count = 0;
    while (text[count]) count++;
    return (count + 1) * (DWORD)sizeof(WCHAR);
}

static DWORD ascii_bytes(const char *text)
{
    DWORD count = 0;
    while (text[count]) count++;
    return count;
}

static void report(const char *text)
{
    DWORD written = 0;
    WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), text, ascii_bytes(text), &written, 0);
}

static DWORD find_target(void)
{
    PROCESSENTRY32W entry;
    HANDLE snapshot;
    DWORD i;
    DWORD found = 0;

    for (i = 0; i < sizeof(entry); i++) ((unsigned char *)&entry)[i] = 0;
    entry.size = sizeof(entry);
    snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snapshot == INVALID_HANDLE_VALUE) return 0;
    if (Process32FirstW(snapshot, &entry))
    {
        do
        {
            if (equal_ascii_folded(entry.executable, target_name))
            {
                if (found) { ambiguous_target = 1; CloseHandle(snapshot); return 0; }
                found = entry.process_id;
            }
        } while (Process32NextW(snapshot, &entry));
    }
    CloseHandle(snapshot);
    return found;
}

/* Arguments are bare executable names or file paths, which cannot contain literal quotes.
 * Accept surrounding quotes for spaces, reject truncation/unbalanced quotes/extra arguments. */
static int next_argument(const WCHAR **input, WCHAR *output, unsigned int capacity)
{
    const WCHAR *cursor = *input;
    unsigned int length = 0;
    int quoted;
    while (*cursor == L' ' || *cursor == L'\t') cursor++;
    if (!*cursor) { *input = cursor; return 0; }
    quoted = *cursor == L'"';
    if (quoted) cursor++;
    while (*cursor && (quoted ? *cursor != L'"' : (*cursor != L' ' && *cursor != L'\t'))) {
        if (*cursor == L'"' || *cursor < 32 || length + 1 >= capacity) return -1;
        if (output) output[length] = *cursor;
        length++; cursor++;
    }
    if (quoted) {
        if (*cursor != L'"') return -1;
        cursor++;
        if (*cursor && *cursor != L' ' && *cursor != L'\t') return -1;
    }
    if (!length) return -1;
    if (output) output[length] = 0;
    *input = cursor;
    return 1;
}

static int target_from_command_line(void)
{
    const WCHAR *cursor = GetCommandLineW();
    int status;
    unsigned int i;
    if (!cursor || next_argument(&cursor, 0, 32768) != 1) return 0;
    status = next_argument(&cursor, target_name, 260);
    if (status < 0) return 0;
    if (!status) return 1; /* preserve the historical default target */
    if (equal_ascii_folded(target_name, L"--query") || equal_ascii_folded(target_name, L"--find")) {
        query_mode = equal_ascii_folded(target_name, L"--query") ? 1 : 2;
        if (next_argument(&cursor, target_name, 260) != 1) return 0;
    }
    for (i = 0; target_name[i]; i++)
        if (target_name[i] == L'/' || target_name[i] == L'\\') return 0;
    if (query_mode == 2) return next_argument(&cursor, 0, 4096) == 0;
    status = next_argument(&cursor, hook_name, 4096);
    if (status < 0 || (query_mode == 1 && !status)) return 0;
    if (status) {
        if (!((hook_name[0] >= L'A' && hook_name[0] <= L'Z') ||
              (hook_name[0] >= L'a' && hook_name[0] <= L'z')) || hook_name[1] != L':' ||
            (hook_name[2] != L'/' && hook_name[2] != L'\\')) return 0;
    }
    return next_argument(&cursor, 0, 4096) == 0;
}


static void report_number(const char *key, DWORD value) {
    char buffer[12];
    unsigned int pos = sizeof(buffer)-1;
    buffer[pos] = 0;
    do { buffer[--pos] = (char)('0'+value%10); value /= 10; } while (value);
    report(key); report(buffer+pos); report("\r\n");
}

static int read_image(HANDLE process, const MODULEENTRY32W *module, DWORD offset, void *data, SIZE_T size) {
    SIZE_T read = 0;
    if (offset > module->image_size || size > module->image_size-offset) return 0;
    return ReadProcessMemory(process, module->base+offset, data, size, &read) && read == size;
}

/* Resolve an export in the target's mapped PE image. Never assume a locally rebuilt
 * DLL has the same function offsets as a module already loaded in a long-lived process. */
static THREAD_START_ROUTINE remote_export(HANDLE process, DWORD pid, const char *name) {
    MODULEENTRY32W module;
    HANDLE snapshot;
    const WCHAR *basename = hook_name;
    DWORD i, nt, signature, export_info[2], directory[10];
    unsigned short magic;
    for (i=0; hook_name[i]; i++) if (hook_name[i] == L'/' || hook_name[i] == L'\\') basename=hook_name+i+1;
    for (i=0; i<sizeof(module); i++) ((unsigned char*)&module)[i]=0;
    module.size=sizeof(module);
    snapshot=CreateToolhelp32Snapshot(TH32CS_SNAPMODULE,pid);
    if (snapshot==INVALID_HANDLE_VALUE) return 0;
    if (!Module32FirstW(snapshot,&module)) { CloseHandle(snapshot); return 0; }
    while (!equal_ascii_folded(module.name,basename)) {
        if (!Module32NextW(snapshot,&module)) { CloseHandle(snapshot); return 0; }
    }
    CloseHandle(snapshot);
    if (!read_image(process,&module,0x3c,&nt,sizeof(nt)) || nt>module.image_size || module.image_size-nt<144 ||
        !read_image(process,&module,nt,&signature,sizeof(signature)) || signature!=0x4550 ||
        !read_image(process,&module,nt+24,&magic,sizeof(magic)) || magic!=0x20b ||
        !read_image(process,&module,nt+136,export_info,sizeof(export_info)) || export_info[1]<40 ||
        export_info[0]>module.image_size || export_info[1]>module.image_size-export_info[0] ||
        !read_image(process,&module,export_info[0],directory,sizeof(directory)) ||
        directory[5]>4096 || directory[6]>4096 ||
        (SIZE_T)directory[7]+(SIZE_T)directory[5]*4>module.image_size ||
        (SIZE_T)directory[8]+(SIZE_T)directory[6]*4>module.image_size ||
        (SIZE_T)directory[9]+(SIZE_T)directory[6]*2>module.image_size) return 0;
    for (i=0; i<directory[6]; i++) {
        DWORD name_rva, function_rva, j, size=ascii_bytes(name)+1;
        unsigned short ordinal;
        char candidate[64];
        if (size>sizeof(candidate) || directory[8]>module.image_size || directory[9]>module.image_size ||
            !read_image(process,&module,directory[8]+i*4,&name_rva,sizeof(name_rva)) ||
            !read_image(process,&module,name_rva,candidate,size)) return 0;
        for (j=0; j<size && candidate[j]==name[j]; j++) {}
        if (j!=size) continue;
        if (!read_image(process,&module,directory[9]+i*2,&ordinal,sizeof(ordinal)) || ordinal>=directory[5] ||
            directory[7]>module.image_size ||
            !read_image(process,&module,directory[7]+ordinal*4,&function_rva,sizeof(function_rva)) ||
            !function_rva || function_rva>=module.image_size ||
            (function_rva>=export_info[0] && function_rva-export_info[0]<export_info[1])) return 0;
        return (THREAD_START_ROUTINE)(module.base+function_rva);
    }
    return 0;
}

static int remote_call(HANDLE process, THREAD_START_ROUTINE function, LPVOID argument, DWORD *result) {
    HANDLE thread=CreateRemoteThread(process,0,0,function,argument,0,0);
    int completed;
    if (!thread) return -1;
    completed=WaitForSingleObject(thread,30000)==0 && GetExitCodeThread(thread,result);
    CloseHandle(thread);
    return completed;
}

void mainCRTStartup(void) {
    const DWORD access=PROCESS_CREATE_THREAD|PROCESS_QUERY_INFORMATION|PROCESS_VM_OPERATION|PROCESS_VM_WRITE|PROCESS_VM_READ;
    THREAD_START_ROUTINE function;
    HANDLE process;
    LPVOID remote_name;
    DWORD pid, code=0, name_bytes;
    int completed;
    if (!target_from_command_line()) { report("event=inject_failed reason=invalid_arguments\r\n"); ExitProcess(64); }
    pid=find_target();
    if (!pid) {
        report(ambiguous_target ? "event=inject_failed reason=target_ambiguous\r\n" : "event=inject_failed reason=target_not_found\r\n");
        ExitProcess(ambiguous_target ? 9 : 2);
    }
    if (query_mode==2) { report_number("windows_pid=",pid); ExitProcess(0); }
    process=OpenProcess(access,0,pid);
    if (!process) { report("event=inject_failed reason=open_process\r\n"); ExitProcess(3); }
    name_bytes=wide_bytes(hook_name);
    remote_name=VirtualAllocEx(process,0,name_bytes,MEM_COMMIT|MEM_RESERVE,PAGE_READWRITE);
    if (!remote_name || !WriteProcessMemory(process,remote_name,hook_name,name_bytes,0)) {
        report("event=inject_failed reason=remote_write\r\n");
        if (remote_name) VirtualFreeEx(process,remote_name,0,MEM_RELEASE);
        CloseHandle(process); ExitProcess(4);
    }
    function=(THREAD_START_ROUTINE)GetProcAddress(GetModuleHandleW(L"kernel32.dll"),"LoadLibraryW");
    if (!function) { VirtualFreeEx(process,remote_name,0,MEM_RELEASE); CloseHandle(process); ExitProcess(5); }
    completed=remote_call(process,function,remote_name,&code);
    /* A timed-out thread may still be reading the string. Keep that allocation until process exit. */
    if (completed!=0) VirtualFreeEx(process,remote_name,0,MEM_RELEASE);
    if (completed!=1 || !code) { report("event=inject_failed reason=remote_loadlibrary\r\n"); CloseHandle(process); ExitProcess(7); }
    function=remote_export(process,pid,query_mode==1 ? "NetalMacPidV1" : "NetalStartBridgeV1");
    if (!function) { report("event=inject_failed reason=helper_protocol_missing\r\n"); CloseHandle(process); ExitProcess(8); }
    completed=remote_call(process,function,0,&code);
    CloseHandle(process);
    if (completed!=1 || (query_mode==1 ? (code==0 || code>0x7fffffff) : code!=0)) {
        report("event=inject_failed reason=native_helper\r\n"); ExitProcess(8);
    }
    if (query_mode==1) { report_number("windows_pid=",pid); report_number("mac_pid=",code); }
    else report("event=inject_loaded module=netal_wine_host_v1.dll native_started=1\r\n");
    ExitProcess(0);
}
