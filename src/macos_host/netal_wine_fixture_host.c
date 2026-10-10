/* Test-only Windows host: no game, graphics, input hooks or model execution. */
typedef unsigned long DWORD;
typedef void *HANDLE;
#define IMPORT __declspec(dllimport)
IMPORT void __stdcall Sleep(DWORD);
IMPORT HANDLE __stdcall GetStdHandle(DWORD);
IMPORT int __stdcall WriteFile(HANDLE, const void *, DWORD, DWORD *, void *);
IMPORT void __stdcall ExitProcess(DWORD);
void mainCRTStartup(void) {
    const char message[] = "event=fixture_host_ready\r\n";
    DWORD written;
    WriteFile(GetStdHandle((DWORD)-11), message, sizeof(message)-1, &written, 0);
    Sleep(20000);
    ExitProcess(0);
}
