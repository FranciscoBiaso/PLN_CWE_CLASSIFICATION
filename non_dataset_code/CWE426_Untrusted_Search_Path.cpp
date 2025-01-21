// VulnerableApp.cpp
#include <windows.h>
#include <iostream>

typedef void (*FunctionType)();

int main() {
    // Attempt to load a DLL without specifying the full path
    HMODULE hModule = LoadLibraryA("MyLibrary.dll");
    if (hModule == NULL) {
        std::cerr << "Failed to load MyLibrary.dll. Error: " << GetLastError() << std::endl;
        return 1;
    }

    // Get the address of the function to call
    FunctionType func = (FunctionType)GetProcAddress(hModule, "MyFunction");
    if (func == NULL) {
        std::cerr << "Failed to locate MyFunction. Error: " << GetLastError() << std::endl;
        FreeLibrary(hModule);
        return 1;
    }

    // Call the function from the DLL
    func();
    std::cout << "Function executed successfully." << std::endl;

    // Free the DLL module
    FreeLibrary(hModule);
    return 0;
}
