#include <iostream>
#include <fstream>

void readFile() {
    // RAII ensures the file will be automatically closed when it goes out of scope
    std::ifstream file("example.txt");

    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            std::cout << line << std::endl;
        }
    } else {
        std::cerr << "Failed to open file." << std::endl;
    }
}

int main() {
    readFile();
    return 0;
}
