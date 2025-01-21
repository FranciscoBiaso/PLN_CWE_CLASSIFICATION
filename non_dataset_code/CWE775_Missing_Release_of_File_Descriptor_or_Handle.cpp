#include <iostream>
#include <fstream>

void readFile() {
    std::ifstream file("example.txt");

    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            std::cout << line << std::endl;
        }
        // Missing file.close() here
    } else {
        std::cerr << "Failed to open file." << std::endl;
    }
}

int main() {
    readFile();
    return 0;
}
