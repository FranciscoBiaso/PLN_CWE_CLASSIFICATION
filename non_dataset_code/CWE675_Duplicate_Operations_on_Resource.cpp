#include <iostream>
#include <fstream>
#include <string>

int main() {
    std::string filename = "example.txt";

    // Open the file for writing
    std::ofstream file;
    file.open(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open the file: " << filename << std::endl;
        return 1;
    }

    // Write some data to the file
    file << "This is a sample line of text." << std::endl;

    // Close the file for the first time
    file.close();
    std::cout << "File closed successfully the first time." << std::endl;

    // Attempt to close the file a second time (Duplicate Operation)
    file.close();
    std::cout << "File closed successfully the second time." << std::endl;

    return 0;
}
