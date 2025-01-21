#include <iostream>

// Function that returns a pointer to an integer
int* getNumber(bool returnNull) {
    if (returnNull) {
        return nullptr; // Explicitly return NULL
    } else {
        int* num = new int(42); // Dynamically allocate an integer
        return num;
    }
}

int main() {
    bool condition = true; // Change to false to avoid NULL dereference

    int* number = getNumber(condition);

    // Attempt to dereference the pointer without checking if it's NULL
    std::cout << "The number is: " << *number << std::endl;

    // Clean up memory if not NULL
    if (number != nullptr) {
        delete number;
    }

    return 0;
}
