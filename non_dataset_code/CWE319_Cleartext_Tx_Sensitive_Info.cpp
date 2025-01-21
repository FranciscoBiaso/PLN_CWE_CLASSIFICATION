#include <iostream>
#include <cstring>
#include <string>
#ifdef _WIN32
    #include <winsock2.h>
    #pragma comment(lib, "ws2_32.lib")
    typedef int socklen_t;
#else
    #include <sys/socket.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #define INVALID_SOCKET -1
    #define SOCKET_ERROR -1
    typedef int SOCKET;
#endif

#define PORT 8080
#define BUFFER_SIZE 1024

int main() {
#ifdef _WIN32
    // Initialize Winsock
    WSADATA wsaData;
    int wsaInit = WSAStartup(MAKEWORD(2,2), &wsaData);
    if (wsaInit != 0) {
        std::cerr << "WSAStartup failed: " << wsaInit << std::endl;
        return 1;
    }
#endif

    // Create a socket
    SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == INVALID_SOCKET) {
        std::cerr << "Failed to create socket." << std::endl;
        return 1;
    }

    // Server address setup
    struct sockaddr_in serverAddr;
    std::memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(PORT);
    // Connect to localhost for demonstration purposes
    serverAddr.sin_addr.s_addr = inet_addr("127.0.0.1");

    // Connect to the server
    if (connect(sock, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
        std::cerr << "Connection to server failed." << std::endl;
#ifdef _WIN32
        closesocket(sock);
        WSACleanup();
#else
        close(sock);
#endif
        return 1;
    }

    // Get sensitive information from the user
    std::string username, password;
    std::cout << "Enter username: ";
    std::getline(std::cin, username);
    std::cout << "Enter password: ";
    std::getline(std::cin, password);

    // Prepare the message (in cleartext)
    std::string message = "Username: " + username + "\nPassword: " + password;

    // Send the sensitive information over the network without encryption
    ssize_t sentBytes = send(sock, message.c_str(), message.length(), 0);
    if (sentBytes == SOCKET_ERROR) {
        std::cerr << "Failed to send data." << std::endl;
    } else {
        std::cout << "Credentials sent successfully in cleartext." << std::endl;
    }

    // Close the socket
#ifdef _WIN32
    closesocket(sock);
    WSACleanup();
#else
    close(sock);
#endif

    return 0;
}
