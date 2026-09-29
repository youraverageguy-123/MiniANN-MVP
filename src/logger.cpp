#include "miniann/logger.hpp"
#include <iostream>

namespace miniann {

void ConsoleLogger::log(LogLevel level, const std::string& message) {
    const char* tag = "INFO";
    switch (level) {
        case LogLevel::Debug: tag = "DEBUG"; break;
        case LogLevel::Info: tag = "INFO"; break;
        case LogLevel::Warn: tag = "WARN"; break;
        case LogLevel::Error: tag = "ERROR"; break;
    }
    std::cout << "[" << tag << "] " << message << "\n";
}

} // namespace miniann
