#include "miniann/logger.hpp"
#include <iostream>
#include <stdexcept>

namespace miniann {

// Shared by both loggers so the "[INFO]" style tags stay identical.
static const char* levelTag(LogLevel level) {
    switch (level) {
        case LogLevel::Debug: return "DEBUG";
        case LogLevel::Info:  return "INFO";
        case LogLevel::Warn:  return "WARN";
        case LogLevel::Error: return "ERROR";
    }
    return "INFO";
}

void ConsoleLogger::log(LogLevel level, const std::string& message) {
    std::cout << "[" << levelTag(level) << "] " << message << "\n";
}

FileLogger::FileLogger(const std::string& path, LogLevel minLevel, bool append)
    : path_(path), minLevel_(minLevel),
      out_(path, append ? std::ios::app : std::ios::trunc) {
    if (!out_) throw std::runtime_error("FileLogger: cannot open " + path);
}

void FileLogger::log(LogLevel level, const std::string& message) {
    if (static_cast<int>(level) < static_cast<int>(minLevel_)) return; // filtered out
    out_ << "[" << levelTag(level) << "] " << message << "\n";
    out_.flush(); // keep the file useful even if the program crashes later
}

} // namespace miniann
