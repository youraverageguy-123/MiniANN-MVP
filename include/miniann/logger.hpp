#pragma once
#include "miniann/types.hpp"
#include <fstream>
#include <string>

namespace miniann {

enum class LogLevel { Debug, Info, Warn, Error };

class ILogger {
public:
    virtual ~ILogger() = default;
    virtual void log(LogLevel level, const std::string& message) = 0;
};

class ConsoleLogger : public ILogger {
public:
    void log(LogLevel level, const std::string& message) override;
};

// Writes log lines to a text file instead of the screen, so a training run
// leaves a permanent record. Messages below minLevel are ignored.
// Throws std::runtime_error if the file cannot be opened.
class FileLogger : public ILogger {
public:
    explicit FileLogger(const std::string& path,
                        LogLevel minLevel = LogLevel::Info,
                        bool append = false);
    void log(LogLevel level, const std::string& message) override;
    const std::string& path() const { return path_; }
    LogLevel minLevel() const { return minLevel_; }

private:
    std::string path_;
    LogLevel minLevel_;
    std::ofstream out_;
};

} // namespace miniann
