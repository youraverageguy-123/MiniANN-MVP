#pragma once
#include "miniann/types.hpp"
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

} // namespace miniann
