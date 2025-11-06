#pragma once
#include <string>
#include <chrono>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <mutex>
#include <functional>

namespace utils {

    // ==============================
    // Logging
    // ==============================
    enum class LogLevel {
        INFO,
        WARN,
        ERROR
    };

    class Logger {
    public:
        static Logger& instance();

        void log(LogLevel level, const std::string& msg);
        void info(const std::string& msg);
        void warn(const std::string& msg);
        void error(const std::string& msg);

        void setLogFile(const std::string& filepath);
        void closeLogFile();
        void enableConsole(bool enable);
        std::string getTimestamp();

    private:
        Logger() = default;
        ~Logger() = default;
        Logger(const Logger&) = delete;
        Logger& operator=(const Logger&) = delete;

        std::ofstream logfile;
        bool consoleEnabled = true;
        std::mutex logMutex;

        std::string levelToString(LogLevel level);
    };

    // ==============================
    // Filesystem utilities
    // ==============================
    bool ensureDir(const std::string& path);
    bool fileExists(const std::string& path);
    std::string getFileName(const std::string& path);

    // ==============================
    // String utilities
    // ==============================
    std::string toLower(const std::string& input);
    std::string trim(const std::string& str);

    // ==============================
    // Time utilities
    // ==============================
    std::string nowString();
    class Stopwatch {
    public:
        Stopwatch();
        void reset();
        double elapsedMs() const;
    private:
        std::chrono::high_resolution_clock::time_point start;
    };

    // ==============================
    // Error handling helper
    // ==============================
    void tryCatchLog(const std::function<void()>& func, const std::string& context);

} // namespace utils
