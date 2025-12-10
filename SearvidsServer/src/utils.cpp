/*
 * Copyright (c) 2025 Hyeonjun Park (phjun7150@gmail.com)
 * All rights reserved.
 */

#include "utils.h"
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace fs = std::filesystem;
namespace utils {

// ==============================
// Logger Implementation
// ==============================
Logger& Logger::instance() {
    static Logger inst;
    return inst;
}

std::string Logger::getTimestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm;
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

std::string Logger::levelToString(LogLevel level) {
    switch (level) {
        case LogLevel::INFO: return "INFO";
        case LogLevel::WARN: return "WARN";
        case LogLevel::ERR: return "ERROR";
        default: return "UNKNOWN";
    }
}

void Logger::log(LogLevel level, const std::string& msg) {
    std::lock_guard<std::mutex> lock(logMutex);

    std::ostringstream formatted;
    formatted << "[" << getTimestamp() << "] [" << levelToString(level) << "] " << msg << "\n";

    // Write to console
    if (consoleEnabled)
        std::cout << formatted.str();

    // Write to file (if open)
    if (logfile.is_open()) {
        logfile << formatted.str();
        logfile.flush();  // ✅ ensure written to disk immediately
    }
}

void Logger::info(const std::string& msg) { log(LogLevel::INFO, msg); }
void Logger::warn(const std::string& msg) { log(LogLevel::WARN, msg); }
void Logger::error(const std::string& msg) { log(LogLevel::ERR, msg); }

void Logger::setLogFile(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(logMutex);
    if (logfile.is_open()) logfile.close();  // ✅ close previous file
    logfile.open(filepath, std::ios::app);
    if (!logfile)
        std::cerr << "[Logger] Failed to open log file: " << filepath << std::endl;
}

void Logger::enableConsole(bool enable) {
    consoleEnabled = enable;
}

// Explicitly close the file (for tests and clean shutdown)
void Logger::closeLogFile() {
    std::lock_guard<std::mutex> lock(logMutex);
    if (logfile.is_open()) {
        logfile.flush();
        logfile.close();
    }
}

// ==============================
// Filesystem utilities
// ==============================
bool ensureDir(const std::string& path) {
    try {
        if (!fs::exists(path))
            fs::create_directories(path);
        return true;
    } catch (const std::exception& e) {
        Logger::instance().error("ensureDir failed: " + std::string(e.what()));
        return false;
    }
}

bool fileExists(const std::string& path) {
    return fs::exists(path);
}

std::string getFileName(const std::string& path) {
    return fs::path(path).filename().string();
}

// ==============================
// String utilities
// ==============================
std::string toLower(const std::string& input) {
    std::string s = input;
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return s;
}

std::string trim(const std::string& str) {
    auto start = str.find_first_not_of(" \t\n\r");
    auto end = str.find_last_not_of(" \t\n\r");
    return (start == std::string::npos) ? "" : str.substr(start, end - start + 1);
}

// ==============================
// Time utilities
// ==============================
std::string nowString() {
    return Logger::instance().getTimestamp();
}

Stopwatch::Stopwatch() { reset(); }
void Stopwatch::reset() { start = std::chrono::high_resolution_clock::now(); }
double Stopwatch::elapsedMs() const {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count();
}

// ==============================
// Error handling
// ==============================
void tryCatchLog(const std::function<void()>& func, const std::string& context) {
    try {
        func();
    } catch (const std::exception& e) {
        Logger::instance().error("Exception in " + context + ": " + e.what());
    } catch (...) {
        Logger::instance().error("Unknown exception in " + context);
    }
}

} // namespace utils
