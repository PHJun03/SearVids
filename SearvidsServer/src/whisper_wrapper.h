#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <optional>

namespace whisper_wrapper {

/**
 * Wrapper that runs a whisper.cpp command-line executable to transcribe files.
 *
 * Design:
 *  - By default tries to run `whisper` or `main` in the PATH (configurable).
 *  - transcribe_from_file() runs the CLI, captures stdout/stderr, returns the textual transcript.
 *  - Provides timeout support (seconds).
 *
 * Note:
 *  - The wrapper assumes the whisper CLI accepts an input filename and prints a transcript to stdout.
 *    Adjust the CLI arguments via setCliArgsTemplate() if your built whisper CLI needs different flags.
 */

class WhisperWrapper {
public:
    // Get singleton or construct your own instance
    WhisperWrapper();
    explicit WhisperWrapper(std::string cli_executable_path);

    // Set path to whisper CLI executable (absolute or relative)
    void setCliExecutable(const std::string& path);

    // Optional: template for CLI args; use "{infile}" as placeholder for input path.
    // Example template: "--task transcribe --model tiny.en {infile}"
    void setCliArgsTemplate(const std::string& tpl);

    // Transcribe a file synchronously. Returns transcript on success, throws std::runtime_error on failure.
    // timeout_seconds: if <=0, no timeout.
    std::string transcribe_from_file(const std::string& infile, int timeout_seconds = 600) const;

    // Helper: check if the configured CLI executable exists / is runnable
    bool cli_available() const;

    // Build the command vector from template + infile
    std::vector<std::string> buildCommand(const std::string& infile) const;

    // Run external command with capture + timeout (platform specific)
    // Returns pair { exit_code (or -1 on failure), stdout+stderr string }
    std::pair<int,std::string> runCommandCapture(const std::vector<std::string>& argv, int timeout_seconds) const;

private:
    std::string cliExe;        // path to CLI or program name
    std::string argsTemplate;  // template string containing "{infile}"
};

} // namespace whisper_wrapper
