#include <gtest/gtest.h>
#include "whisper_wrapper.h"

#include <string>
#include <vector>
#include <algorithm>

using whisper_wrapper::WhisperWrapper;

static std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return (char)std::tolower(c); });
    return s;
}

#if defined(_WIN32)

TEST(WhisperWrapperTest, RunCommandSuccessEcho) {
    WhisperWrapper w;
    // Verify we can run a simple shell command and capture stdout
    // Use cmd.exe /C echo hello (stdout ends with \r\n on Windows)
    std::vector<std::string> argv = { "cmd.exe", "/C", "echo", "hello" };
    auto [code, output] = w.runCommandCapture(argv, /*timeout_seconds*/ 5);
    EXPECT_EQ(code, 0);
    auto outLower = toLower(output);
    EXPECT_NE(outLower.find("hello"), std::string::npos);
}

TEST(WhisperWrapperTest, TimeoutTerminatesLongProcess) {
    WhisperWrapper w;
    // Use ping to simulate a long-running command (≈6 seconds)
    // This should be terminated by our 1-second timeout
    std::vector<std::string> argv = { "ping", "127.0.0.1", "-n", "6" };
    auto [code, output] = w.runCommandCapture(argv, /*timeout_seconds*/ 1);

    EXPECT_EQ(code, -1);
    EXPECT_EQ(output, "timeout");
}

TEST(WhisperWrapperTest, TranscribeThrowsWhenCliMissing) {
    WhisperWrapper w;
    // Point to a non-existent executable to force process creation failure
    w.setCliExecutable("___no_such_executable___.exe");
    w.setCliArgsTemplate("{infile}");

    EXPECT_THROW({
        try {
            (void)w.transcribe_from_file("C:\\Windows\\System32\\notepad.exe", 1);
        } catch (const std::runtime_error& e) {
            // Ensure the error message indicates launcher failure
            std::string msg = e.what();
            EXPECT_NE(msg.find("Whisper runner failed"), std::string::npos);
            throw;
        }
    }, std::runtime_error);
}

TEST(WhisperWrapperTest, CliAvailableCmdExe) {
    WhisperWrapper w;
    // cmd.exe should be resolvable on most Windows setups
    w.setCliExecutable("cmd.exe");
    EXPECT_TRUE(w.cli_available());
}

#endif // _WIN32