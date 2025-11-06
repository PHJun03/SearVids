#include "../src/whisper_wrapper.h"
#include <gtest/gtest.h>
#include <fstream>
#include <cstdio>

using namespace whisper_wrapper;

class WhisperWrapperTest : public ::testing::Test {
protected:
    WhisperWrapper wrapper;
};

// ------------------------------
// CLI detection test
// ------------------------------
TEST_F(WhisperWrapperTest, DetectsCliAvailability) {
    // If whisper is not installed, the result may vary, so we only check that it runs without crashing
    EXPECT_NO_THROW({
        bool available = wrapper.cli_available();
        (void)available;
    });
}

// ------------------------------
// Command building test
// ------------------------------
TEST_F(WhisperWrapperTest, BuildsCommandCorrectly) {
    wrapper.setCliExecutable("whisper-cli");
    wrapper.setCliArgsTemplate("--language en {infile}");
    auto cmd = wrapper.buildCommand("audio.wav");
    ASSERT_EQ(cmd.size(), 4);
    EXPECT_EQ(cmd[0], "whisper-cli");
    EXPECT_EQ(cmd[1], "--language");
    EXPECT_EQ(cmd[2], "en");
    // Ensure the input file is correctly replaced
    bool hasInput = false;
    for (auto& c : cmd)
        if (c.find("audio.wav") != std::string::npos) hasInput = true;
    EXPECT_TRUE(hasInput);
}

// ------------------------------
// Run command capture test (Windows vs POSIX are handled internally)
// ------------------------------
TEST_F(WhisperWrapperTest, RunCommandCaptureEcho) {
    // Use a simple echo command that works on both POSIX and Windows
#if defined(_WIN32)
    wrapper.setCliExecutable("cmd");
    wrapper.setCliArgsTemplate("/C echo hello");
#else
    wrapper.setCliExecutable("echo");
    wrapper.setCliArgsTemplate("hello");
#endif
    auto [code, output] = wrapper.runCommandCapture(wrapper.buildCommand("dummy"), 5);
    EXPECT_EQ(code, 0);
    EXPECT_NE(output.find("hello"), std::string::npos);
}

// ------------------------------
// Timeout behavior test
// ------------------------------
TEST_F(WhisperWrapperTest, TimeoutTerminatesLongProcess) {
#if defined(_WIN32)
    wrapper.setCliExecutable("cmd");
    wrapper.setCliArgsTemplate("/C timeout /T 3 >nul");
#else
    wrapper.setCliExecutable("sh");
    wrapper.setCliArgsTemplate("-c 'sleep 3'");
#endif
    auto [code, output] = wrapper.runCommandCapture(wrapper.buildCommand("dummy"), 1);
    // When timeout occurs, code should be -1 and output should indicate "timeout"
    EXPECT_EQ(code, -1);
    EXPECT_EQ(output, "timeout");
}

// --- Transcription behavior test with dummy command ---
TEST_F(WhisperWrapperTest, TranscribeFromFileReturnsOutput) {
#if defined(_WIN32)
    wrapper.setCliExecutable("cmd");
    wrapper.setCliArgsTemplate("/C echo transcribed text");
#else
    wrapper.setCliExecutable("echo");
    wrapper.setCliArgsTemplate("transcribed text");
#endif

    EXPECT_NO_THROW({
        std::string out = wrapper.transcribe_from_file("dummy.wav", 5);
        EXPECT_NE(out.find("transcribed text"), std::string::npos);
    });
}

// ------------------------------
// Error handling test for missing executable
// ------------------------------
TEST_F(WhisperWrapperTest, ThrowsWhenCliNotConfigured) {
    WhisperWrapper bad;
    bad.setCliExecutable("");
    EXPECT_THROW({
        bad.transcribe_from_file("anything.wav", 5);
    }, std::runtime_error);
}
