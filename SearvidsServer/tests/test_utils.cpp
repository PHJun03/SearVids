#include <gtest/gtest.h>
#include "../src/utils.h"
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

// ------------------------------
// String utilities
// ------------------------------
TEST(UtilsTest, ToLowerWorks) {
    EXPECT_EQ(utils::toLower("HELLO"), "hello");
    EXPECT_EQ(utils::toLower("World123"), "world123");
    EXPECT_EQ(utils::toLower("MiXeD"), "mixed");
}

TEST(UtilsTest, TrimWorks) {
    EXPECT_EQ(utils::trim("  hello  "), "hello");
    EXPECT_EQ(utils::trim("\n\t test \r\n"), "test");
    EXPECT_EQ(utils::trim(""), "");
}

// ------------------------------
// Filesystem utilities
// ------------------------------
TEST(UtilsTest, EnsureDirCreatesDirectory) {
    std::string testDir = "test_temp_dir";
    fs::remove_all(testDir);  // cleanup
    EXPECT_FALSE(fs::exists(testDir));

    bool result = utils::ensureDir(testDir);
    EXPECT_TRUE(result);
    EXPECT_TRUE(fs::exists(testDir));

    fs::remove_all(testDir);  // cleanup
}

TEST(UtilsTest, FileExistsReturnsCorrectly) {
    std::string tempFile = "temp.txt";
    {
        std::ofstream ofs(tempFile);
        ofs << "test";
    }
    EXPECT_TRUE(utils::fileExists(tempFile));
    fs::remove(tempFile);
    EXPECT_FALSE(utils::fileExists(tempFile));
}

TEST(UtilsTest, GetFileNameExtractsCorrectly) {
    EXPECT_EQ(utils::getFileName("/path/to/file.txt"), "file.txt");
    EXPECT_EQ(utils::getFileName("C:\\temp\\data.bin"), "data.bin");
}

// ------------------------------
// Time & Stopwatch
// ------------------------------
TEST(UtilsTest, StopwatchMeasuresTime) {
    utils::Stopwatch sw;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    double elapsed = sw.elapsedMs();
    EXPECT_GE(elapsed, 45.0); // should be >= 45 ms
}

// ------------------------------
// Logger (basic checks)
// ------------------------------
TEST(UtilsTest, LoggerWritesToFile) {
    utils::Logger& logger = utils::Logger::instance();
    std::string logFile = "test_log.txt";

    logger.setLogFile(logFile);
    logger.enableConsole(false);
    logger.info("Hello GTest!");

    logger.closeLogFile();

    std::ifstream file(logFile);
    std::string line;
    bool found = false;
    while (std::getline(file, line)) {
        if (line.find("Hello GTest!") != std::string::npos) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);

    file.close();
    fs::remove(logFile);
}

// ------------------------------
// tryCatchLog (exception handling)
// ------------------------------
TEST(UtilsTest, TryCatchLogsException) {
    utils::Logger& logger = utils::Logger::instance();
    logger.enableConsole(false);

    utils::tryCatchLog([]() {
        throw std::runtime_error("Test exception");
    }, "TryCatchTest");

    SUCCEED(); // Test passes if no crash
}