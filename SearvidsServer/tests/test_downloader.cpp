#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>
#include <cstdio>
#include "downloader.h"

namespace fs = std::filesystem;

class DownloaderTest : public ::testing::Test {
protected:
    std::string test_dir = "./test_downloads";
    
    void SetUp() override {
        // Create test directory
        if (!fs::exists(test_dir)) {
            fs::create_directories(test_dir);
        }
    }
    
    void TearDown() override {
        // Clean up test files
        if (fs::exists(test_dir)) {
            fs::remove_all(test_dir);
        }
    }
    
    // Helper: create a simple test file on disk for mock download
    std::string create_test_file(const std::string& filename, const std::string& content) {
        std::string filepath = test_dir + "/" + filename;
        std::ofstream file(filepath);
        file << content;
        file.close();
        return filepath;
    }
};

// Test 1: Validate URL — valid http
TEST_F(DownloaderTest, ValidateUrlHttp) {
    EXPECT_TRUE(downloader::is_valid_url("http://example.com/video.mp4"));
}

// Test 2: Validate URL — valid https
TEST_F(DownloaderTest, ValidateUrlHttps) {
    EXPECT_TRUE(downloader::is_valid_url("https://example.com/video.mp4"));
}

// Test 3: Validate URL — invalid (ftp)
TEST_F(DownloaderTest, ValidateUrlInvalid) {
    EXPECT_FALSE(downloader::is_valid_url("ftp://example.com/file.txt"));
}

// Test 4: Validate URL — invalid (no scheme)
TEST_F(DownloaderTest, ValidateUrlNoScheme) {
    EXPECT_FALSE(downloader::is_valid_url("example.com/file.txt"));
}

// Test 5: Extract filename — simple case
TEST_F(DownloaderTest, ExtractFilenameSimple) {
    std::string url = "https://example.com/video.mp4";
    std::string filename = downloader::extract_filename(url);
    EXPECT_EQ(filename, "video.mp4");
}

// Test 6: Extract filename — with query parameters
TEST_F(DownloaderTest, ExtractFilenameWithQueryParams) {
    std::string url = "https://example.com/download/video.mp4?token=abc123&expire=3600";
    std::string filename = downloader::extract_filename(url);
    EXPECT_EQ(filename, "video.mp4");
}

// Test 7: Extract filename — nested path
TEST_F(DownloaderTest, ExtractFilenameNestedPath) {
    std::string url = "https://cdn.example.com/videos/2024/01/movie.mp4";
    std::string filename = downloader::extract_filename(url);
    EXPECT_EQ(filename, "movie.mp4");
}

// Test 8: Extract filename — invalid URL (no slash)
TEST_F(DownloaderTest, ExtractFilenameInvalid) {
    std::string url = "invalid-url";
    std::string filename = downloader::extract_filename(url);
    EXPECT_EQ(filename, "");
}

// Test 9: Extract filename — trailing slash
TEST_F(DownloaderTest, ExtractFilenameTrailingSlash) {
    std::string url = "https://example.com/files/";
    std::string filename = downloader::extract_filename(url);
    EXPECT_EQ(filename, "");
}

// Test 10: Progress callback structure
TEST_F(DownloaderTest, ProgressCallbackStructure) {
    downloader::DownloadProgress progress;
    progress.downloaded_bytes = 500;
    progress.total_bytes = 1000;
    progress.progress_percent = 50.0f;
    
    EXPECT_EQ(progress.downloaded_bytes, 500);
    EXPECT_EQ(progress.total_bytes, 1000);
    EXPECT_FLOAT_EQ(progress.progress_percent, 50.0f);
}

// Test 11: Extract filename — special characters
TEST_F(DownloaderTest, ExtractFilenameSpecialChars) {
    std::string url = "https://example.com/download/my-video_2024.mp4";
    std::string filename = downloader::extract_filename(url);
    EXPECT_EQ(filename, "my-video_2024.mp4");
}

// Test 12: Extract filename — multiple query params with fragment
TEST_F(DownloaderTest, ExtractFilenameComplexUrl) {
    std::string url = "https://example.com/api/download?file=test.zip&format=mp4#section";
    std::string filename = downloader::extract_filename(url);
    EXPECT_EQ(filename, "download");  // Returns last path component before query
}

// Test 13: Validate URL — case sensitivity
TEST_F(DownloaderTest, ValidateUrlCaseSensitivity) {
    EXPECT_TRUE(downloader::is_valid_url("HTTP://example.com/file.txt"));  // uppercase
    EXPECT_TRUE(downloader::is_valid_url("HTTPS://example.com/file.txt"));
}

// Test 14: Extract filename — URL with port
TEST_F(DownloaderTest, ExtractFilenameWithPort) {
    std::string url = "https://example.com:8080/videos/movie.mkv";
    std::string filename = downloader::extract_filename(url);
    EXPECT_EQ(filename, "movie.mkv");
}

// Test 15: Download progress calculation
TEST_F(DownloaderTest, ProgressPercentCalculation) {
    downloader::DownloadProgress progress;
    progress.downloaded_bytes = 250;
    progress.total_bytes = 1000;
    progress.progress_percent = (progress.downloaded_bytes / (float)progress.total_bytes) * 100.0f;
    
    EXPECT_FLOAT_EQ(progress.progress_percent, 25.0f);
}

// Test 16: Extract filename — unicode/encoded characters
TEST_F(DownloaderTest, ExtractFilenameEncoded) {
    std::string url = "https://example.com/files/video%20with%20spaces.mp4";
    std::string filename = downloader::extract_filename(url);
    EXPECT_EQ(filename, "video%20with%20spaces.mp4");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}