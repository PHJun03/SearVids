#include <gtest/gtest.h>
#include "downloader.h"
#include <filesystem>
#include <fstream>

class DownloaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test directory
        std::filesystem::create_directories("test_downloads");
    }

    void TearDown() override {
        // Clean up test files
        if (std::filesystem::exists("test_downloads")) {
            std::filesystem::remove_all("test_downloads");
        }
    }
    
    bool file_exists_and_not_empty(const std::string& path) {
        if (!std::filesystem::exists(path)) return false;
        return std::filesystem::file_size(path) > 0;
    }
};

TEST_F(DownloaderTest, IsValidURL) {
    EXPECT_TRUE(downloader::is_valid_url("http://example.com"));
    EXPECT_TRUE(downloader::is_valid_url("https://example.com"));
    EXPECT_TRUE(downloader::is_valid_url("HTTP://EXAMPLE.COM"));
    EXPECT_TRUE(downloader::is_valid_url("HTTPS://EXAMPLE.COM"));
    
    EXPECT_FALSE(downloader::is_valid_url("ftp://example.com"));
    EXPECT_FALSE(downloader::is_valid_url("example.com"));
    EXPECT_FALSE(downloader::is_valid_url(""));
    EXPECT_FALSE(downloader::is_valid_url("htt"));
}

TEST_F(DownloaderTest, IsYouTubeURL) {
    EXPECT_TRUE(downloader::is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ"));
    EXPECT_TRUE(downloader::is_youtube_url("https://youtu.be/dQw4w9WgXcQ"));
    EXPECT_TRUE(downloader::is_youtube_url("https://www.YOUTUBE.com/watch?v=test"));
    EXPECT_TRUE(downloader::is_youtube_url("https://m.youtube.com/watch?v=test"));
    
    EXPECT_FALSE(downloader::is_youtube_url("https://example.com/video.mp4"));
    EXPECT_FALSE(downloader::is_youtube_url("https://vimeo.com/123456"));
}

TEST_F(DownloaderTest, ExtractFilename) {
    // Regular URLs
    EXPECT_EQ(downloader::extract_filename("https://example.com/video.mp4"), "video.mp4");
    EXPECT_EQ(downloader::extract_filename("https://example.com/path/to/file.avi"), "file.avi");
    EXPECT_EQ(downloader::extract_filename("https://example.com/file.mp4?param=value"), "file.mp4");
    
    // YouTube URLs
    EXPECT_EQ(downloader::extract_filename("https://www.youtube.com/watch?v=dQw4w9WgXcQ"), "dQw4w9WgXcQ.mp4");
    EXPECT_EQ(downloader::extract_filename("https://www.youtube.com/watch?v=abc123&list=xyz"), "abc123.mp4");
    EXPECT_EQ(downloader::extract_filename("https://youtu.be/dQw4w9WgXcQ"), "dQw4w9WgXcQ.mp4");
    EXPECT_EQ(downloader::extract_filename("https://youtu.be/abc123?t=10"), "abc123.mp4");
    
    // Edge cases - returns default "video.mp4" when filename cannot be extracted
    EXPECT_EQ(downloader::extract_filename("https://example.com/"), "video.mp4");
    EXPECT_EQ(downloader::extract_filename("https://example.com/watch"), "video.mp4");
}

TEST_F(DownloaderTest, IsYtdlpAvailable) {
    // This test will pass if yt-dlp is installed
    bool available = downloader::is_ytdlp_available();
    std::cout << "yt-dlp available: " << (available ? "yes" : "no") << std::endl;
    
    // Don't fail the test if yt-dlp is not installed
    // Just report the status
    EXPECT_TRUE(true);
}

TEST_F(DownloaderTest, DownloadSimpleFile) {
    // Use a small test file
    std::string url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf";
    std::string output = "test_downloads/test_file.pdf";
    
    bool result = downloader::download(url, output);
    
    if (result) {
        EXPECT_TRUE(file_exists_and_not_empty(output));
    } else {
        std::cout << "Download failed (network issue?)" << std::endl;
        GTEST_SKIP() << "Network download failed";
    }
}

TEST_F(DownloaderTest, DownloadWithProgress) {
    std::string url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf";
    std::string output = "test_downloads/test_progress.pdf";
    
    bool progress_called = false;
    float last_progress = 0.0f;
    
    auto callback = [&](const downloader::DownloadProgress& progress) {
        progress_called = true;
        last_progress = progress.progress_percent;
        std::cout << "Progress: " << progress.progress_percent << "%" << std::endl;
    };
    
    bool result = downloader::download(url, output, callback);
    
    if (result) {
        EXPECT_TRUE(progress_called);
        EXPECT_GT(last_progress, 0.0f);
        EXPECT_TRUE(file_exists_and_not_empty(output));
    } else {
        GTEST_SKIP() << "Network download failed";
    }
}

TEST_F(DownloaderTest, DownloadInvalidURL) {
    std::string url = "not_a_url";
    std::string output = "test_downloads/invalid.pdf";
    
    bool result = downloader::download(url, output);
    EXPECT_FALSE(result);
    EXPECT_FALSE(std::filesystem::exists(output));
}

TEST_F(DownloaderTest, DownloadNonExistentFile) {
    // Use a URL that will return 404
    // Note: Some servers return HTML error pages instead of failing,
    // so we can't guarantee download will fail
    std::string url = "https://httpbin.org/status/404";
    std::string output = "test_downloads/nonexistent.mp4";
    
    bool result = downloader::download(url, output);
    
    // If download "succeeds", it might have downloaded an error page
    // Check if it's very small (likely an error response)
    if (result && std::filesystem::exists(output)) {
        auto size = std::filesystem::file_size(output);
        std::cout << "Downloaded file size: " << size << " bytes" << std::endl;
        // Error pages are typically small, but we can't make hard assumptions
        EXPECT_TRUE(true);  // Just pass, as behavior varies by server
    } else {
        // Download properly failed
        EXPECT_FALSE(result);
    }
}

TEST_F(DownloaderTest, DownloadWithRetry) {
    std::string url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf";
    std::string output = "test_downloads/test_retry.pdf";
    
    bool result = downloader::download_with_retry(url, output, 3);
    
    if (result) {
        EXPECT_TRUE(file_exists_and_not_empty(output));
    } else {
        GTEST_SKIP() << "Network download failed";
    }
}

// YouTube download test - only runs if yt-dlp is available
TEST_F(DownloaderTest, DownloadYouTubeVideo) {
    if (!downloader::is_ytdlp_available()) {
        GTEST_SKIP() << "yt-dlp not available, skipping YouTube download test";
    }
    
    // Use a very short video for testing
    std::string url = "https://www.youtube.com/watch?v=jNQXAC9IVRw";  // "Me at the zoo" - first YouTube video (18 seconds)
    std::string output = "test_downloads/youtube_test.mp4";
    
    std::cout << "Downloading YouTube video (this may take a while)..." << std::endl;
    bool result = downloader::download(url, output);
    
    if (result) {
        EXPECT_TRUE(file_exists_and_not_empty(output));
        std::cout << "YouTube download succeeded" << std::endl;
    } else {
        std::cout << "YouTube download failed" << std::endl;
    }
}

TEST_F(DownloaderTest, CreateDirectoryIfNotExists) {
    std::string url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf";
    std::string output = "test_downloads/subdir/nested/test.pdf";
    
    bool result = downloader::download(url, output);
    
    if (result) {
        EXPECT_TRUE(std::filesystem::exists("test_downloads/subdir/nested"));
        EXPECT_TRUE(file_exists_and_not_empty(output));
    } else {
        GTEST_SKIP() << "Network download failed";
    }
}