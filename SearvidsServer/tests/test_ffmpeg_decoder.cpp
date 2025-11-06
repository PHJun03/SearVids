#include <gtest/gtest.h>
#include "../src/ffmpeg_decoder.h"
#include <fstream>
#include <cstdlib>

using namespace ffmpeg_decoder;

// Helper: make sure ffmpeg_decoder initializes without crashing
TEST(FFmpegDecoderTest, InitDoesNotThrow) {
    EXPECT_NO_THROW(init_ffmpeg());
    EXPECT_NO_THROW(init_ffmpeg()); // idempotent check
}

// Helper: convert a valid sample file path (change this if you have a test file)
static std::string getSampleFilePath() {
    // Adjust this path to an existing small media file in your environment
    // For CI or testing, you can download a tiny sample from FFmpeg samples
    std::string path = "sample.mp4";
    return path;
}

// Simple sanity check: probe() should throw on invalid file
TEST(FFmpegDecoderTest, ProbeThrowsOnInvalidFile) {
    EXPECT_THROW(probe("nonexistent_file.mp4"), std::runtime_error);
}

// If you have a valid media file, probe() should return valid metadata
TEST(FFmpegDecoderTest, ProbeReturnsValidMetadata) {
    std::string path = getSampleFilePath();
    if (!std::ifstream(path).good()) {
        GTEST_SKIP() << "Skipping: sample file not found (" << path << ")";
    }

    EXPECT_NO_THROW({
        VideoInfo info = probe(path);
        EXPECT_TRUE(info.has_video || info.has_audio);
        EXPECT_GE(info.duration_ms, 0);
        if (info.has_video) {
            EXPECT_GT(info.width, 0);
            EXPECT_GT(info.height, 0);
            EXPECT_GT(info.fps, 0);
        }
    });
}

// count_frames() should not crash or throw even for invalid file
TEST(FFmpegDecoderTest, CountFramesThrowsOnInvalidFile) {
    EXPECT_THROW(count_frames("does_not_exist.mp4"), std::runtime_error);
}

// count_frames() should return a reasonable number for a valid short video
TEST(FFmpegDecoderTest, CountFramesReturnsPositiveForValidFile) {
    std::string path = getSampleFilePath();
    if (!std::ifstream(path).good()) {
        GTEST_SKIP() << "Skipping: sample file not found (" << path << ")";
    }

    EXPECT_NO_THROW({
        int frames = count_frames(path);
        EXPECT_GE(frames, 0);
        EXPECT_LT(frames, 100000); // sanity upper limit
    });
}

// avtime_to_ms() basic conversion check
TEST(FFmpegDecoderTest, AVTimeToMsConversionWorks) {
    int64_t avtime = 5000; // arbitrary
    int64_t result = avtime_to_ms(avtime, 1, 1000); // 5000 * (1/1000) * 1000 = 5000
    EXPECT_EQ(result, 5000);
}