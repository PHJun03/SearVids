/*
 * Copyright (c) 2025 Hyeonjun Park (phjun7150@gmail.com)
 * All rights reserved.
 */

#pragma once

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>


namespace ffmpeg_decoder {

struct VideoInfo {
  bool has_video = false;
  bool has_audio = false;

  int width = 0;
  int height = 0;
  double fps = 0.0;
  int64_t duration_ms = 0; // duration in milliseconds
  int64_t nb_frames = 0;   // reported/decoded frames (0 if unknown)
  std::string format_name;
  std::string codec_name;

  // Audio stream information
  int audio_sample_rate = 0;
  int audio_channels = 0; // 1=mono, 2=stereo
  std::string audio_codec_name;
};

// Initialize FFmpeg global state (thread-safe, idempotent).
void init_ffmpeg();

// Probe a media file and return basic metadata. Throws std::runtime_error on
// error.
VideoInfo probe(const std::string &path);

// Count actual video frames by opening codec and decoding frames.
// This performs decoding; it is slower but accurate. Throws std::runtime_error
// on error.
int count_frames(const std::string &path);

// Extract audio track from video and save as WAV file.
bool extract_audio(
    const std::string &video_path, // Input video file path
    const std::string &audio_path, // Output WAV file path (will be overwritten)
    int sample_rate =
        16000, // Target sample rate in Hz (default: 16000 for Whisper)
    int channels =
        1, // Target channels (1=mono, 2=stereo; default: 1 for Whisper)
    std::function<void(double)> progress_callback =
        nullptr); // Progress callback (0.0 to 1.0), can be nullptr

// Extract video frames at specified intervals
struct FrameData {
  std::vector<uint8_t> rgb_data; // RGB24 format (width * height * 3 bytes)
  int width;
  int height;
  int64_t timestamp_ms; // Frame timestamp in milliseconds
};

enum class FrameExtractionMethod {
  INTERVAL,    // Extract frames at fixed intervals
  KEYFRAMES,   // Extract only keyframes (I-frames)
  SCENE_DETECT // Extract frames based on scene changes (content-aware)
};

std::vector<FrameData> extract_frames(
    const std::string &video_path, // Input video file path
    double interval_seconds =
        2.0, // Time interval between extracted frames (default: 2.0 seconds)
    int max_frames = 0, // Maximum number of frames to extract (0 = no limit)
    int64_t start_time_ms = 0, // Start time in milliseconds (default: 0)
    int64_t end_time_ms = 0,   // End time in milliseconds (0 = until end)
    int target_width = -1,  // Target width for resizing (-1 = original width)
    int target_height = -1, // Target height for resizing (-1 = original height)
    FrameExtractionMethod method = FrameExtractionMethod::INTERVAL,
    double min_scene_len =
        0.0); // Adaptive Scene Detect: Max duration (sec) without split

// Extract video frames with callback (streaming)
void extract_frames_with_callback(
    const std::string &video_path,
    std::function<void(const FrameData &)> callback,
    double interval_seconds = 2.0, int max_frames = 0,
    int64_t start_time_ms = 0, int64_t end_time_ms = 0, int target_width = -1,
    int target_height = -1,
    FrameExtractionMethod method = FrameExtractionMethod::INTERVAL,
    double min_scene_len = 0.0);

// Extract a single frame at specific timestamp
FrameData extract_frame_at(
    const std::string &video_path, // Input video file path
    int64_t timestamp_ms,          // Timestamp in milliseconds
    bool seek_backward =
        true,              // If true, seek to nearest keyframe before timestamp
    int target_width = -1, // Target width for resizing (-1 = original width)
    int target_height =
        -1); // Target height for resizing (-1 = original height)

// Utility: convert duration (AV) to ms (used internally)
int64_t avtime_to_ms(int64_t avtime, int64_t time_base_num,
                     int64_t time_base_den);

// Utility: convert ms to AV time base
int64_t ms_to_avtime(int64_t ms, int64_t time_base_num, int64_t time_base_den);

// Get FFmpeg version information
struct FFmpegVersion {
  std::string avformat_version;
  std::string avcodec_version;
  std::string avutil_version;
  std::string swresample_version;
  std::string swscale_version;
};

FFmpegVersion get_ffmpeg_version();

} // namespace ffmpeg_decoder
