#pragma once

#include <string>
#include <stdexcept>
#include <cstdint>

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
};

// Initialize FFmpeg global state (thread-safe, idempotent).
void init_ffmpeg();

// Probe a media file and return basic metadata. Throws std::runtime_error on error.
VideoInfo probe(const std::string& path);

// Count actual video frames by opening codec and decoding frames.
// This performs decoding; it is slower but accurate. Throws std::runtime_error on error.
int count_frames(const std::string& path);

// Utility: convert duration (AV) to ms (used internally)
inline int64_t avtime_to_ms(int64_t avtime, int64_t time_base_num, int64_t time_base_den);

} // namespace ffmpeg_decoder
