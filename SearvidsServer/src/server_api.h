#pragma once
#include <crow/app.h>
#include <crow/http_response.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>

#include "downloader.h"
#include "ffmpeg_decoder.h"
#include "whisper_wrapper.h"
#include "clip_onnx.h"
#include "hnsw_index.h"

namespace server_api {

// Video analysis session
struct VideoSession {
    std::string video_id;
    std::string source_url;
    std::string local_path;
    std::atomic<bool> analyzing{false};
    std::atomic<bool> done{false};
    std::string error;
    int64_t duration_ms{0};
    int64_t nb_frames{0};
};

// Search request structure
struct SearchRequest {
    std::string query;
    std::vector<float> embedding;
    int topk = 5;
};

// Search result structure
struct SearchResult {
    int id;
    float start_time;
    float end_time;
    std::string caption;
    float similarity;
};

// Global state (defined in .cpp)
extern std::mutex g_sessions_mtx;
extern std::unordered_map<std::string, VideoSession> g_sessions;
extern whisper_wrapper::WhisperWrapper g_whisper;
extern clip_onnx::ClipOnnx g_clip;

/**
 * Generate video ID from URL
 */
std::string make_video_id(const std::string& url);

/**
 * Async video analysis pipeline
 */
void analyze_video_async(VideoSession& sess);

/**
 * Initialize API routes
 */
void setup_routes(crow::SimpleApp& app);

/**
 * Parse search request from JSON
 */
SearchRequest parse_search_request(const std::string& body);

/**
 * Create search response JSON
 */
crow::response create_search_response(const std::vector<SearchResult>& results);

/**
 * Health check endpoint
 */
crow::response health_check();

} // namespace server_api