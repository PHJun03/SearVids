/*
 * Copyright (c) 2025 Hyeonjun Park (phjun7150@gmail.com)
 * All rights reserved.
 */

#pragma once
#include <crow/app.h>
#include <crow/http_response.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <chrono>

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
    std::atomic<bool> cancelled{false};
    std::atomic<int> progress_percent{0};

    std::string current_stage;
    std::string error;

    int64_t duration_ms{0};
    int64_t nb_frames{0};
    std::atomic<int64_t> indexed_audio_segments{0};
    std::atomic<int64_t> indexed_visual_frames{0};

    std::chrono::system_clock::time_point started_at;
    std::chrono::system_clock::time_point completed_at;
};

// Search request structure
struct SearchRequest {
    std::string query;
    std::vector<float> embedding;
    int topk = 5;
    int offset = 0;
    float min_similarity = 0.0f;
    std::string search_type = "both";
    std::string video_id;
};

// Search result structure
struct SearchResult {
    int id;
    float start_time;
    float end_time;
    std::string caption;
    float similarity;
    std::string result_type;
};

// Global state (defined in .cpp)
extern std::mutex g_sessions_mtx;
extern std::unordered_map<std::string, std::shared_ptr<VideoSession>> g_sessions;
extern whisper_wrapper::WhisperWrapper g_whisper;
// NOTE: removed extern ClipOnnx g_clip; we use a lazy getter in .cpp

// Generate video ID from URL
std::string make_video_id(const std::string& url);

// Async video analysis pipeline
void analyze_video_async(std::shared_ptr<VideoSession> sess);

// Initialize API routes
void setup_routes(crow::SimpleApp& app);

// Parse search request from JSON (supports "query" and "query_text")
SearchRequest parse_search_request(const std::string& body);

// Create search response JSON
crow::response create_search_response(const std::vector<SearchResult>& results);

// Health check endpoint
crow::response health_check();

} // namespace server_api