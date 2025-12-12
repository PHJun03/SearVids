/*
 * Copyright (c) 2025 Hyeonjun Park (phjun7150@gmail.com)
 * All rights reserved.
 */

#include "server_api.h"
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
#include <functional>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <regex>
#include <algorithm>
#include <cctype>
#include <future>

namespace server_api {

std::mutex g_sessions_mtx;
std::unordered_map<std::string, std::shared_ptr<VideoSession>> g_sessions;
whisper_wrapper::WhisperWrapper g_whisper;

// WebSocket connections: video_id -> list of connections
struct WsConnection {
    crow::websocket::connection* conn;
};
std::mutex g_ws_mtx;
std::unordered_map<std::string, std::vector<WsConnection>> g_ws_connections;

// Helper to broadcast status update
void broadcast_status(const std::string& video_id, const VideoSession& sess) {
    std::lock_guard<std::mutex> lk(g_ws_mtx);
    auto it = g_ws_connections.find(video_id);
    if (it == g_ws_connections.end()) return;

    nlohmann::json j;
    j["type"] = "status";
    j["status"] = sess.error.empty() ? (sess.done ? "done" : (sess.analyzing ? "analyzing" : "pending")) : "error";
    j["progress_percent"] = static_cast<int>(sess.progress_percent);
    j["indexed_visual_frames"] = static_cast<int64_t>(sess.indexed_visual_frames);
    j["indexed_audio_segments"] = static_cast<int64_t>(sess.indexed_audio_segments);
    
    std::string msg = j.dump();
    
    // Iterate and send. Remove closed connections? 
    // Crow handles connection lifecycle, but we should be careful.
    // We just send. If send fails, Crow might close it.
    for (auto& ws : it->second) {
        try {
            ws.conn->send_text(msg);
        } catch (...) {}
    }
}

// Lazy initialize CLIP to avoid loading models during static initialization
static std::unique_ptr<clip_onnx::ClipOnnx> g_clip_ptr;
static std::mutex g_clip_mtx;

static clip_onnx::ClipOnnx& get_clip() {
    std::lock_guard<std::mutex> lock(g_clip_mtx);
    if (!g_clip_ptr) {
        g_clip_ptr = std::make_unique<clip_onnx::ClipOnnx>(
            "/app/models/clip_text_sim.onnx",
            "/app/models/clip_vision_sim.onnx",
            true,           // device_gpu
            224             // image size
        );
    }
    return *g_clip_ptr;
}

// JSON helpers (adds Content-Type and CORS)
static crow::response json_ok(const nlohmann::json& j, int code = 200) {
    crow::response res(code, j.dump());
    res.add_header("Content-Type", "application/json");
    res.add_header("Access-Control-Allow-Origin", "*");
    res.add_header("Cache-Control", "no-store, no-cache, must-revalidate");
    return res;
}
static crow::response json_err(int code, const std::string& msg) {
    nlohmann::json j = nlohmann::json::object();
    j["status"] = "error";
    j["message"] = msg;
    return json_ok(j, code);
}

std::string make_video_id(const std::string& url) {
    std::hash<std::string> h;
    return std::to_string(h(url));
}

void analyze_video_async(std::shared_ptr<VideoSession> sess_ptr) {
    auto& sess = *sess_ptr;
    sess.analyzing = true;
    sess.started_at = std::chrono::system_clock::now();
    sess.current_stage = "initializing";
    sess.progress_percent = 0;

    try {
        if (!downloader::is_valid_url(sess.source_url)) {
            throw std::runtime_error("Invalid URL");
        }

        // Try to get duration for chunk-based processing
        std::cout << "[" << sess.video_id << "] Getting video duration..." << std::endl;
        int64_t total_duration_sec = downloader::get_duration(sess.source_url);
        
        // If duration is available, use Chunk-Based Processing
        if (total_duration_sec > 0) {
            sess.duration_ms = total_duration_sec * 1000;
            std::cout << "[" << sess.video_id << "] Chunk-based analysis. Total duration: " << total_duration_sec << "s" << std::endl;
            
            const int CHUNK_SIZE = 180; // 3 minutes
            int current_start = 0;
            
            // Pipeline: Download N+1 while Analyzing N
            std::future<void> last_analysis_future;
            
            while (current_start < total_duration_sec) {
                int current_end = std::min((int)total_duration_sec, current_start + CHUNK_SIZE);
                
                // Update stage
                sess.current_stage = "processing_chunk_" + std::to_string(current_start) + "_" + std::to_string(current_end);
                broadcast_status(sess.video_id, sess);
                
                // Download Chunk (Blocking, but runs in parallel with last_analysis_future)
                std::string chunk_filename = sess.video_id + "_chunk_" + std::to_string(current_start) + ".mp4";
                std::string chunk_path = "data/" + chunk_filename;
                
                std::cout << "[" << sess.video_id << "] Downloading chunk " << current_start << "-" << current_end << "s" << std::endl;
                if (!downloader::download_section(sess.source_url, chunk_path, current_start, current_end)) {
                    if (current_start == 0) throw std::runtime_error("Failed to download first chunk");
                    std::cerr << "Failed to download chunk " << current_start << "-" << current_end << ", stopping." << std::endl;
                    break;
                }
                
                // Wait for previous analysis to finish before starting new one
                if (last_analysis_future.valid()) {
                    last_analysis_future.get(); // Propagate exceptions
                }
                
                // Start Analysis (Async)
                // Capture by value to ensure variables are valid
                last_analysis_future = std::async(std::launch::async, [=, &sess]() {
                    try {
                        // Probe Chunk
                        auto info = ffmpeg_decoder::probe(chunk_path);
                        
                        // Parallel Analysis (Visual + Audio) within the chunk
                        auto visual_future = std::async(std::launch::async, [&sess, &info, chunk_path, current_start, total_duration_sec]() {
                            if (!info.has_video) return;
                            try {
                                auto& clip = get_clip();
                                
                                // Pipeline Parallelism (Producer-Consumer) for Chunk
                                struct QueueItem {
                                    ffmpeg_decoder::FrameData frame;
                                    bool is_end = false;
                                };
                                struct TSQueue {
                                    std::queue<QueueItem> q;
                                    std::mutex m;
                                    std::condition_variable cv;
                                    void push(QueueItem item) {
                                        std::lock_guard<std::mutex> lk(m);
                                        q.push(std::move(item));
                                        cv.notify_one();
                                    }
                                    QueueItem pop() {
                                        std::unique_lock<std::mutex> lk(m);
                                        cv.wait(lk, [this]{ return !q.empty(); });
                                        QueueItem item = std::move(q.front());
                                        q.pop();
                                        return item;
                                    }
                                } frame_queue;

                                auto consumer_thread = std::thread([&sess, &clip, &frame_queue, current_start, total_duration_sec]() {
                                    while (true) {
                                        auto item = frame_queue.pop();
                                        if (item.is_end) break;
                                        try {
                                            auto& frame = item.frame;
                                            int64_t real_timestamp_ms = frame.timestamp_ms + (current_start * 1000);
                                            
                                            auto emb = clip.encodeImage(frame.rgb_data, frame.width, frame.height);
                                            hnsw_index::add(
                                                emb,
                                                static_cast<float>(real_timestamp_ms) / 1000.0f,
                                                static_cast<float>(real_timestamp_ms) / 1000.0f + 1.0f,
                                                "visual_frame"
                                            );
                                            sess.indexed_visual_frames++;
                                            
                                            double progress = (double)real_timestamp_ms / (double)(total_duration_sec * 1000);
                                            if (progress > 1.0) progress = 1.0;
                                            int p = (int)(progress * 100.0);
                                            if (p > sess.progress_percent) sess.progress_percent = p;
                                            
                                            broadcast_status(sess.video_id, sess);
                                        } catch (...) {}
                                    }
                                });

                                ffmpeg_decoder::extract_frames_with_callback(
                                    chunk_path,
                                    [&frame_queue](const ffmpeg_decoder::FrameData& frame) {
                                        frame_queue.push({frame, false});
                                    },
                                    2.0, 0, 0, 0, 224, 224
                                );
                                frame_queue.push({{}, true});
                                if (consumer_thread.joinable()) consumer_thread.join();
                                
                            } catch (...) {}
                        });

                        auto audio_future = std::async(std::launch::async, [&sess, &info, chunk_path, current_start, total_duration_sec]() {
                            if (!info.has_audio) return;
                            try {
                                std::string audio_path = chunk_path + ".wav";
                                if (!ffmpeg_decoder::extract_audio(chunk_path, audio_path, 16000, 1, nullptr)) return;
                                
                                g_whisper.setCliExecutable("python3");
                                g_whisper.setCliArgsTemplate("/app/whisper_ct2.py {infile}");
                                
                                std::regex re(R"(\[(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s-->\s(\d{2}):(\d{2}):(\d{2})\.(\d{3})\]\s+(.*))");
                                auto& clip = get_clip();
                                std::string buffer;
                                
                                g_whisper.transcribe_with_callback(audio_path, 
                                    [&sess, &clip, &re, &buffer, current_start, total_duration_sec](const std::string& chunk) {
                                        buffer += chunk;
                                        size_t pos;
                                        while ((pos = buffer.find('\n')) != std::string::npos) {
                                            std::string line = buffer.substr(0, pos);
                                            buffer.erase(0, pos + 1);
                                            if (!line.empty() && line.back() == '\r') line.pop_back();
                                            
                                            std::smatch match;
                                            if (std::regex_search(line, match, re)) {
                                                try {
                                                    float start = std::stof(match[1]) * 3600 + std::stof(match[2]) * 60 + std::stof(match[3]) + std::stof(match[4]) / 1000.0f;
                                                    float end = std::stof(match[5]) * 3600 + std::stof(match[6]) * 60 + std::stof(match[7]) + std::stof(match[8]) / 1000.0f;
                                                    std::string text = match[9];
                                                    
                                                    start += current_start;
                                                    end += current_start;
                                                    
                                                    text.erase(0, text.find_first_not_of(" \t"));
                                                    text.erase(text.find_last_not_of(" \t") + 1);
                                                    
                                                    if (!text.empty()) {
                                                        auto emb = clip.encodeText(text);
                                                        hnsw_index::add(emb, start, end, text);
                                                        sess.indexed_audio_segments++;
                                                        
                                                        double progress = (double)(end * 1000.0) / (double)(total_duration_sec * 1000);
                                                        if (progress > 1.0) progress = 1.0;
                                                        int p = (int)(progress * 100.0);
                                                        if (p > sess.progress_percent) sess.progress_percent = p;
                                                        
                                                        broadcast_status(sess.video_id, sess);
                                                    }
                                                } catch (...) {}
                                            }
                                        }
                                    }, 
                                    600
                                );
                                std::filesystem::remove(audio_path);
                            } catch (...) {}
                        });

                        visual_future.wait();
                        audio_future.wait();
                        
                        std::filesystem::remove(chunk_path);
                    } catch (const std::exception& e) {
                        std::cerr << "Error analyzing chunk " << current_start << ": " << e.what() << std::endl;
                    }
                });
                
                current_start += CHUNK_SIZE;
            }
            
            // Wait for the final chunk analysis
            if (last_analysis_future.valid()) {
                last_analysis_future.get();
            }
            
        } else {
            // Fallback to Original Logic (Full Download)
            auto filename = downloader::extract_filename(sess.source_url);
            if (filename.empty()) filename = "input.mp4";
            std::string out = std::string("data/") + sess.video_id + "_" + filename;

            std::cout << "[" << sess.video_id << "] Downloading: " << sess.source_url << std::endl;

            bool ok = downloader::download_with_retry(sess.source_url, out, 3);
            if (!ok) throw std::runtime_error("Download failed after 3 retries");
            
            sess.local_path = out;
            sess.progress_percent = 10;
            std::cout << "[" << sess.video_id << "] Download complete: " << out << std::endl;

            // 2) Probe video metadata (10-15%)
            sess.current_stage = "probing";
            std::cout << "[" << sess.video_id << "] Probing video metadata..." << std::endl;
            
            auto info = ffmpeg_decoder::probe(out);
            
            if (!info.has_video && !info.has_audio) {
                throw std::runtime_error("No valid media streams found");
            }
            
            sess.duration_ms = info.duration_ms;
            sess.nb_frames = (info.nb_frames > 0) ? info.nb_frames : ffmpeg_decoder::count_frames(out);
            sess.progress_percent = 15;

            std::cout << "[" << sess.video_id << "] Video info: "
                    << "duration=" << sess.duration_ms << "ms, "
                    << "frames=" << sess.nb_frames << ", "
                    << "has_audio=" << info.has_audio 
                    << ", has_video=" << info.has_video << std::endl;

            sess.current_stage = "processing_content";
            
            // Parallel processing: Visual (30%) + Audio (45%)
            auto visual_future = std::async(std::launch::async, [&sess, &info]() {
                if (!info.has_video) return;
                try {
                    std::cout << "[" << sess.video_id << "] Extracting visual frames..." << std::endl;
                    auto& clip = get_clip();
                    
                    // Pipeline Parallelism: Producer (FFmpeg) -> Queue -> Consumer (CLIP)
                    struct QueueItem {
                        ffmpeg_decoder::FrameData frame;
                        bool is_end = false;
                    };
                    
                    // Simple thread-safe queue
                    struct TSQueue {
                        std::queue<QueueItem> q;
                        std::mutex m;
                        std::condition_variable cv;
                        
                        void push(QueueItem item) {
                            std::lock_guard<std::mutex> lk(m);
                            q.push(std::move(item));
                            cv.notify_one();
                        }
                        
                        QueueItem pop() {
                            std::unique_lock<std::mutex> lk(m);
                            cv.wait(lk, [this]{ return !q.empty(); });
                            QueueItem item = std::move(q.front());
                            q.pop();
                            return item;
                        }
                    } frame_queue;

                    // Consumer Thread: CLIP Inference & Indexing
                    auto consumer_thread = std::thread([&sess, &clip, &frame_queue]() {
                        while (true) {
                            auto item = frame_queue.pop();
                            if (item.is_end) break;
                            
                            try {
                                auto& frame = item.frame;
                                auto emb = clip.encodeImage(frame.rgb_data, frame.width, frame.height);
                                hnsw_index::add(
                                    emb,
                                    static_cast<float>(frame.timestamp_ms) / 1000.0f,
                                    static_cast<float>(frame.timestamp_ms) / 1000.0f + 1.0f,
                                    "visual_frame"
                                );
                                sess.indexed_visual_frames++;
                                
                                // Update progress (Visual is approx 30% of total, from 15% to 45%)
                                if (sess.nb_frames > 0) {
                                    double progress = (double)frame.timestamp_ms / (double)sess.duration_ms;
                                    if (progress > 1.0) progress = 1.0;
                                    int p = 15 + (int)(30.0 * progress);
                                    if (p > sess.progress_percent) sess.progress_percent = p;
                                }
                                
                                // Broadcast update
                                broadcast_status(sess.video_id, sess);
                            } catch (const std::exception& e) {
                                std::cerr << "[" << sess.video_id << "] CLIP inference failed: " << e.what() << std::endl;
                            }
                        }
                    });

                    // Producer: FFmpeg Decoding
                    ffmpeg_decoder::extract_frames_with_callback(
                        sess.local_path,
                        [&frame_queue](const ffmpeg_decoder::FrameData& frame) {
                            frame_queue.push({frame, false});
                        },
                        2.0, 0, 0, 0, 224, 224
                    );
                    
                    // Signal end
                    frame_queue.push({{}, true});
                    
                    // Wait for consumer
                    if (consumer_thread.joinable()) {
                        consumer_thread.join();
                    }
                    
                    std::cout << "[" << sess.video_id << "] Indexed " << sess.indexed_visual_frames << " visual frames" << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "[" << sess.video_id << "] Visual processing failed: " << e.what() << std::endl;
                }
            });

            auto audio_future = std::async(std::launch::async, [&sess, &info]() {
                if (!info.has_audio) return;
                try {
                    std::string audio_path = "data/" + sess.video_id + "_audio.wav";
                    std::cout << "[" << sess.video_id << "] Extracting audio..." << std::endl;
                    
                    if (!ffmpeg_decoder::extract_audio(sess.local_path, audio_path, 16000, 1, nullptr)) {
                        throw std::runtime_error("Failed to extract audio");
                    }
                    
                    std::cout << "[" << sess.video_id << "] Starting transcription (CTranslate2)..." << std::endl;
                    
                    // Setup Whisper (Use Python script with faster-whisper)
                    g_whisper.setCliExecutable("python3");
                    g_whisper.setCliArgsTemplate("/app/whisper_ct2.py {infile}");
                    
                    // Parse and Index incrementally
                    std::regex re(R"(\[(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s-->\s(\d{2}):(\d{2}):(\d{2})\.(\d{3})\]\s+(.*))");
                    auto& clip = get_clip();
                    std::string buffer;
                    
                    g_whisper.transcribe_with_callback(audio_path, 
                        [&sess, &clip, &re, &buffer](const std::string& chunk) {
                            buffer += chunk;
                            size_t pos;
                            while ((pos = buffer.find('\n')) != std::string::npos) {
                                std::string line = buffer.substr(0, pos);
                                buffer.erase(0, pos + 1);
                                
                                if (!line.empty() && line.back() == '\r') line.pop_back();
                                
                                // Debug: print all lines to debug
                                if (line.find("-->") == std::string::npos) {
                                    std::cout << "[Whisper Log] " << line << std::endl;
                                }

                                std::smatch match;
                                if (std::regex_search(line, match, re)) {
                                    try {
                                        float start = std::stof(match[1]) * 3600 + std::stof(match[2]) * 60 + std::stof(match[3]) + std::stof(match[4]) / 1000.0f;
                                        float end = std::stof(match[5]) * 3600 + std::stof(match[6]) * 60 + std::stof(match[7]) + std::stof(match[8]) / 1000.0f;
                                        std::string text = match[9];
                                        // trim
                                        text.erase(0, text.find_first_not_of(" \t"));
                                        text.erase(text.find_last_not_of(" \t") + 1);
                                        
                                        if (!text.empty()) {
                                            auto emb = clip.encodeText(text);
                                            hnsw_index::add(emb, start, end, text);
                                            sess.indexed_audio_segments++;
                                            
                                            // Update progress (Audio is approx 45% of total, from 45% to 90%)
                                            // We use end time to estimate progress
                                            if (sess.duration_ms > 0) {
                                                double progress = (double)(end * 1000.0) / (double)sess.duration_ms;
                                                if (progress > 1.0) progress = 1.0;
                                                int p = 15 + 30 + (int)(45.0 * progress); // 45% to 90%
                                                if (p > sess.progress_percent) sess.progress_percent = p;
                                            }
                                            
                                            // Broadcast update
                                            broadcast_status(sess.video_id, sess);
                                        }
                                    } catch (...) {}
                                }
                            }
                        }, 
                        600
                    );
                    
                    std::filesystem::remove(audio_path);
                    std::cout << "[" << sess.video_id << "] Indexed " << sess.indexed_audio_segments << " audio segments" << std::endl;

                } catch (const std::exception& e) {
                    std::cerr << "[" << sess.video_id << "] Audio processing failed: " << e.what() << std::endl;
                }
            });

            // Wait for both
            visual_future.wait();
            audio_future.wait();
        }

        sess.progress_percent = 100;
        sess.current_stage = "completed";
        sess.done = true;
        sess.completed_at = std::chrono::system_clock::now();
        
        broadcast_status(sess.video_id, sess);
        
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(
            sess.completed_at - sess.started_at
        ).count();
        
        std::cout << "[" << sess.video_id << "] Analysis completed in " 
                  << duration << " seconds (audio_segments: " << sess.indexed_audio_segments
                  << ", visual_frames: " << sess.indexed_visual_frames << ")" << std::endl;

    } catch (const std::exception& e) {
        sess.error = e.what();
        sess.current_stage = "failed";
        sess.completed_at = std::chrono::system_clock::now();
        
        broadcast_status(sess.video_id, sess);
        
        std::cerr << "[" << sess.video_id << "] Analysis failed: " 
                  << e.what() << std::endl;
    }
    
    sess.analyzing = false;
}

SearchRequest parse_search_request(const std::string& body) {
    SearchRequest req;
    try {
        auto j = nlohmann::json::parse(body);
        if (j.contains("query")) req.query = j["query"];
        if (j.contains("query_text")) req.query = j["query_text"];
        if (j.contains("embedding") && j["embedding"].is_array()) {
            req.embedding = j["embedding"].get<std::vector<float>>();
        }
        if (j.contains("topk")) req.topk = j["topk"];
        if (j.contains("offset")) req.offset = j["offset"];
        if (j.contains("min_similarity")) req.min_similarity = j["min_similarity"];
        if (j.contains("search_type")) req.search_type = j["search_type"];
    } catch (...) {}
    return req;
}

crow::response create_search_response(const std::vector<SearchResult>& results) {
    nlohmann::json j_results = nlohmann::json::array();
    for (const auto& r : results) {
        nlohmann::json item;
        item["id"] = r.id;
        item["start"] = r.start_time;
        item["end"] = r.end_time;
        item["text"] = r.caption;
        item["score"] = r.similarity;
        item["type"] = r.result_type.empty() ? (r.caption == "visual_frame" ? "visual" : "audio") : r.result_type;
        j_results.push_back(item);
    }
    nlohmann::json j;
    j["status"] = "success";
    j["results"] = j_results;
    return json_ok(j);
}

static std::vector<SearchResult> filter_results(const std::vector<hnsw_index::TimelineEntry>& raw_results, const std::string& query) {
    std::vector<SearchResult> filtered;
    filtered.reserve(raw_results.size());
    
    for (const auto& r : raw_results) {
        bool keep = false;
        bool is_visual = (r.caption == "visual_frame");
        
        // 1. Exact Keyword Match (Case-insensitive)
        if (!is_visual && !query.empty()) {
             auto it = std::search(
                r.caption.begin(), r.caption.end(),
                query.begin(), query.end(),
                [](char a, char b) { return std::tolower(a) == std::tolower(b); }
             );
             if (it != r.caption.end()) {
                 keep = true;
             }
        }

        // 2. Threshold
        if (!keep) {
            float threshold = is_visual ? 0.25f : 0.75f;
            if (r.similarity >= threshold) {
                keep = true;
            }
        }

        if (keep) {
            filtered.push_back({r.id, r.start_time, r.end_time, r.caption, r.similarity});
        }
    }
    return filtered;
}

void setup_routes(crow::SimpleApp& app) {
    // WebSocket Route
    CROW_WEBSOCKET_ROUTE(app, "/ws/videos/<string>/status")
    .onopen([&](crow::websocket::connection& conn) {
        // We can't easily get the video_id from the connection object in onopen in older Crow versions?
        // But the route has <string>.
        // Crow passes args to the handler.
        // Wait, CROW_WEBSOCKET_ROUTE syntax with args:
        // .onopen([&](crow::websocket::connection& conn) { ... })
        // It doesn't pass the args to onopen.
        // We need to parse it from conn.get_url()? No.
        // Actually, Crow's websocket route doesn't support capturing args in onopen easily.
        // But we can use a lambda that captures nothing?
        // Let's check Crow documentation or source if available.
        // Assuming we can't get it easily, we might need the client to send a "subscribe" message.
    })
    .onmessage([&](crow::websocket::connection& conn, const std::string& data, bool is_binary) {
        if (is_binary) return;
        try {
            auto j = nlohmann::json::parse(data);
            if (j.contains("type") && j["type"] == "subscribe" && j.contains("video_id")) {
                std::string vid = j["video_id"];
                std::lock_guard<std::mutex> lk(g_ws_mtx);
                g_ws_connections[vid].push_back({&conn});
                
                // Send initial status if exists
                std::lock_guard<std::mutex> lk2(g_sessions_mtx);
                auto it = g_sessions.find(vid);
                if (it != g_sessions.end()) {
                    auto& sess = *it->second;
                    nlohmann::json resp;
                    resp["type"] = "status";
                    resp["status"] = sess.error.empty() ? (sess.done ? "done" : (sess.analyzing ? "analyzing" : "pending")) : "error";
                    resp["progress_percent"] = static_cast<int>(sess.progress_percent);
                    resp["indexed_visual_frames"] = static_cast<int64_t>(sess.indexed_visual_frames);
                    resp["indexed_audio_segments"] = static_cast<int64_t>(sess.indexed_audio_segments);
                    conn.send_text(resp.dump());
                }
            }
        } catch (...) {}
    })
    .onclose([&](crow::websocket::connection& conn, const std::string& reason) {
        std::lock_guard<std::mutex> lk(g_ws_mtx);
        for (auto& kv : g_ws_connections) {
            auto& list = kv.second;
            list.erase(std::remove_if(list.begin(), list.end(), 
                [&](const WsConnection& c) { return c.conn == &conn; }), list.end());
        }
    });

    // POST /videos/analyze: start async ingestion
    CROW_ROUTE(app, "/videos/analyze").methods(crow::HTTPMethod::POST)
    ([](const crow::request& req) {
        nlohmann::json body;
        try { body = nlohmann::json::parse(req.body); } catch (...) { body = nlohmann::json::object(); }
        std::string url = body.value("url", "");
        if (url.empty()) return json_err(400, "url is required");

        auto vid = make_video_id(url);
        {
            std::lock_guard<std::mutex> lk(g_sessions_mtx);
            // Use shared_ptr
            auto it = g_sessions.find(vid);
            if (it == g_sessions.end()) {
                auto sess = std::make_shared<VideoSession>();
                sess->video_id = vid;
                sess->source_url = url;
                g_sessions[vid] = sess;
                
                std::thread([sess]() {
                    analyze_video_async(sess);
                }).detach();
            } else {
                // Already exists, if not analyzing and not done, maybe restart?
                // For now, just return existing
                auto sess = it->second;
                if (!sess->analyzing && !sess->done) {
                     std::thread([sess]() {
                        analyze_video_async(sess);
                    }).detach();
                }
            }
        }
        nlohmann::json resp = nlohmann::json::object();
        resp["status"] = "accepted";
        resp["video_id"] = vid;
        return json_ok(resp, 202);
    });

    // POST /api/analyze (alias of /videos/analyze)
    CROW_ROUTE(app, "/api/analyze").methods(crow::HTTPMethod::POST)
    ([](const crow::request& req) {
        nlohmann::json body;
        try { body = nlohmann::json::parse(req.body); } catch (...) { body = nlohmann::json::object(); }
        std::string url = body.value("url", "");
        if (url.empty()) return json_err(400, "url is required");

        auto vid = make_video_id(url);
        {
            std::lock_guard<std::mutex> lk(g_sessions_mtx);
            auto it = g_sessions.find(vid);
            if (it == g_sessions.end()) {
                auto sess = std::make_shared<VideoSession>();
                sess->video_id = vid;
                sess->source_url = url;
                g_sessions[vid] = sess;
                std::thread([sess]() { analyze_video_async(sess); }).detach();
            } else {
                auto sess = it->second;
                if (!sess->analyzing && !sess->done) {
                     std::thread([sess]() { analyze_video_async(sess); }).detach();
                }
            }
        }
        nlohmann::json resp = nlohmann::json::object();
        resp["status"] = "accepted";
        resp["video_id"] = vid;
        return json_ok(resp, 202);
    });

    // POST /analyze (for proxy_pass that strips /api/)
    CROW_ROUTE(app, "/analyze").methods(crow::HTTPMethod::POST)
    ([](const crow::request& req) {
        // same with /api/analyze
        nlohmann::json body;
        try { body = nlohmann::json::parse(req.body); } catch (...) { body = nlohmann::json::object(); }
        std::string url = body.value("url", "");
        if (url.empty()) return json_err(400, "url is required");
        auto vid = make_video_id(url);
        {
            std::lock_guard<std::mutex> lk(g_sessions_mtx);
            auto it = g_sessions.find(vid);
            if (it == g_sessions.end()) {
                auto sess = std::make_shared<VideoSession>();
                sess->video_id = vid;
                sess->source_url = url;
                g_sessions[vid] = sess;
                std::thread([sess]() { analyze_video_async(sess); }).detach();
            } else {
                auto sess = it->second;
                if (!sess->analyzing && !sess->done) {
                     std::thread([sess]() { analyze_video_async(sess); }).detach();
                }
            }
        }
        nlohmann::json resp = nlohmann::json::object();
        resp["status"] = "accepted";
        resp["video_id"] = vid;
        return json_ok(resp, 202);
    });

    // GET /videos/{video_id}/status
    CROW_ROUTE(app, "/videos/<string>/status").methods(crow::HTTPMethod::GET)
    ([](const std::string& video_id) {
        std::lock_guard<std::mutex> lk(g_sessions_mtx);
        auto it = g_sessions.find(video_id);
        if (it == g_sessions.end()) return json_err(404, "not found");

        const auto& s = *it->second;
        std::string status = s.error.empty()
            ? (s.done ? "done" : (s.analyzing ? "analyzing" : "pending"))
            : "error";
        nlohmann::json j = nlohmann::json::object();
        j["status"] = status;
        j["error"] = s.error;
        j["analyzing"] = static_cast<bool>(s.analyzing);
        j["done"] = static_cast<bool>(s.done);
        j["progress_percent"] = static_cast<int>(s.progress_percent);
        j["current_stage"] = s.current_stage;
        j["duration_ms"] = static_cast<int64_t>(s.duration_ms);
        j["nb_frames"] = static_cast<int64_t>(s.nb_frames);
        j["indexed_visual_frames"] = static_cast<int64_t>(s.indexed_visual_frames);
        j["indexed_audio_segments"] = static_cast<int64_t>(s.indexed_audio_segments);
        return json_ok(j);
    });

    // Alias for proxy: GET /api/videos/{video_id}/status
    CROW_ROUTE(app, "/api/videos/<string>/status").methods(crow::HTTPMethod::GET)
    ([](const std::string& video_id) {
        std::lock_guard<std::mutex> lk(g_sessions_mtx);
        auto it = g_sessions.find(video_id);
        if (it == g_sessions.end()) return json_err(404, "not found");

        const auto& s = *it->second;
        std::string status = s.error.empty()
            ? (s.done ? "done" : (s.analyzing ? "analyzing" : "pending"))
            : "error";
        nlohmann::json j = nlohmann::json::object();
        j["status"] = status;
        j["error"] = s.error;
        j["analyzing"] = static_cast<bool>(s.analyzing);
        j["done"] = static_cast<bool>(s.done);
        j["progress_percent"] = static_cast<int>(s.progress_percent);
        j["current_stage"] = s.current_stage;
        j["duration_ms"] = static_cast<int64_t>(s.duration_ms);
        j["nb_frames"] = static_cast<int64_t>(s.nb_frames);
        j["indexed_visual_frames"] = static_cast<int64_t>(s.indexed_visual_frames);
        j["indexed_audio_segments"] = static_cast<int64_t>(s.indexed_audio_segments);
        return json_ok(j);
    });

    // POST /videos/{video_id}/search
    CROW_ROUTE(app, "/videos/<string>/search").methods(crow::HTTPMethod::POST)
    ([](const crow::request& req, const std::string& video_id) {
        {
            std::lock_guard<std::mutex> lk(g_sessions_mtx);
            if (g_sessions.find(video_id) == g_sessions.end())
                return json_err(404, "video not found");
        }

        auto search_req = parse_search_request(req.body);
        std::vector<float> query_emb;
        if (!search_req.embedding.empty()) {
            query_emb = search_req.embedding;
        } else if (!search_req.query.empty()) {
            try { query_emb = get_clip().encodeText(search_req.query); }
            catch (...) { return json_err(500, "CLIP encoding failed"); }
        } else {
            return json_err(400, "query or embedding required");
        }

        auto hnsw_results = hnsw_index::search(query_emb, search_req.topk);
        auto api_results = filter_results(hnsw_results, search_req.query);
        return create_search_response(api_results);
    });

    // GET /videos/{video_id}/thumbnail?timestamp=12345
    CROW_ROUTE(app, "/videos/<string>/thumbnail").methods(crow::HTTPMethod::GET)
    ([](const crow::request& req, const std::string& video_id) {
        std::string path;
        {
            std::lock_guard<std::mutex> lk(g_sessions_mtx);
            auto it = g_sessions.find(video_id);
            if (it == g_sessions.end()) return crow::response(404);
            path = it->second->local_path;
        }

        char* ts_str = req.url_params.get("timestamp");
        if (!ts_str) return crow::response(400, "timestamp required");
        int64_t ts = std::stoll(ts_str);

        try {
            // Extract frame (resize to 320x180 for thumbnail)
            auto frame = ffmpeg_decoder::extract_frame_at(path, ts, true, 320, 180);
            
            // Convert to OpenCV Mat
            cv::Mat img(frame.height, frame.width, CV_8UC3, frame.rgb_data.data());
            cv::cvtColor(img, img, cv::COLOR_RGB2BGR); // OpenCV uses BGR

            // Encode to JPEG
            std::vector<uchar> buf;
            cv::imencode(".jpg", img, buf);

            std::string s(buf.begin(), buf.end());
            crow::response res(s);
            res.add_header("Content-Type", "image/jpeg");
            res.add_header("Access-Control-Allow-Origin", "*");
            res.add_header("Cache-Control", "public, max-age=3600");
            return res;
        } catch (const std::exception& e) {
            std::cerr << "Thumbnail error: " << e.what() << std::endl;
            return crow::response(500);
        }
    });

    // Alias: GET /api/videos/{video_id}/thumbnail
    CROW_ROUTE(app, "/api/videos/<string>/thumbnail").methods(crow::HTTPMethod::GET)
    ([](const crow::request& req, const std::string& video_id) {
        std::string path;
        {
            std::lock_guard<std::mutex> lk(g_sessions_mtx);
            auto it = g_sessions.find(video_id);
            if (it == g_sessions.end()) return crow::response(404);
            path = it->second->local_path;
        }

        char* ts_str = req.url_params.get("timestamp");
        if (!ts_str) return crow::response(400, "timestamp required");
        int64_t ts = std::stoll(ts_str);

        try {
            auto frame = ffmpeg_decoder::extract_frame_at(path, ts, true, 320, 180);
            cv::Mat img(frame.height, frame.width, CV_8UC3, frame.rgb_data.data());
            cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
            std::vector<uchar> buf;
            cv::imencode(".jpg", img, buf);
            std::string s(buf.begin(), buf.end());
            crow::response res(s);
            res.add_header("Content-Type", "image/jpeg");
            res.add_header("Access-Control-Allow-Origin", "*");
            res.add_header("Cache-Control", "public, max-age=3600");
            return res;
        } catch (const std::exception& e) {
            std::cerr << "Thumbnail error: " << e.what() << std::endl;
            return crow::response(500);
        }
    });

    // Compatibility: GET /index/info for smoke test
    CROW_ROUTE(app, "/index/info").methods(crow::HTTPMethod::GET)
    ([]() {
        nlohmann::json j = nlohmann::json::object();
        j["status"] = "success";
        j["index_size"] = static_cast<int>(hnsw_index::size());
        return json_ok(j);
    });

    // Compatibility: POST /search (query_text or embedding)
    CROW_ROUTE(app, "/search").methods(crow::HTTPMethod::POST)
    ([](const crow::request& req) {
        auto sreq = parse_search_request(req.body);
        if (sreq.query.empty() && sreq.embedding.empty())
            return json_err(400, "query_text or embedding required");

        // 1) ready for embedding
        std::vector<float> emb;
        if (!sreq.embedding.empty()) {
            emb = sreq.embedding;
        } else {
            try {
                emb = get_clip().encodeText(sreq.query);
            } catch (...) {
                return json_err(500, "CLIP encoding failed");
            }
        }

        // 2) HNSW search
        auto hnsw_results = hnsw_index::search(emb, sreq.topk);

        // 3) filter and convert
        auto api_results = filter_results(hnsw_results, sreq.query);

        // 4) create response
        return create_search_response(api_results);
    });

    // Alias: POST /api/search (frontend proxies to /api/*)
    CROW_ROUTE(app, "/api/search").methods(crow::HTTPMethod::POST)
    ([](const crow::request& req) {
        try {
            auto sreq = parse_search_request(req.body);
            if (sreq.query.empty() && sreq.embedding.empty())
                return json_err(400, "query_text or embedding required");

            std::vector<float> emb;
            if (sreq.embedding.empty()) {
                try {
                    emb = get_clip().encodeText(sreq.query);
                } catch (const std::exception& e) {
                    return json_err(500, std::string("CLIP encoding failed: ") + e.what());
                }
            } else {
                emb = sreq.embedding;
            }

            auto hnsw_results = hnsw_index::search(emb, sreq.topk);
            auto api_results = filter_results(hnsw_results, sreq.query);
            return create_search_response(api_results);
        } catch (const std::exception& e) {
            return json_err(500, std::string("Search failed: ") + e.what());
        }
    });

    // GET /api/health
    CROW_ROUTE(app, "/api/health").methods("GET"_method)([] {
        return crow::response(200, "OK");
    });

    // CORS preflight
    CROW_ROUTE(app, "/<path>").methods(crow::HTTPMethod::OPTIONS)
    ([](const crow::request&, std::string) {
        crow::response res(204);
        res.add_header("Access-Control-Allow-Origin", "*");
        res.add_header("Access-Control-Allow-Headers", "Content-Type");
        res.add_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
        return res;
    });
}

} // namespace server_api