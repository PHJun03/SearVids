/*
 * Copyright (c) 2025 Hyeonjun Park (phjun7150@gmail.com)
 * All rights reserved.
 */

#include "server_api.h"
#include <nlohmann/json.hpp>
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
std::unordered_map<std::string, VideoSession> g_sessions;
whisper_wrapper::WhisperWrapper g_whisper;

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

void analyze_video_async(VideoSession& sess) {
    sess.analyzing = true;
    sess.started_at = std::chrono::system_clock::now();
    sess.current_stage = "downloading";
    sess.progress_percent = 0;

    try {
        // 1) Download video (0-10%)
        if (!downloader::is_valid_url(sess.source_url)) {
            throw std::runtime_error("Invalid URL");
        }
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
        // Base progress: 15%
        // Visual: 0 -> 30
        // Audio: 0 -> 45
        
        auto visual_future = std::async(std::launch::async, [&sess, &info]() {
            if (!info.has_video) return;
            try {
                std::cout << "[" << sess.video_id << "] Extracting visual frames..." << std::endl;
                auto& clip = get_clip();
                
                ffmpeg_decoder::extract_frames_with_callback(
                    sess.local_path,
                    [&sess, &clip](const ffmpeg_decoder::FrameData& frame) {
                        try {
                            auto emb = clip.encodeImage(frame.rgb_data, frame.width, frame.height);
                            hnsw_index::add(
                                emb,
                                static_cast<float>(frame.timestamp_ms) / 1000.0f,
                                static_cast<float>(frame.timestamp_ms) / 1000.0f + 1.0f,
                                "visual_frame"
                            );
                            sess.indexed_visual_frames++;
                        } catch (...) {}
                    },
                    2.0, 0, 0, 0
                );
                
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

        sess.progress_percent = 100;
        sess.current_stage = "completed";
        sess.done = true;
        sess.completed_at = std::chrono::system_clock::now();
        
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
        
        std::cerr << "[" << sess.video_id << "] Analysis failed: " 
                  << e.what() << std::endl;
    }
    
    sess.analyzing = false;
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
            auto& sess = g_sessions[vid];
            if (sess.video_id.empty()) {
                sess.video_id = vid;
                sess.source_url = url;
                // Capture video_id to avoid dangling reference
                std::thread([video_id = sess.video_id]() {
                    VideoSession* psess = nullptr;
                    {
                        std::lock_guard<std::mutex> lk2(g_sessions_mtx);
                        auto it = g_sessions.find(video_id);
                        if (it != g_sessions.end()) {
                            psess = &it->second;
                        }
                    } // lock released here
                    if (psess) {
                        analyze_video_async(*psess);
                    }
                }).detach();
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
            auto& sess = g_sessions[vid];
            if (sess.video_id.empty()) {
                sess.video_id = vid;
                sess.source_url = url;
                std::thread([video_id = sess.video_id]() {
                    VideoSession* psess = nullptr;
                    {
                        std::lock_guard<std::mutex> lk2(g_sessions_mtx);
                        auto it = g_sessions.find(video_id);
                        if (it != g_sessions.end()) psess = &it->second;
                    }
                    if (psess) analyze_video_async(*psess);
                }).detach();
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
            auto& sess = g_sessions[vid];
            if (sess.video_id.empty()) {
                sess.video_id = vid;
                sess.source_url = url;
                std::thread([video_id = sess.video_id]() {
                    VideoSession* psess = nullptr;
                    {
                        std::lock_guard<std::mutex> lk2(g_sessions_mtx);
                        auto it = g_sessions.find(video_id);
                        if (it != g_sessions.end()) psess = &it->second;
                    }
                    if (psess) analyze_video_async(*psess);
                }).detach();
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

        const auto& s = it->second;
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

        const auto& s = it->second;
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


SearchRequest parse_search_request(const std::string& body) {
    SearchRequest req;
    try {
        auto json = nlohmann::json::parse(body);
        
        if (json.contains("query") && json["query"].is_string())
            req.query = json["query"].get<std::string>();
        else if (json.contains("query_text") && json["query_text"].is_string())
            req.query = json["query_text"].get<std::string>();

        req.topk = json.value("topk", 5);
        req.offset = json.value("offset", 0);
        req.min_similarity = json.value("min_similarity", 0.0f);
        req.search_type = json.value("search_type", "both");
        
        if (json.contains("embedding") && json["embedding"].is_array()) {
            req.embedding = json["embedding"].get<std::vector<float>>();
        }
    } catch (...) {
        // keep defaults
    }
    return req;
}

crow::response create_search_response(const std::vector<SearchResult>& results) {
    nlohmann::json json_results = nlohmann::json::array();
    for (const auto& r : results) {
        nlohmann::json item = nlohmann::json::object();
        item["id"] = r.id;
        item["start_time"] = r.start_time;
        item["end_time"] = r.end_time;
        item["caption"] = r.caption;
        item["similarity"] = r.similarity;
        json_results.push_back(item);
    }
    nlohmann::json out = nlohmann::json::object();
    out["status"] = "ok";
    out["count"] = static_cast<int>(results.size());
    out["results"] = json_results;
    return json_ok(out);
}

crow::response health_check() {
    nlohmann::json j = nlohmann::json::object();
    j["status"] = "ok";
    j["service"] = "SearvidsServer";
    j["index_size"] = static_cast<int>(hnsw_index::size());
    j["whisper_cli_available"] = server_api::g_whisper.cli_available();
    return json_ok(j);
}

} // namespace server_api