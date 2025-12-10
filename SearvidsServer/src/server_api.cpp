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
            false,          // device_gpu
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

        // 3) Extract and index visual frames (15-45%)
        if (info.has_video) {
            sess.current_stage = "extracting_frames";
            std::cout << "[" << sess.video_id << "] Extracting visual frames..." << std::endl;

            try {
                // Extract frames every 2 seconds
                auto frames = ffmpeg_decoder::extract_frames(
                    sess.local_path,
                    2.0,  // 2 seconds interval
                    0,    // no max limit
                    0,    // start from beginning
                    0     // until end
                );

                sess.progress_percent = 30;
                std::cout << "[" << sess.video_id << "] Extracted " << frames.size() << " frames" << std::endl;

                // Index frames with CLIP Vision
                sess.current_stage = "indexing_frames";
                auto& clip = get_clip();

                for (size_t i = 0; i < frames.size(); ++i) {
                    const auto& frame = frames[i];
                    
                    try {
                        // Encode frame with CLIP Vision
                        auto emb = clip.encodeImage(frame.rgb_data, frame.width, frame.height);
                        
                        // Add to HNSW index with "visual" tag
                        hnsw_index::add(
                            emb,
                            static_cast<float>(frame.timestamp_ms) / 1000.0f,
                            static_cast<float>(frame.timestamp_ms) / 1000.0f + 1.0f,
                            "visual_frame"
                        );
                        
                        sess.indexed_visual_frames++;
                        
                        // Update progress: 30% -> 45%
                        int progress = 30 + static_cast<int>((i + 1) * 15.0 / frames.size());
                        sess.progress_percent = progress;
                        
                    } catch (const std::exception& e) {
                        std::cerr << "[" << sess.video_id << "] Failed to encode frame at " 
                                  << frame.timestamp_ms << "ms: " << e.what() << std::endl;
                    }
                }

                std::cout << "[" << sess.video_id << "] Indexed " << sess.indexed_visual_frames 
                          << " visual frames" << std::endl;
                
            } catch (const std::exception& e) {
                std::cerr << "[" << sess.video_id << "] Frame extraction failed: " 
                          << e.what() << std::endl;
                // Continue to audio processing
            }
        }

        sess.progress_percent = 45;

        // 4) Extract audio (45-55%)
        if (info.has_audio) {
            sess.current_stage = "extracting_audio";
            std::string audio_path = "data/" + sess.video_id + "_audio.wav";

            std::cout << "[" << sess.video_id << "] Extracting audio..." << std::endl;

            bool audio_ok = ffmpeg_decoder::extract_audio(
                sess.local_path,
                audio_path,
                16000,  // 16kHz for Whisper
                1,      // Mono
                [&sess](double progress) {
                    int new_progress = 45 + static_cast<int>(progress * 10.0);
                    sess.progress_percent = new_progress;
                }
            );

            if (!audio_ok) {
                throw std::runtime_error("Failed to extract audio");
            }
            
            sess.progress_percent = 55;
            std::cout << "[" << sess.video_id << "] Audio extraction complete" << std::endl;

            // 5) Transcribe with Whisper (55-75%)
            sess.current_stage = "transcribing";
            std::string transcript;

            try {
                std::cout << "[" << sess.video_id << "] Starting transcription..." << std::endl;
                
                // Try to find whisper-cli in the build directory first
                std::string whisperPath = "SearvidsServer/third_party/whisper.cpp/build/bin/Release/whisper-cli.exe";
                if (!std::filesystem::exists(whisperPath)) {
                    if (std::filesystem::exists("/app/whisper-cli")) {
                        whisperPath = "/app/whisper-cli";
                    } else {
                        whisperPath = "whisper-cli.exe";
                    }
                }
                g_whisper.setCliExecutable(whisperPath);

                const char* envModel = std::getenv("WHISPER_MODEL_PATH");
                std::string modelPath = envModel
                    ? std::string(envModel)
                    : std::string("/app/models/ggml-tiny.en.bin");

                std::filesystem::create_directories("data");
                
                // Use stdout capture instead of file output
                // -nt: no timestamps (just text)
                // -np: no prints (only results)
                // -f: input file (explicit flag)
                // Remove --task as it is not supported by this version of whisper-cli
                g_whisper.setCliArgsTemplate(
                    std::string("-m ") + modelPath +
                    " -nt -np -f {infile}"
                );

                // Run whisper and capture stdout
                transcript = g_whisper.transcribe_from_file(audio_path, 600);

                // Simple cleanup of transcript (remove potential system logs if they appear in stdout)
                // whisper.cpp usually prints system info to stderr, so stdout should be mostly text.
                
                std::cout << "[" << sess.video_id << "] Transcription complete: " 
                          << transcript.length() << " characters" << std::endl;
                if (!transcript.empty()) {
                     std::cout << "[" << sess.video_id << "] Transcript preview: " << transcript.substr(0, 50) << "..." << std::endl;
                }

                std::cout << "[" << sess.video_id << "] Transcription complete: " 
                          << transcript.length() << " characters" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[" << sess.video_id << "] Transcription failed: " 
                          << e.what() << std::endl;
                transcript.clear();
            }

            // Clean up audio file
            std::error_code ec;
            std::filesystem::remove(audio_path, ec);
            
            sess.progress_percent = 75;

            // 6) Index audio transcript (75-90%)
            sess.current_stage = "indexing_audio";
            struct Segment { float start; float end; std::string text; };
            std::vector<Segment> segments;
            float dur_s = (sess.duration_ms > 0) 
                ? static_cast<float>(sess.duration_ms) / 1000.0f 
                : 0.0f;
            
            segments.push_back({
                0.0f, 
                dur_s, 
                transcript.empty() ? "no transcript" : transcript
            });

            std::cout << "[" << sess.video_id << "] Indexing audio segments..." << std::endl;

            auto& clip = get_clip();
            for (const auto& seg : segments) {
                try {
                    auto emb = clip.encodeText(seg.text);
                    hnsw_index::add(emb, seg.start, seg.end, seg.text);
                    sess.indexed_audio_segments++;
                } catch (const std::exception& e) {
                    std::cerr << "[" << sess.video_id << "] Failed to encode audio segment: " 
                              << e.what() << std::endl;
                }
            }
        }

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
        std::vector<SearchResult> api_results;
        api_results.reserve(hnsw_results.size());
        for (const auto& r : hnsw_results) {
            api_results.push_back({r.id, r.start_time, r.end_time, r.caption, r.similarity});
        }
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

        // 3) convert to API model
        std::vector<SearchResult> api_results;
        api_results.reserve(hnsw_results.size());
        for (const auto& r : hnsw_results) {
            api_results.push_back({r.id, r.start_time, r.end_time, r.caption, r.similarity});
        }

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

            std::vector<SearchResult> api_results;
            api_results.reserve(hnsw_results.size());
            for (const auto& r : hnsw_results) {
                api_results.push_back({r.id, r.start_time, r.end_time, r.caption, r.similarity});
            }

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