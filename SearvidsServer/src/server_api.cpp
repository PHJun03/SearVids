#include "server_api.h"
#include <nlohmann/json.hpp>
#include <thread>
#include <functional>
#include <memory>

namespace server_api {

// Define global state
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
            "onnx/onnx_models/clip_text_sim.onnx",
            "onnx/onnx_models/clip_vision_sim.onnx",
            false,  // CPU mode
            224     // image size
        );
    }
    return *g_clip_ptr;
}

std::string make_video_id(const std::string& url) {
    std::hash<std::string> h;
    return std::to_string(h(url));
}

void analyze_video_async(VideoSession& sess) {
    sess.analyzing = true;
    try {
        // 1) Download video
        if (!downloader::is_valid_url(sess.source_url)) {
            throw std::runtime_error("Invalid URL");
        }
        auto filename = downloader::extract_filename(sess.source_url);
        if (filename.empty()) filename = "input.mp4";
        
        std::string out = std::string("data/") + sess.video_id + "_" + filename;
        downloader::download(sess.source_url, out);
        sess.local_path = out;

        // 2) Extract video metadata
        auto info = ffmpeg_decoder::probe(out);
        sess.duration_ms = info.duration_ms;
        sess.nb_frames = (info.nb_frames > 0) ? info.nb_frames : ffmpeg_decoder::count_frames(out);

        // 3) Transcribe with Whisper
        std::string transcript = g_whisper.transcribe_from_file(out);
        
        // Create single segment (TODO: parse timestamps from whisper output)
        struct Segment {
            float start;
            float end;
            std::string text;
        };
        std::vector<Segment> segments;
        segments.push_back({
            0.0f, 
            static_cast<float>(sess.duration_ms) / 1000.0f, 
            transcript
        });

        // 4) Generate embeddings and index
        for (const auto& seg : segments) {
            auto emb = get_clip().encodeText(seg.text);  // Use get_clip() instead of g_clip
            hnsw_index::add(emb, seg.start, seg.end, seg.text);
        }

        sess.done = true;
    } catch (const std::exception& e) {
        sess.error = e.what();
    }
    sess.analyzing = false;
}

void setup_routes(crow::SimpleApp& app) {
    // POST /videos/analyze
    CROW_ROUTE(app, "/videos/analyze")
        .methods(crow::HTTPMethod::POST)
        ([](const crow::request& req) {
            nlohmann::json body;
            try { 
                body = nlohmann::json::parse(req.body); 
            } catch (...) { 
                body = nlohmann::json::object(); 
            }
            
            std::string url = body.value("url", "");
            if (url.empty()) {
                return crow::response(400, R"({"status":"error","message":"url is required"})");
            }

            auto vid = make_video_id(url);
            {
                std::lock_guard<std::mutex> lk(g_sessions_mtx);
                auto& sess = g_sessions[vid];
                if (sess.video_id.empty()) {
                    sess.video_id = vid;
                    sess.source_url = url;
                    std::thread([&sess]() { analyze_video_async(sess); }).detach();
                }
            }
            
            nlohmann::json res{{"status", "accepted"}, {"video_id", vid}};
            return crow::response(202, res.dump());
        });

    // GET /videos/{video_id}/status
    CROW_ROUTE(app, "/videos/<string>/status")
        .methods(crow::HTTPMethod::GET)
        ([](const std::string& video_id) {
            std::lock_guard<std::mutex> lk(g_sessions_mtx);
            auto it = g_sessions.find(video_id);
            if (it == g_sessions.end()) {
                return crow::response(404, R"({"status":"error","message":"not found"})");
            }
            
            const auto& s = it->second;
            std::string status = s.error.empty() 
                ? (s.done ? "done" : (s.analyzing ? "analyzing" : "pending"))
                : "error";
            
            nlohmann::json res{
                {"status", status},
                {"error", s.error},
                {"duration_ms", s.duration_ms},
                {"nb_frames", s.nb_frames}
            };
            return crow::response(200, res.dump());
        });

    // POST /videos/{video_id}/search
    CROW_ROUTE(app, "/videos/<string>/search")
        .methods(crow::HTTPMethod::POST)
        ([](const crow::request& req, const std::string& video_id) {
            // Check video exists
            {
                std::lock_guard<std::mutex> lk(g_sessions_mtx);
                if (g_sessions.find(video_id) == g_sessions.end()) {
                    return crow::response(404, R"({"status":"error","message":"video not found"})");
                }
            }

            // Parse search request
            auto search_req = parse_search_request(req.body);
            std::vector<float> query_emb;

            if (!search_req.embedding.empty()) {
                query_emb = search_req.embedding;
            } else if (!search_req.query.empty()) {
                try {
                    query_emb = get_clip().encodeText(search_req.query);  // Use get_clip() instead of g_clip
                } catch (const std::exception& e) {
                    return crow::response(500, R"({"status":"error","message":"CLIP encoding failed"})");
                }
            } else {
                return crow::response(400, R"({"status":"error","message":"query or embedding required"})");
            }

            // Search index
            auto hnsw_results = hnsw_index::search(query_emb, search_req.topk);
            
            // Convert to API result format
            std::vector<SearchResult> api_results;
            for (const auto& r : hnsw_results) {
                api_results.push_back({
                    r.id, 
                    r.start_time, 
                    r.end_time, 
                    r.caption, 
                    r.similarity
                });
            }
            
            return create_search_response(api_results);
        });

    // GET /health
    CROW_ROUTE(app, "/health")([]() {
        return health_check();
    });
}

SearchRequest parse_search_request(const std::string& body) {
    SearchRequest req;
    try {
        auto json = nlohmann::json::parse(body);
        req.query = json.value("query", "");
        req.topk = json.value("topk", 5);
        if (json.contains("embedding") && json["embedding"].is_array()) {
            req.embedding = json["embedding"].get<std::vector<float>>();
        }
    } catch (...) {
        // Return default
    }
    return req;
}

crow::response create_search_response(const std::vector<SearchResult>& results) {
    nlohmann::json json_results = nlohmann::json::array();
    for (const auto& r : results) {
        json_results.push_back({
            {"id", r.id},
            {"start_time", r.start_time},
            {"end_time", r.end_time},
            {"caption", r.caption},
            {"similarity", r.similarity}
        });
    }
    nlohmann::json response{{"status", "ok"}, {"results", json_results}};
    return crow::response(200, response.dump());
}

crow::response health_check() {
    nlohmann::json json{{"status", "ok"}};
    return crow::response(200, json.dump());
}

} // namespace server_api