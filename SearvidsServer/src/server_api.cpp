/*
 * Copyright (c) 2025 Hyeonjun Park (phjun7150@gmail.com)
 * All rights reserved.
 */

#include "server_api.h"

#ifdef DELETE
#undef DELETE
#endif

#include <crow/multipart.h>
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
#include <pqxx/pqxx>

namespace server_api {

std::mutex g_sessions_mtx;
std::unordered_map<std::string, std::shared_ptr<VideoSession>> g_sessions;
whisper_wrapper::WhisperWrapper g_whisper;

// Database connection string
const std::string DB_CONN_STR = "postgresql://searvids_user:searvids_pass@db:5432/searvids_db";

class ConnectionPool {
    std::mutex m_mutex;
    std::vector<std::shared_ptr<pqxx::connection>> m_pool;
    std::string m_conn_str;
    const size_t MAX_POOL_SIZE = 10;

public:
    ConnectionPool(const std::string& conn_str) : m_conn_str(conn_str) {
        m_pool.reserve(MAX_POOL_SIZE);
    }

    std::shared_ptr<pqxx::connection> get_connection() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_pool.empty()) {
            auto conn = m_pool.back();
            m_pool.pop_back();
            if (conn->is_open()) return conn;
        }
        return std::make_shared<pqxx::connection>(m_conn_str);
    }

    void return_connection(std::shared_ptr<pqxx::connection> conn) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_pool.size() < MAX_POOL_SIZE && conn->is_open()) {
            m_pool.push_back(conn);
        }
    }
};

ConnectionPool g_db_pool(DB_CONN_STR);

class PooledConnection {
    std::shared_ptr<pqxx::connection> conn;
public:
    PooledConnection() : conn(g_db_pool.get_connection()) {}
    ~PooledConnection() { g_db_pool.return_connection(conn); }
    pqxx::connection& get() { return *conn; }
    pqxx::connection* operator->() { return conn.get(); }
    bool is_open() const { return conn->is_open(); }
};

// Helper to check DB connection
bool check_db_connection() {
    try {
        PooledConnection C;
        if (C.is_open()) {
            return true;
        } else {
            return false;
        }
    } catch (const std::exception &e) {
        std::cerr << "DB Connection Error: " << e.what() << std::endl;
        return false;
    }
}

void init_db() {
    try {
        PooledConnection C;
        if (C.is_open()) {
            pqxx::work W(C.get());
            W.exec(R"(
                CREATE TABLE IF NOT EXISTS videos (
                    video_id TEXT PRIMARY KEY,
                    url TEXT UNIQUE NOT NULL,
                    platform TEXT,
                    index_path TEXT,
                    access_count INT DEFAULT 1,
                    last_accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    dataset_group VARCHAR(50),
                    duration_ms BIGINT
                );
            )");
            
            // Migration for existing tables
            try { W.exec("ALTER TABLE videos ADD COLUMN IF NOT EXISTS access_count INT DEFAULT 1;"); } catch (...) {}
            try { W.exec("ALTER TABLE videos ADD COLUMN IF NOT EXISTS last_accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;"); } catch (...) {}
            try { W.exec("ALTER TABLE videos ADD COLUMN IF NOT EXISTS duration_ms BIGINT;"); } catch (...) {}

            // Access Logs for Monthly Stats
            W.exec(R"(
                CREATE TABLE IF NOT EXISTS access_logs (
                    id SERIAL PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_access_logs_video_id ON access_logs(video_id);
                CREATE INDEX IF NOT EXISTS idx_access_logs_date ON access_logs(accessed_at);
            )");

            W.commit();
            std::cout << "Database initialized successfully." << std::endl;
        }
    } catch (const std::exception &e) {
        std::cerr << "DB Init Error: " << e.what() << std::endl;
    }
}

void update_video_stats(const std::string& video_id) {
    try {
        PooledConnection C;
        if (C.is_open()) {
            pqxx::work W(C.get());
            // Update total count
            W.exec_params(R"(
                UPDATE videos 
                SET access_count = access_count + 1, 
                    last_accessed_at = CURRENT_TIMESTAMP 
                WHERE video_id = $1
            )", video_id);
            
            // Insert log entry
            W.exec_params("INSERT INTO access_logs (video_id) VALUES ($1)", video_id);
            
            W.commit();
        }
    } catch (...) {}
}

void enforce_disk_cache_policy() {
    const int MAX_CACHE_SIZE = 5; // Keep only top 5 for testing
    try {
        PooledConnection C;
        if (C.is_open()) {
            pqxx::work W(C.get());
            
            // 1. Cleanup logs older than 30 days
            W.exec("DELETE FROM access_logs WHERE accessed_at < NOW() - INTERVAL '30 days'");

            // 2. Find videos to evict based on MONTHLY popularity (count in access_logs)
            // We select videos that have index_path, order by their log count DESC, and skip top N.
            pqxx::result R = W.exec_params(R"(
                SELECT v.video_id, v.index_path, COUNT(a.id) as monthly_count
                FROM videos v
                LEFT JOIN access_logs a ON v.video_id = a.video_id
                WHERE v.index_path IS NOT NULL
                GROUP BY v.video_id
                ORDER BY monthly_count DESC, v.last_accessed_at DESC 
                OFFSET $1
            )", MAX_CACHE_SIZE);

            for (const auto& row : R) {
                std::string vid = row[0].as<std::string>();
                std::string path = row[1].as<std::string>();
                
                // Delete files
                try {
                    if (std::filesystem::exists(path + ".index")) std::filesystem::remove(path + ".index");
                    if (std::filesystem::exists(path + ".meta")) std::filesystem::remove(path + ".meta");
                    std::cout << "[Cache Eviction] Removed " << vid << " (Low Monthly Rank)" << std::endl;
                } catch (...) {}

                // Update DB
                W.exec_params("UPDATE videos SET index_path = NULL WHERE video_id = $1", vid);
            }
            W.commit();
        }
    } catch (const std::exception& e) {
        std::cerr << "Cache Policy Error: " << e.what() << std::endl;
    }
}

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

// Lazy initialize SigLIP to avoid loading models during static initialization
static std::unique_ptr<siglip_onnx::SiglipOnnx> g_siglip_ptr;
static std::mutex g_siglip_mtx;

static siglip_onnx::SiglipOnnx& get_siglip() {
    std::lock_guard<std::mutex> lock(g_siglip_mtx);
    if (!g_siglip_ptr) {
        bool use_gpu = false;
        if (const char* env_p = std::getenv("USE_GPU")) {
            std::string env_s(env_p);
            std::transform(env_s.begin(), env_s.end(), env_s.begin(), ::tolower);
            if (env_s == "true" || env_s == "1") use_gpu = true;
        }
        std::cout << "[Server] Initializing SigLIP with GPU=" << (use_gpu ? "ON" : "OFF") << std::endl;

        g_siglip_ptr = std::make_unique<siglip_onnx::SiglipOnnx>(
            "/app/models/siglip_text/model.onnx",
            "/app/models/siglip_vision/model.onnx",
            use_gpu,           // device_gpu
            224             // image size
        );
    }
    return *g_siglip_ptr;
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
    if (sess.cancelled) return;
    sess.analyzing = true;
    sess.started_at = std::chrono::system_clock::now();
    sess.current_stage = "initializing";
    sess.progress_percent = 0;

    try {
        bool is_local = sess.source_url.rfind("local:", 0) == 0;

        if (!is_local && !downloader::is_valid_url(sess.source_url)) {
            throw std::runtime_error("Invalid URL");
        }

        // Try to get duration for chunk-based processing
        int64_t total_duration_sec = 0;
        if (!is_local) {
            std::cout << "[" << sess.video_id << "] Getting video duration..." << std::endl;
            total_duration_sec = downloader::get_duration(sess.source_url);
        }
        
        // If duration is available, use Chunk-Based Processing
        if (total_duration_sec > 0) {
            sess.duration_ms = total_duration_sec * 1000;
            std::cout << "[" << sess.video_id << "] Chunk-based analysis. Total duration: " << total_duration_sec << "s" << std::endl;
            
            const int CHUNK_SIZE = 180; // 3 minutes
            int current_start = 0;
            
            // Pipeline: Download N+1 while Analyzing N
            std::future<void> last_analysis_future;
            
            while (current_start < total_duration_sec) {
                if (sess.cancelled) throw std::runtime_error("Analysis cancelled");
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
                                auto& siglip = get_siglip();
                                
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
                                    // [NEW] Batch Pop
                                    std::vector<QueueItem> pop_batch(int limit) {
                                        std::unique_lock<std::mutex> lk(m);
                                        cv.wait(lk, [this]{ return !q.empty(); });
                                        std::vector<QueueItem> items;
                                        items.reserve(limit);
                                        while (!q.empty() && items.size() < limit) {
                                            items.push_back(std::move(q.front()));
                                            q.pop();
                                        }
                                        return items;
                                    }
                                } frame_queue;

                                auto consumer_thread = std::thread([&sess, &siglip, &frame_queue, current_start, total_duration_sec]() {
                                    while (true) {
                                        if (sess.cancelled) break;
                                        
                                        // 1. Get Batch (Size 32)
                                        auto items = frame_queue.pop_batch(32);
                                        if (items.empty()) break;

                                        std::vector<std::vector<uint8_t>> batch_rgb;
                                        std::vector<size_t> valid_indices;
                                        bool received_end = false;

                                        // 2. Filter valid frames
                                        for(size_t i=0; i<items.size(); ++i) {
                                            if (items[i].is_end) {
                                                received_end = true;
                                                // Don't break immediately, process what we have first? 
                                                // Actually if is_end is in batch, subsequent items shouldn't exist ideally.
                                                // But let's just stop collecting.
                                                break; 
                                            }
                                            batch_rgb.push_back(std::move(items[i].frame.rgb_data));
                                            valid_indices.push_back(i);
                                        }

                                        if (!batch_rgb.empty()) {
                                            try {
                                                // 3. Batch Inference
                                                // Use width/height from first frame
                                                int w = items[valid_indices[0]].frame.width;
                                                int h = items[valid_indices[0]].frame.height;
                                                
                                                auto results = siglip.encodeBatch(batch_rgb, w, h);

                                                // 4. Indexing
                                                for(size_t k=0; k<results.size(); ++k) {
                                                    const auto& frame = items[valid_indices[k]].frame;
                                                    const auto& emb = results[k];
                                                    int64_t real_timestamp_ms = frame.timestamp_ms + (current_start * 1000);
                                                    
                                                    hnsw_index::add(
                                                        emb,
                                                        sess.video_id,
                                                        static_cast<float>(real_timestamp_ms) / 1000.0f,
                                                        static_cast<float>(real_timestamp_ms) / 1000.0f + 1.0f,
                                                        "visual_frame"
                                                    );
                                                    sess.indexed_visual_frames++;
                                                    
                                                    // Update Progress (only occasionally)
                                                    if (k == results.size() - 1) { 
                                                        double progress = (double)real_timestamp_ms / (double)(total_duration_sec * 1000);
                                                        if (progress > 1.0) progress = 1.0;
                                                        int p = (int)(progress * 100.0);
                                                        if (p > sess.progress_percent) sess.progress_percent = p;
                                                        broadcast_status(sess.video_id, sess);
                                                    }
                                                }
                                            } catch (...) {}
                                        }

                                        if (received_end) break;
                                    }
                                });

                                ffmpeg_decoder::extract_frames_with_callback(
                                    chunk_path,
                                    [&frame_queue](const ffmpeg_decoder::FrameData& frame) {
                                        frame_queue.push({frame, false});
                                    },
                                    2.0, 0, 0, 0, 0, 0, ffmpeg_decoder::FrameExtractionMethod::SCENE_DETECT
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
                                
                                if (!ffmpeg_decoder::extract_audio(chunk_path, audio_path, 16000, 1, nullptr)) return;
                                
                                const char* w_url = std::getenv("WHISPER_URL");
                                if (w_url) {
                                    g_whisper.setServerUrl(w_url);
                                } else {
                                    g_whisper.setCliExecutable("python3");
                                    g_whisper.setCliArgsTemplate("/app/whisper_ct2.py {infile}");
                                }
                                
                                std::regex re(R"(\[(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s-->\s(\d{2}):(\d{2}):(\d{2})\.(\d{3})\]\s+(.*))");
                                auto& siglip = get_siglip();
                                std::string buffer;
                                
                                g_whisper.transcribe_with_callback(audio_path, 
                                    [&sess, &siglip, &re, &buffer, current_start, total_duration_sec](const std::string& chunk) {
                                        if (sess.cancelled) return;
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
                                                        auto emb = siglip.encodeText(text);
                                                        hnsw_index::add(emb, sess.video_id, start, end, text);
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
            if (!is_local) {
                auto filename = downloader::extract_filename(sess.source_url);
                if (filename.empty()) filename = "input.mp4";
                std::string out = std::string("data/") + sess.video_id + "_" + filename;

                std::cout << "[" << sess.video_id << "] Downloading: " << sess.source_url << std::endl;

                bool ok = downloader::download_with_retry(sess.source_url, out, 3);
                if (!ok) throw std::runtime_error("Download failed after 3 retries");
                
                sess.local_path = out;
                sess.progress_percent = 10;
                std::cout << "[" << sess.video_id << "] Download complete: " << out << std::endl;
            } else {
                // Local file, path is already set
                std::cout << "[" << sess.video_id << "] Using local file: " << sess.local_path << std::endl;
                sess.progress_percent = 10;
            }

            // 2) Probe video metadata (10-15%)
            sess.current_stage = "probing";
            std::cout << "[" << sess.video_id << "] Probing video metadata..." << std::endl;
            
            auto info = ffmpeg_decoder::probe(sess.local_path);
            
            if (!info.has_video && !info.has_audio) {
                throw std::runtime_error("No valid media streams found");
            }
            
            sess.duration_ms = info.duration_ms;
            sess.nb_frames = (info.nb_frames > 0) ? info.nb_frames : ffmpeg_decoder::count_frames(sess.local_path);
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
                    auto& siglip = get_siglip();
                    
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

                        // [NEW] Batch Pop
                        std::vector<QueueItem> pop_batch(int limit) {
                            std::unique_lock<std::mutex> lk(m);
                            cv.wait(lk, [this]{ return !q.empty(); });
                            std::vector<QueueItem> items;
                            items.reserve(limit);
                            while (!q.empty() && items.size() < limit) {
                                items.push_back(std::move(q.front()));
                                q.pop();
                            }
                            return items;
                        }
                    } frame_queue;

                    // Consumer Thread: CLIP Inference & Indexing
                    auto consumer_thread = std::thread([&sess, &siglip, &frame_queue]() {
                        while (true) {
                            if (sess.cancelled) break;
                            
                            // 1. Get Batch
                            auto items = frame_queue.pop_batch(32);
                            if (items.empty()) break;

                            std::vector<std::vector<uint8_t>> batch_rgb;
                            std::vector<size_t> valid_indices;
                            bool received_end = false;

                            for(size_t i=0; i<items.size(); ++i) {
                                if (items[i].is_end) {
                                    received_end = true;
                                    break;
                                }
                                batch_rgb.push_back(std::move(items[i].frame.rgb_data));
                                valid_indices.push_back(i);
                            }

                            if (!batch_rgb.empty()) {
                                try {
                                    int w = items[valid_indices[0]].frame.width;
                                    int h = items[valid_indices[0]].frame.height;
                                    
                                    auto results = siglip.encodeBatch(batch_rgb, w, h);

                                    for(size_t k=0; k<results.size(); ++k) {
                                        const auto& frame = items[valid_indices[k]].frame;
                                        const auto& emb = results[k];
                                        
                                        hnsw_index::add(
                                            emb,
                                            sess.video_id,
                                            static_cast<float>(frame.timestamp_ms) / 1000.0f,
                                            static_cast<float>(frame.timestamp_ms) / 1000.0f + 1.0f,
                                            "visual_frame"
                                        );
                                        sess.indexed_visual_frames++;
                                        
                                        // Update progress
                                        if (sess.nb_frames > 0 && k == results.size() - 1) {
                                            double progress = (double)frame.timestamp_ms / (double)sess.duration_ms;
                                            if (progress > 1.0) progress = 1.0;
                                            int p = 15 + (int)(30.0 * progress);
                                            if (p > sess.progress_percent) sess.progress_percent = p;
                                            broadcast_status(sess.video_id, sess);
                                        }
                                    }
                                } catch (const std::exception& e) {
                                    std::cerr << "[" << sess.video_id << "] CLIP inference failed: " << e.what() << std::endl;
                                }
                            }
                            
                            if (received_end) break;
                        }
                    });

                    // Producer: FFmpeg Decoding
                    ffmpeg_decoder::extract_frames_with_callback(
                        sess.local_path,
                        [&frame_queue](const ffmpeg_decoder::FrameData& frame) {
                            frame_queue.push({frame, false});
                        },
                        2.0, 0, 0, 0, 224, 224, ffmpeg_decoder::FrameExtractionMethod::SCENE_DETECT
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
                    const char* w_url = std::getenv("WHISPER_URL");
                    if (w_url) {
                        g_whisper.setServerUrl(w_url);
                    } else {
                        g_whisper.setCliExecutable("python3");
                        g_whisper.setCliArgsTemplate("/app/whisper_ct2.py {infile}");
                    }
                    
                    // Parse and Index incrementally
                    std::regex re(R"(\[(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s-->\s(\d{2}):(\d{2}):(\d{2})\.(\d{3})\]\s+(.*))");
                    auto& siglip = get_siglip();
                    std::string buffer;
                    
                    g_whisper.transcribe_with_callback(audio_path, 
                        [&sess, &siglip, &re, &buffer](const std::string& chunk) {
                            if (sess.cancelled) return;
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
                                        // trim
                                        text.erase(0, text.find_first_not_of(" \t"));
                                        text.erase(text.find_last_not_of(" \t") + 1);
                                        
                                        if (!text.empty()) {
                                            auto emb = siglip.encodeText(text);
                                            hnsw_index::add(emb, sess.video_id, start, end, text);
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

        // --- DB SAVE START ---
        try {
            // Ensure data directory exists
            std::filesystem::create_directories("data");
            std::string index_path = "data/" + sess.video_id;
            
            // Save HNSW index to disk
            hnsw_index::save(index_path);
            
            PooledConnection C;
            if (C.is_open()) {
                pqxx::work W(C.get());
                // Upsert (Insert or Update)
                W.exec_params(R"(
                    INSERT INTO videos (video_id, url, index_path) 
                    VALUES ($1, $2, $3)
                    ON CONFLICT (video_id) DO UPDATE SET index_path = EXCLUDED.index_path;
                )", sess.video_id, sess.source_url, index_path);
                W.commit();
                std::cout << "[" << sess.video_id << "] Saved to DB cache." << std::endl;
                
                // Update stats and enforce policy
                update_video_stats(sess.video_id);
                enforce_disk_cache_policy();
            }
        } catch (const std::exception& e) {
            std::cerr << "DB Save Error: " << e.what() << std::endl;
        }
        // --- DB SAVE END ---

        // Cleanup downloaded video file
        if (!sess.local_path.empty() && sess.source_url.find("http") == 0 && std::filesystem::exists(sess.local_path)) {
             std::filesystem::remove(sess.local_path);
             std::cout << "[" << sess.video_id << "] Removed temporary video file." << std::endl;
        }

    } catch (const std::exception& e) {
        sess.error = e.what();
        sess.current_stage = "failed";
        sess.completed_at = std::chrono::system_clock::now();
        
        broadcast_status(sess.video_id, sess);
        
        std::cerr << "[" << sess.video_id << "] Analysis failed: " 
                  << e.what() << std::endl;

        // Cleanup downloaded video file on error
        if (!sess.local_path.empty() && sess.source_url.find("http") == 0 && std::filesystem::exists(sess.local_path)) {
             std::filesystem::remove(sess.local_path);
        }
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
        if (j.contains("video_id")) req.video_id = j["video_id"];
    } catch (...) {}
    return req;
}

crow::response create_search_response(const std::vector<SearchResult>& results) {
    nlohmann::json j_results = nlohmann::json::array();
    for (const auto& r : results) {
        nlohmann::json item;
        item["id"] = r.id;
        item["start_time"] = r.start_time;
        item["end_time"] = r.end_time;
        item["caption"] = r.caption;
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
            // Lower thresholds for SigLIP
            // 0.02 was too high for some queries (e.g. "green snake" -> 0.013)
            float threshold = is_visual ? 0.01f : 0.15f;
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
    CROW_WEBSOCKET_ROUTE(app, "/api/ws/videos/<string>/status")
    .onopen([&](crow::websocket::connection& conn) {
        // Crow doesn't easily pass route args to onopen, so we wait for a "subscribe" message.
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

        // --- DB CACHE CHECK START ---
        try {
            pqxx::connection C(DB_CONN_STR);
            if (C.is_open()) {
                pqxx::work W(C);
                pqxx::result R = W.exec_params("SELECT index_path FROM videos WHERE url = $1", url);
                if (!R.empty()) {
                    std::string index_path = R[0][0].as<std::string>();
                    if (std::filesystem::exists(index_path + ".index")) {
                        std::cout << "[Cache Hit] Loading index for " << vid << std::endl;
                        hnsw_index::load(index_path);
                        
                        auto sess = std::make_shared<VideoSession>();
                        sess->video_id = vid;
                        sess->source_url = url;
                        sess->done = true;
                        sess->progress_percent = 100;
                        sess->current_stage = "completed (cached)";
                        
                        std::lock_guard<std::mutex> lk(g_sessions_mtx);
                        g_sessions[vid] = sess;
                        
                        // Update stats and enforce policy
                        std::thread([vid](){
                            update_video_stats(vid);
                            enforce_disk_cache_policy();
                        }).detach();

                        nlohmann::json resp = nlohmann::json::object();
                        resp["status"] = "cached";
                        resp["video_id"] = vid;
                        return json_ok(resp, 200);
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "DB Cache Check Error: " << e.what() << std::endl;
        }
        // --- DB CACHE CHECK END ---

        {
            std::lock_guard<std::mutex> lk(g_sessions_mtx);
            
            // Cancel other sessions
            for (auto& kv : g_sessions) {
                if (kv.first != vid && kv.second->analyzing) {
                    kv.second->cancelled = true;
                }
            }

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
                     sess->cancelled = false;
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

        // --- DB CACHE CHECK START ---
        try {
            pqxx::connection C(DB_CONN_STR);
            if (C.is_open()) {
                pqxx::work W(C);
                pqxx::result R = W.exec_params("SELECT index_path FROM videos WHERE url = $1", url);
                if (!R.empty()) {
                    std::string index_path = R[0][0].as<std::string>();
                    if (std::filesystem::exists(index_path + ".index")) {
                        std::cout << "[Cache Hit] Loading index for " << vid << std::endl;
                        hnsw_index::load(index_path);
                        
                        // Create a fake completed session for status checks
                        auto sess = std::make_shared<VideoSession>();
                        sess->video_id = vid;
                        sess->source_url = url;
                        sess->done = true;
                        sess->progress_percent = 100;
                        sess->current_stage = "completed (cached)";
                        
                        std::lock_guard<std::mutex> lk(g_sessions_mtx);
                        g_sessions[vid] = sess;
                        
                        // Update stats and enforce policy
                        std::thread([vid](){
                            update_video_stats(vid);
                            enforce_disk_cache_policy();
                        }).detach();

                        nlohmann::json resp = nlohmann::json::object();
                        resp["status"] = "cached";
                        resp["video_id"] = vid;
                        return json_ok(resp, 200);
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "DB Cache Check Error: " << e.what() << std::endl;
        }
        // --- DB CACHE CHECK END ---

        {
            std::lock_guard<std::mutex> lk(g_sessions_mtx);

            // Cancel other sessions
            for (auto& kv : g_sessions) {
                if (kv.first != vid && kv.second->analyzing) {
                    kv.second->cancelled = true;
                }
            }

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
                     sess->cancelled = false;
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

        // --- DB CACHE CHECK START ---
        try {
            pqxx::connection C(DB_CONN_STR);
            if (C.is_open()) {
                pqxx::work W(C);
                pqxx::result R = W.exec_params("SELECT index_path, duration_ms FROM videos WHERE url = $1", url);
                if (!R.empty()) {
                    std::string index_path = R[0][0].as<std::string>();
                    int64_t dur = R[0][1].as<int64_t>(0);

                    if (std::filesystem::exists(index_path + ".index")) {
                        std::cout << "[Cache Hit] Loading index for " << vid << std::endl;
                        hnsw_index::load(index_path);
                        
                        auto sess = std::make_shared<VideoSession>();
                        sess->video_id = vid;
                        sess->source_url = url;
                        sess->duration_ms = dur;
                        sess->done = true;
                        sess->progress_percent = 100;
                        sess->current_stage = "completed (cached)";
                        
                        std::lock_guard<std::mutex> lk(g_sessions_mtx);
                        g_sessions[vid] = sess;
                        
                        // Update stats and enforce policy
                        std::thread([vid](){
                            update_video_stats(vid);
                            enforce_disk_cache_policy();
                        }).detach();

                        nlohmann::json resp = nlohmann::json::object();
                        resp["status"] = "cached";
                        resp["video_id"] = vid;
                        return json_ok(resp, 200);
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "DB Cache Check Error: " << e.what() << std::endl;
        }
        // --- DB CACHE CHECK END ---

        {
            std::lock_guard<std::mutex> lk(g_sessions_mtx);

            // Cancel other sessions
            for (auto& kv : g_sessions) {
                if (kv.first != vid && kv.second->analyzing) {
                    kv.second->cancelled = true;
                }
            }

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
                     sess->cancelled = false;
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

        // Handle chunked videos
        int64_t actual_ts = ts;
        if (path.empty() || !std::filesystem::exists(path)) {
            int chunk_size = 180; // 3 minutes
            int chunk_start = (ts / 1000 / chunk_size) * chunk_size;
            std::string chunk_path = "data/" + video_id + "_chunk_" + std::to_string(chunk_start) + ".mp4";
            if (std::filesystem::exists(chunk_path)) {
                path = chunk_path;
                actual_ts = ts % (chunk_size * 1000);
            }
        }

        try {
            // Extract frame (resize to 256x144 for thumbnail, optimized size)
            auto frame = ffmpeg_decoder::extract_frame_at(path, actual_ts, true, 256, 144);
            
            // Convert to OpenCV Mat
            cv::Mat img(frame.height, frame.width, CV_8UC3, frame.rgb_data.data());
            cv::cvtColor(img, img, cv::COLOR_RGB2BGR); // OpenCV uses BGR

            // Encode to JPEG with 70% quality
            std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 70};
            std::vector<uchar> buf;
            cv::imencode(".jpg", img, buf, params);

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

        // Handle chunked videos
        int64_t actual_ts = ts;
        if (path.empty() || !std::filesystem::exists(path)) {
            int chunk_size = 180; // 3 minutes
            int chunk_start = (ts / 1000 / chunk_size) * chunk_size;
            std::string chunk_path = "data/" + video_id + "_chunk_" + std::to_string(chunk_start) + ".mp4";
            if (std::filesystem::exists(chunk_path)) {
                path = chunk_path;
                actual_ts = ts % (chunk_size * 1000);
            }
        }

        try {
            auto frame = ffmpeg_decoder::extract_frame_at(path, actual_ts, true, 256, 144);
            cv::Mat img(frame.height, frame.width, CV_8UC3, frame.rgb_data.data());
            cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
            
            std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 70};
            std::vector<uchar> buf;
            cv::imencode(".jpg", img, buf, params);
            
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
                emb = get_siglip().encodeText(sreq.query);
            } catch (...) {
                return json_err(500, "SigLIP encoding failed");
            }
        }

        // 2) HNSW search
        auto hnsw_results = hnsw_index::search(emb, sreq.topk, sreq.video_id);

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
                    emb = get_siglip().encodeText(sreq.query);
                } catch (const std::exception& e) {
                    return json_err(500, std::string("SigLIP encoding failed: ") + e.what());
                }
            } else {
                emb = sreq.embedding;
            }

            std::vector<hnsw_index::TimelineEntry> hnsw_results;
            
            if (sreq.search_type == "visual") {
                hnsw_results = hnsw_index::search(emb, sreq.topk, sreq.video_id, "visual");
            } else if (sreq.search_type == "audio") {
                hnsw_results = hnsw_index::search(emb, sreq.topk, sreq.video_id, "audio");
            } else {
                // Both: search separately to ensure we get results from both modalities
                // This prevents audio results from drowning out visual results if their score distributions differ
                auto visual_results = hnsw_index::search(emb, sreq.topk, sreq.video_id, "visual");
                auto audio_results = hnsw_index::search(emb, sreq.topk, sreq.video_id, "audio");
                
                hnsw_results.insert(hnsw_results.end(), visual_results.begin(), visual_results.end());
                hnsw_results.insert(hnsw_results.end(), audio_results.begin(), audio_results.end());
                
                // Sort by similarity descending
                std::sort(hnsw_results.begin(), hnsw_results.end(), [](const auto& a, const auto& b) {
                    return a.similarity > b.similarity;
                });
            }

            auto api_results = filter_results(hnsw_results, sreq.query);
            return create_search_response(api_results);
        } catch (const std::exception& e) {
            return json_err(500, std::string("Search failed: ") + e.what());
        }
    });

    // POST /api/upload
    CROW_ROUTE(app, "/api/upload").methods(crow::HTTPMethod::POST)
    ([](const crow::request& req) {
        crow::multipart::message msg(req);
        std::string filename;
        std::string content;
        
        for (const auto& part : msg.parts) {
            if (part.headers.count("Content-Disposition")) {
                auto it = part.headers.find("Content-Disposition");
                auto param_it = it->second.params.find("filename");
                if (param_it != it->second.params.end()) {
                    filename = param_it->second;
                    content = part.body;
                    break;
                }
            }
        }

        if (content.empty()) return json_err(400, "No file uploaded");

        auto now = std::chrono::system_clock::now().time_since_epoch().count();
        std::string vid = "local_" + std::to_string(now);
        
        std::string path = "data/" + vid + ".mp4";
        std::ofstream out(path, std::ios::binary);
        out.write(content.data(), content.size());
        out.close();

        {
            std::lock_guard<std::mutex> lk(g_sessions_mtx);
            for (auto& kv : g_sessions) {
                if (kv.first != vid && kv.second->analyzing) {
                    kv.second->cancelled = true;
                }
            }

            auto sess = std::make_shared<VideoSession>();
            sess->video_id = vid;
            sess->source_url = "local:" + filename;
            sess->local_path = path;
            g_sessions[vid] = sess;
            
            std::thread([sess]() {
                analyze_video_async(sess);
            }).detach();
        }

        nlohmann::json resp = nlohmann::json::object();
        resp["status"] = "accepted";
        resp["video_id"] = vid;
        return json_ok(resp, 202);
    });

    // DELETE /api/videos/<video_id>
    CROW_ROUTE(app, "/api/videos/<string>").methods(crow::HTTPMethod::DELETE)
    ([](const std::string& video_id) {
        std::lock_guard<std::mutex> lk(g_sessions_mtx);
        auto it = g_sessions.find(video_id);
        if (it != g_sessions.end()) {
            it->second->cancelled = true;
            if (!it->second->local_path.empty() && std::filesystem::exists(it->second->local_path)) {
                std::filesystem::remove(it->second->local_path);
            }
            g_sessions.erase(it);
        }
        
        hnsw_index::remove_video(video_id);
        
        for (const auto& entry : std::filesystem::directory_iterator("data")) {
            if (entry.path().filename().string().find(video_id) != std::string::npos) {
                std::filesystem::remove(entry.path());
            }
        }

        return crow::response(200);
    });

    // GET /api/health
    CROW_ROUTE(app, "/api/health").methods("GET"_method)([] {
        bool db_ok = check_db_connection();
        nlohmann::json j;
        j["status"] = "OK";
        j["db_connected"] = db_ok;
        return json_ok(j);
    });

    // DEBUG: Text Similarity
    CROW_ROUTE(app, "/api/debug/text_sim").methods(crow::HTTPMethod::POST)
    ([](const crow::request& req) {
        auto j = nlohmann::json::parse(req.body);
        std::string t1 = j["text1"];
        std::string t2 = j["text2"];
        
        auto& siglip = get_siglip();
        auto e1 = siglip.encodeText(t1);
        auto e2 = siglip.encodeText(t2);
        
        float sim = siglip_onnx::SiglipOnnx::cosineSimilarity(e1, e2);
        
        nlohmann::json resp;
        resp["similarity"] = sim;
        return json_ok(resp);
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