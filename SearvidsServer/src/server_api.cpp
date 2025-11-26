#include "server_api.h"
#include "hnsw_index.h"
#include "downloader.h"
#include "ffmpeg_decoder.h"
#include "clip_onnx.h"
#include "whisper_wrapper.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <sstream>

using json = nlohmann::json;

namespace server_api {

SearchRequest parse_search_request(const std::string& body) {
    SearchRequest req;
    try {
        auto j = json::parse(body);
        
        if (j.contains("embedding") && j["embedding"].is_array()) {
            req.embedding = j["embedding"].get<std::vector<float>>();
        }
        
        if (j.contains("topk") && j["topk"].is_number_integer()) {
            req.topk = j["topk"].get<int>();
        }
        
    } catch (const json::exception& e) {
        std::cerr << "JSON parse error: " << e.what() << std::endl;
    }
    
    return req;
}

crow::response create_search_response(const std::vector<SearchResult>& results) {
    json response;
    response["status"] = "success";
    response["count"] = results.size();
    
    json results_json = json::array();
    for (const auto& result : results) {
        json item;
        item["id"] = result.id;
        item["start_time"] = result.start_time;
        item["end_time"] = result.end_time;
        item["caption"] = result.caption;
        item["similarity"] = result.similarity;
        results_json.push_back(item);
    }
    
    response["results"] = results_json;
    
    auto res = crow::response(response.dump());
    res.add_header("Content-Type", "application/json");
    return res;
}

crow::response health_check() {
    json response;
    response["status"] = "ok";
    response["service"] = "SearvidsServer";
    response["index_size"] = static_cast<int>(hnsw_index::size());
    
    auto res = crow::response(response.dump());
    res.add_header("Content-Type", "application/json");
    return res;
}

void setup_routes(crow::SimpleApp& app) {
    
    // Health check endpoint
    CROW_ROUTE(app, "/health")
    .methods("GET"_method)
    ([](const crow::request&) {
        return health_check();
    });
    
    // Search endpoint — search by embedding vector
    CROW_ROUTE(app, "/search")
    .methods("POST"_method)
    ([](const crow::request& req) {
        try {
            auto search_req = parse_search_request(req.body);
            
            if (search_req.embedding.empty()) {
                json error;
                error["status"] = "error";
                error["message"] = "embedding vector is required";
                auto res = crow::response(400, error.dump());
                res.add_header("Content-Type", "application/json");
                return res;
            }
            
            // Perform search using hnsw_index
            auto entries = hnsw_index::search(search_req.embedding, search_req.topk);
            
            // Convert to SearchResult (with dummy similarity for now)
            std::vector<SearchResult> results;
            for (const auto& entry : entries) {
                SearchResult sr;
                sr.id = entry.id;
                sr.start_time = entry.start_time;
                sr.end_time = entry.end_time;
                sr.caption = entry.caption;
                sr.similarity = 0.0f;  // TODO: compute actual cosine similarity
                results.push_back(sr);
            }
            
            return create_search_response(results);
            
        } catch (const std::exception& e) {
            json error;
            error["status"] = "error";
            error["message"] = e.what();
            auto res = crow::response(500, error.dump());
            res.add_header("Content-Type", "application/json");
            return res;
        }
    });
    
    // Index info endpoint
    CROW_ROUTE(app, "/index/info")
    .methods("GET"_method)
    ([](const crow::request&) {
        json response;
        response["status"] = "success";
        response["index_size"] = static_cast<int>(hnsw_index::size());
        
        auto res = crow::response(response.dump());
        res.add_header("Content-Type", "application/json");
        return res;
    });
    
    // Download video endpoint
    CROW_ROUTE(app, "/download")
    .methods("POST"_method)
    ([](const crow::request& req) {
        try {
            auto j = json::parse(req.body);
            
            if (!j.contains("url")) {
                json error;
                error["status"] = "error";
                error["message"] = "url is required";
                auto res = crow::response(400, error.dump());
                res.add_header("Content-Type", "application/json");
                return res;
            }
            
            std::string url = j["url"].get<std::string>();
            std::string output_path = j.contains("output_path") ? 
                j["output_path"].get<std::string>() : "./downloads/video.mp4";
            
            // Validate URL
            if (!downloader::is_valid_url(url)) {
                json error;
                error["status"] = "error";
                error["message"] = "invalid URL";
                auto res = crow::response(400, error.dump());
                res.add_header("Content-Type", "application/json");
                return res;
            }
            
            // Start download (with retry)
            bool success = downloader::download_with_retry(url, output_path, 3);
            
            json response;
            response["status"] = success ? "success" : "failed";
            response["url"] = url;
            response["output_path"] = output_path;
            
            auto res = crow::response(success ? 200 : 500, response.dump());
            res.add_header("Content-Type", "application/json");
            return res;
            
        } catch (const std::exception& e) {
            json error;
            error["status"] = "error";
            error["message"] = e.what();
            auto res = crow::response(500, error.dump());
            res.add_header("Content-Type", "application/json");
            return res;
        }
    });
    
    // Add entry endpoint — add embedding to index
    CROW_ROUTE(app, "/index/add")
    .methods("POST"_method)
    ([](const crow::request& req) {
        try {
            auto j = json::parse(req.body);
            
            if (!j.contains("embedding") || !j.contains("start_time") || 
                !j.contains("end_time") || !j.contains("caption")) {
                json error;
                error["status"] = "error";
                error["message"] = "embedding, start_time, end_time, caption are required";
                auto res = crow::response(400, error.dump());
                res.add_header("Content-Type", "application/json");
                return res;
            }
            
            auto embedding = j["embedding"].get<std::vector<float>>();
            float start_time = j["start_time"].get<float>();
            float end_time = j["end_time"].get<float>();
            std::string caption = j["caption"].get<std::string>();
            
            int id = hnsw_index::add(embedding, start_time, end_time, caption);
            
            json response;
            response["status"] = "success";
            response["id"] = id;
            response["index_size"] = static_cast<int>(hnsw_index::size());
            
            auto res = crow::response(response.dump());
            res.add_header("Content-Type", "application/json");
            return res;
            
        } catch (const std::exception& e) {
            json error;
            error["status"] = "error";
            error["message"] = e.what();
            auto res = crow::response(500, error.dump());
            res.add_header("Content-Type", "application/json");
            return res;
        }
    });
    
    std::cout << "✅ API routes setup complete" << std::endl;
}

} // namespace server_api