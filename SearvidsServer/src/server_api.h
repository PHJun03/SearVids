#pragma once
#include <crow/app.h>
#include <crow/http_response.h>
#include <crow/routing.h>
#include <string>
#include <vector>

namespace server_api {

struct SearchRequest {
    std::vector<float> embedding;
    int topk = 5;
};

struct SearchResult {
    int id;
    float start_time;
    float end_time;
    std::string caption;
    float similarity;  // cosine similarity score
};

/**
 * Initialize API routes
 * @param app: Crow application instance
 */
void setup_routes(crow::SimpleApp& app);

/**
 * Parse JSON embedding from request
 * @param body: request body JSON string
 * @return SearchRequest object
 */
SearchRequest parse_search_request(const std::string& body);

/**
 * Convert SearchResult to JSON
 * @param results: vector of search results
 * @return JSON response string
 */
crow::response create_search_response(const std::vector<SearchResult>& results);

/**
 * Health check endpoint
 * @return health status JSON
 */
crow::response health_check();

} // namespace server_api