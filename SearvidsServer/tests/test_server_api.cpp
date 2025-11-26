#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "server_api.h"
#include "hnsw_index.h"

using json = nlohmann::json;

class ServerApiTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize HNSW index before each test
        hnsw_index::create(128, "cosine");
    }

    void TearDown() override {
        // Cleanup after each test
    }

    // Helper: create random embedding
    std::vector<float> randomEmbedding(int dim) {
        std::vector<float> emb(dim);
        for (int i = 0; i < dim; ++i) {
            emb[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        return emb;
    }

    // Helper: normalize embedding
    std::vector<float> normalize(const std::vector<float>& emb) {
        float norm = 0.0f;
        for (float v : emb) norm += v * v;
        norm = std::sqrt(norm);
        std::vector<float> result = emb;
        if (norm > 1e-6f) {
            for (float& v : result) v /= norm;
        }
        return result;
    }
};

// Test 1: Parse search request — valid JSON
TEST_F(ServerApiTest, ParseSearchRequestValid) {
    std::string body = R"({
        "embedding": [0.1, 0.2, 0.3],
        "topk": 10
    })";
    
    auto req = server_api::parse_search_request(body);
    
    EXPECT_EQ(req.embedding.size(), 3);
    EXPECT_FLOAT_EQ(req.embedding[0], 0.1f);
    EXPECT_FLOAT_EQ(req.embedding[1], 0.2f);
    EXPECT_FLOAT_EQ(req.embedding[2], 0.3f);
    EXPECT_EQ(req.topk, 10);
}

// Test 2: Parse search request — default topk
TEST_F(ServerApiTest, ParseSearchRequestDefaultTopk) {
    std::string body = R"({
        "embedding": [0.1, 0.2, 0.3]
    })";
    
    auto req = server_api::parse_search_request(body);
    
    EXPECT_EQ(req.embedding.size(), 3);
    EXPECT_EQ(req.topk, 5);  // default value
}

// Test 3: Parse search request — empty embedding
TEST_F(ServerApiTest, ParseSearchRequestEmptyEmbedding) {
    std::string body = R"({
        "embedding": [],
        "topk": 5
    })";
    
    auto req = server_api::parse_search_request(body);
    
    EXPECT_EQ(req.embedding.size(), 0);
}

// Test 4: Parse search request — invalid JSON
TEST_F(ServerApiTest, ParseSearchRequestInvalidJson) {
    std::string body = "{ invalid json }";
    
    auto req = server_api::parse_search_request(body);
    
    EXPECT_EQ(req.embedding.size(), 0);
    EXPECT_EQ(req.topk, 5);  // default value
}

// Test 5: Parse search request — missing embedding
TEST_F(ServerApiTest, ParseSearchRequestMissingEmbedding) {
    std::string body = R"({
        "topk": 3
    })";
    
    auto req = server_api::parse_search_request(body);
    
    EXPECT_EQ(req.embedding.size(), 0);
    EXPECT_EQ(req.topk, 3);
}

// Test 6: Create search response — empty results
TEST_F(ServerApiTest, CreateSearchResponseEmpty) {
    std::vector<server_api::SearchResult> results;
    
    auto response = server_api::create_search_response(results);
    auto body_json = json::parse(response.body);
    
    EXPECT_EQ(body_json["status"], "success");
    EXPECT_EQ(body_json["count"], 0);
    EXPECT_EQ(body_json["results"].size(), 0);
}

// Test 7: Create search response — single result
TEST_F(ServerApiTest, CreateSearchResponseSingleResult) {
    std::vector<server_api::SearchResult> results;
    server_api::SearchResult sr;
    sr.id = 0;
    sr.start_time = 10.5f;
    sr.end_time = 15.5f;
    sr.caption = "test caption";
    sr.similarity = 0.95f;
    results.push_back(sr);
    
    auto response = server_api::create_search_response(results);
    auto body_json = json::parse(response.body);
    
    EXPECT_EQ(body_json["status"], "success");
    EXPECT_EQ(body_json["count"], 1);
    EXPECT_EQ(body_json["results"][0]["id"], 0);
    EXPECT_FLOAT_EQ(body_json["results"][0]["start_time"], 10.5f);
    EXPECT_FLOAT_EQ(body_json["results"][0]["end_time"], 15.5f);
    EXPECT_EQ(body_json["results"][0]["caption"], "test caption");
    EXPECT_FLOAT_EQ(body_json["results"][0]["similarity"], 0.95f);
}

// Test 8: Create search response — multiple results
TEST_F(ServerApiTest, CreateSearchResponseMultipleResults) {
    std::vector<server_api::SearchResult> results;
    for (int i = 0; i < 3; ++i) {
        server_api::SearchResult sr;
        sr.id = i;
        sr.start_time = static_cast<float>(i * 10);
        sr.end_time = static_cast<float>(i * 10 + 5);
        sr.caption = "caption " + std::to_string(i);
        sr.similarity = 0.9f - (i * 0.05f);
        results.push_back(sr);
    }
    
    auto response = server_api::create_search_response(results);
    auto body_json = json::parse(response.body);
    
    EXPECT_EQ(body_json["count"], 3);
    EXPECT_EQ(body_json["results"].size(), 3);
    
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(body_json["results"][i]["id"], i);
    }
}

// Test 9: Health check response
TEST_F(ServerApiTest, HealthCheckResponse) {
    // Add some entries to index
    auto emb = normalize(randomEmbedding(128));
    hnsw_index::add(emb, 0.0f, 5.0f, "test");
    
    auto response = server_api::health_check();
    auto body_json = json::parse(response.body);
    
    EXPECT_EQ(body_json["status"], "ok");
    EXPECT_EQ(body_json["service"], "SearvidsServer");
    EXPECT_GE(body_json["index_size"], 1);
}

// Test 10: Health check response header
TEST_F(ServerApiTest, HealthCheckResponseHeader) {
    auto response = server_api::health_check();
    
    EXPECT_GT(response.headers.count("Content-Type"), 0);
}

// Test 11: Search result structure validation
TEST_F(ServerApiTest, SearchResultStructure) {
    server_api::SearchResult sr;
    sr.id = 42;
    sr.start_time = 100.0f;
    sr.end_time = 200.0f;
    sr.caption = "test video segment";
    sr.similarity = 0.87f;
    
    EXPECT_EQ(sr.id, 42);
    EXPECT_FLOAT_EQ(sr.start_time, 100.0f);
    EXPECT_FLOAT_EQ(sr.end_time, 200.0f);
    EXPECT_EQ(sr.caption, "test video segment");
    EXPECT_FLOAT_EQ(sr.similarity, 0.87f);
}

// Test 12: Parse search request — large embedding
TEST_F(ServerApiTest, ParseSearchRequestLargeEmbedding) {
    json j;
    std::vector<float> large_emb(1024);
    for (int i = 0; i < 1024; ++i) {
        large_emb[i] = static_cast<float>(i) / 1024.0f;
    }
    j["embedding"] = large_emb;
    j["topk"] = 20;
    
    auto req = server_api::parse_search_request(j.dump());
    
    EXPECT_EQ(req.embedding.size(), 1024);
    EXPECT_EQ(req.topk, 20);
}

// Test 13: Create search response — JSON structure integrity
TEST_F(ServerApiTest, SearchResponseJsonIntegrity) {
    std::vector<server_api::SearchResult> results;
    server_api::SearchResult sr;
    sr.id = 1;
    sr.start_time = 5.0f;
    sr.end_time = 10.0f;
    sr.caption = "test";
    sr.similarity = 0.9f;
    results.push_back(sr);
    
    auto response = server_api::create_search_response(results);
    auto body_json = json::parse(response.body);
    
    // Verify all required fields exist
    EXPECT_TRUE(body_json.contains("status"));
    EXPECT_TRUE(body_json.contains("count"));
    EXPECT_TRUE(body_json.contains("results"));
    EXPECT_TRUE(body_json["results"][0].contains("id"));
    EXPECT_TRUE(body_json["results"][0].contains("start_time"));
    EXPECT_TRUE(body_json["results"][0].contains("end_time"));
    EXPECT_TRUE(body_json["results"][0].contains("caption"));
    EXPECT_TRUE(body_json["results"][0].contains("similarity"));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}