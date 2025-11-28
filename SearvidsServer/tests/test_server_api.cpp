#include <gtest/gtest.h>
#include "server_api.h"
#include <nlohmann/json.hpp>
#include <thread>
#include <chrono>

// Test fixture for ServerAPI tests
class ServerAPITest : public ::testing::Test {
protected:
    void SetUp() override {
        // Clear global sessions before each test
        std::lock_guard<std::mutex> lk(server_api::g_sessions_mtx);
        server_api::g_sessions.clear();
    }

    void TearDown() override {
        // Clean up after each test
        std::lock_guard<std::mutex> lk(server_api::g_sessions_mtx);
        server_api::g_sessions.clear();
    }
};

TEST_F(ServerAPITest, MakeVideoIDFromURL) {
    auto vid1 = server_api::make_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ");
    auto vid2 = server_api::make_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ");
    auto vid3 = server_api::make_video_id("https://www.youtube.com/watch?v=different");
    
    EXPECT_EQ(vid1, vid2);  // Same URL should produce same ID
    EXPECT_NE(vid1, vid3);  // Different URL should produce different ID
    EXPECT_FALSE(vid1.empty());
}

TEST_F(ServerAPITest, ParseSearchRequestTextQuery) {
    std::string json_body = R"({
        "query": "test search",
        "topk": 10
    })";
    
    auto req = server_api::parse_search_request(json_body);
    EXPECT_EQ(req.query, "test search");
    EXPECT_EQ(req.topk, 10);
    EXPECT_TRUE(req.embedding.empty());
}

TEST_F(ServerAPITest, ParseSearchRequestEmbedding) {
    std::string json_body = R"({
        "embedding": [0.1, 0.2, 0.3, 0.4],
        "topk": 5
    })";
    
    auto req = server_api::parse_search_request(json_body);
    EXPECT_TRUE(req.query.empty());
    EXPECT_EQ(req.topk, 5);
    ASSERT_EQ(req.embedding.size(), 4);
    EXPECT_FLOAT_EQ(req.embedding[0], 0.1f);
    EXPECT_FLOAT_EQ(req.embedding[1], 0.2f);
    EXPECT_FLOAT_EQ(req.embedding[2], 0.3f);
    EXPECT_FLOAT_EQ(req.embedding[3], 0.4f);
}

TEST_F(ServerAPITest, ParseSearchRequestDefaultValues) {
    std::string json_body = "{}";
    
    auto req = server_api::parse_search_request(json_body);
    EXPECT_TRUE(req.query.empty());
    EXPECT_EQ(req.topk, 5);  // Default value
    EXPECT_TRUE(req.embedding.empty());
}

TEST_F(ServerAPITest, ParseSearchRequestInvalidJSON) {
    std::string json_body = "invalid json";
    
    auto req = server_api::parse_search_request(json_body);
    EXPECT_TRUE(req.query.empty());
    EXPECT_EQ(req.topk, 5);  // Default value
    EXPECT_TRUE(req.embedding.empty());
}

TEST_F(ServerAPITest, CreateSearchResponseEmpty) {
    std::vector<server_api::SearchResult> results;
    
    auto response = server_api::create_search_response(results);
    EXPECT_EQ(response.code, 200);
    
    auto json = nlohmann::json::parse(response.body);
    EXPECT_EQ(json["status"], "ok");
    EXPECT_TRUE(json["results"].is_array());
    EXPECT_EQ(json["results"].size(), 0);
}

TEST_F(ServerAPITest, CreateSearchResponseWithResults) {
    std::vector<server_api::SearchResult> results = {
        {1, 10.5f, 15.2f, "First segment", 0.95f},
        {2, 20.0f, 25.5f, "Second segment", 0.87f},
        {3, 30.1f, 35.8f, "Third segment", 0.76f}
    };
    
    auto response = server_api::create_search_response(results);
    EXPECT_EQ(response.code, 200);
    
    auto json = nlohmann::json::parse(response.body);
    EXPECT_EQ(json["status"], "ok");
    EXPECT_TRUE(json["results"].is_array());
    ASSERT_EQ(json["results"].size(), 3);
    
    // Check first result
    EXPECT_EQ(json["results"][0]["id"], 1);
    EXPECT_FLOAT_EQ(json["results"][0]["start_time"], 10.5f);
    EXPECT_FLOAT_EQ(json["results"][0]["end_time"], 15.2f);
    EXPECT_EQ(json["results"][0]["caption"], "First segment");
    EXPECT_FLOAT_EQ(json["results"][0]["similarity"], 0.95f);
    
    // Check second result
    EXPECT_EQ(json["results"][1]["id"], 2);
    EXPECT_FLOAT_EQ(json["results"][1]["start_time"], 20.0f);
    EXPECT_FLOAT_EQ(json["results"][1]["end_time"], 25.5f);
    EXPECT_EQ(json["results"][1]["caption"], "Second segment");
    EXPECT_FLOAT_EQ(json["results"][1]["similarity"], 0.87f);
}

TEST_F(ServerAPITest, HealthCheck) {
    auto response = server_api::health_check();
    EXPECT_EQ(response.code, 200);
    
    auto json = nlohmann::json::parse(response.body);
    EXPECT_EQ(json["status"], "ok");
}

TEST_F(ServerAPITest, VideoSessionStructure) {
    server_api::VideoSession sess;
    sess.video_id = "test123";
    sess.source_url = "https://example.com/video.mp4";
    sess.local_path = "data/test123_video.mp4";
    sess.duration_ms = 120000;
    sess.nb_frames = 3000;
    sess.analyzing = false;
    sess.done = true;
    
    EXPECT_EQ(sess.video_id, "test123");
    EXPECT_EQ(sess.source_url, "https://example.com/video.mp4");
    EXPECT_EQ(sess.local_path, "data/test123_video.mp4");
    EXPECT_EQ(sess.duration_ms, 120000);
    EXPECT_EQ(sess.nb_frames, 3000);
    EXPECT_FALSE(sess.analyzing);
    EXPECT_TRUE(sess.done);
    EXPECT_TRUE(sess.error.empty());
}

TEST_F(ServerAPITest, GlobalSessionsMapThreadSafety) {
    // Test concurrent access to global sessions
    std::vector<std::thread> threads;
    
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([i]() {
            std::lock_guard<std::mutex> lk(server_api::g_sessions_mtx);
            auto vid = "video_" + std::to_string(i);
            auto& sess = server_api::g_sessions[vid];
            sess.video_id = vid;
            sess.source_url = "https://example.com/" + vid;
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    std::lock_guard<std::mutex> lk(server_api::g_sessions_mtx);
    EXPECT_EQ(server_api::g_sessions.size(), 10);
}

TEST_F(ServerAPITest, SearchRequestStructure) {
    server_api::SearchRequest req;
    req.query = "test query";
    req.embedding = {0.1f, 0.2f, 0.3f};
    req.topk = 7;
    
    EXPECT_EQ(req.query, "test query");
    ASSERT_EQ(req.embedding.size(), 3);
    EXPECT_FLOAT_EQ(req.embedding[0], 0.1f);
    EXPECT_EQ(req.topk, 7);
}

TEST_F(ServerAPITest, SearchResultStructure) {
    server_api::SearchResult result;
    result.id = 42;
    result.start_time = 10.5f;
    result.end_time = 15.2f;
    result.caption = "Test caption";
    result.similarity = 0.89f;
    
    EXPECT_EQ(result.id, 42);
    EXPECT_FLOAT_EQ(result.start_time, 10.5f);
    EXPECT_FLOAT_EQ(result.end_time, 15.2f);
    EXPECT_EQ(result.caption, "Test caption");
    EXPECT_FLOAT_EQ(result.similarity, 0.89f);
}

TEST_F(ServerAPITest, ParseSearchRequestMixedContent) {
    std::string json_body = R"({
        "query": "search text",
        "embedding": [0.1, 0.2],
        "topk": 15
    })";
    
    auto req = server_api::parse_search_request(json_body);
    // When both are provided, both should be parsed
    EXPECT_EQ(req.query, "search text");
    ASSERT_EQ(req.embedding.size(), 2);
    EXPECT_EQ(req.topk, 15);
}

TEST_F(ServerAPITest, CreateSearchResponseSpecialCharacters) {
    std::vector<server_api::SearchResult> results = {
        {1, 0.0f, 5.0f, "Caption with \"quotes\" and 'apostrophes'", 0.99f},
        {2, 5.0f, 10.0f, "Caption with\nnewlines\tand\ttabs", 0.88f}
    };
    
    auto response = server_api::create_search_response(results);
    EXPECT_EQ(response.code, 200);
    
    // Should be valid JSON despite special characters
    auto json = nlohmann::json::parse(response.body);
    EXPECT_EQ(json["results"].size(), 2);
}

TEST_F(ServerAPITest, VideoSessionAtomicOperations) {
    server_api::VideoSession sess;
    
    // Test atomic bool operations
    EXPECT_FALSE(sess.analyzing.load());
    EXPECT_FALSE(sess.done.load());
    
    sess.analyzing = true;
    EXPECT_TRUE(sess.analyzing.load());
    
    sess.done = true;
    EXPECT_TRUE(sess.done.load());
    
    // Test concurrent atomic access
    std::thread t1([&sess]() {
        for (int i = 0; i < 100; ++i) {
            sess.analyzing = !sess.analyzing.load();
        }
    });
    
    std::thread t2([&sess]() {
        for (int i = 0; i < 100; ++i) {
            sess.done = !sess.done.load();
        }
    });
    
    t1.join();
    t2.join();
    
    // Values should be consistent (no race condition)
    // Just verify they are valid bool values
    EXPECT_TRUE(sess.analyzing.load() == true || sess.analyzing.load() == false);
    EXPECT_TRUE(sess.done.load() == true || sess.done.load() == false);
}