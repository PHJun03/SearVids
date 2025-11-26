#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "hnsw_index.h"

class HnswIndexTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize with 128-dim cosine space before each test
        hnsw_index::create(128, "cosine");
    }

    void TearDown() override {
        // No cleanup needed for static singleton
    }

    // Helper: create a random embedding
    std::vector<float> randomEmbedding(int dim) {
        std::vector<float> emb(dim);
        for (int i = 0; i < dim; ++i) {
            emb[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        return emb;
    }

    // Helper: normalize embedding (for cosine distance)
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

// Test 1: Create index and verify size
TEST_F(HnswIndexTest, CreateAndCheckSize) {
    EXPECT_EQ(hnsw_index::size(), 0);
}

// Test 2: Add single entry and verify
TEST_F(HnswIndexTest, AddSingleEntry) {
    auto emb = randomEmbedding(128);
    int id = hnsw_index::add(emb, 0.0f, 5.0f, "test caption");
    
    EXPECT_EQ(id, 0);
    EXPECT_EQ(hnsw_index::size(), 1);
}

// Test 3: Add multiple entries
TEST_F(HnswIndexTest, AddMultipleEntries) {
    for (int i = 0; i < 10; ++i) {
        auto emb = randomEmbedding(128);
        int id = hnsw_index::add(emb, static_cast<float>(i * 5), static_cast<float>(i * 5 + 5), 
                                 "caption " + std::to_string(i));
        EXPECT_EQ(id, i);
    }
    EXPECT_EQ(hnsw_index::size(), 10);
}

// Test 4: Search for exact match
TEST_F(HnswIndexTest, SearchExactMatch) {
    std::vector<float> emb = normalize(randomEmbedding(128));
    int id = hnsw_index::add(emb, 10.0f, 15.0f, "target caption");
    
    // Search with same embedding
    auto results = hnsw_index::search(emb, 1);
    
    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].id, id);
    EXPECT_EQ(results[0].start_time, 10.0f);
    EXPECT_EQ(results[0].end_time, 15.0f);
    EXPECT_EQ(results[0].caption, "target caption");
}

// Test 5: Search top-k with controlled similarity
TEST_F(HnswIndexTest, SearchTopKWithControlledSimilarity) {
    // Base embedding
    std::vector<float> base = normalize(randomEmbedding(128));
    
    // Add base embedding (ID 0)
    hnsw_index::add(base, 0.0f, 5.0f, "base caption");
    
    // Add 4 different embeddings (IDs 1-4)
    for (int i = 1; i < 5; ++i) {
        auto emb = normalize(randomEmbedding(128));
        hnsw_index::add(emb, static_cast<float>(i * 10), static_cast<float>(i * 10 + 5), 
                        "caption " + std::to_string(i));
    }
    
    // Search with base embedding, top-3
    auto results = hnsw_index::search(base, 3);
    
    EXPECT_LE(results.size(), 3);
    EXPECT_GE(results.size(), 1);
    
    // ID 0 (exact match) should be in results
    bool found_id_0 = false;
    for (const auto& result : results) {
        if (result.id == 0) {
            found_id_0 = true;
            break;
        }
    }
    EXPECT_TRUE(found_id_0) << "Exact match (ID 0) should be in top-3 results";
}

// Test 6: Timeline entry metadata
TEST_F(HnswIndexTest, TimelineEntryMetadata) {
    auto emb = randomEmbedding(128);
    float start = 123.45f;
    float end = 150.67f;
    std::string caption = "cat jumping on sofa";
    
    int id = hnsw_index::add(emb, start, end, caption);
    auto results = hnsw_index::search(emb, 1);
    
    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].start_time, start);
    EXPECT_EQ(results[0].end_time, end);
    EXPECT_EQ(results[0].caption, caption);
}

// Test 7: Search with topk larger than index size
TEST_F(HnswIndexTest, SearchTopkLargerThanSize) {
    for (int i = 0; i < 3; ++i) {
        auto emb = normalize(randomEmbedding(128));
        hnsw_index::add(emb, static_cast<float>(i), static_cast<float>(i + 1), "cap");
    }
    
    auto results = hnsw_index::search(normalize(randomEmbedding(128)), 100);
    
    EXPECT_LE(results.size(), 3);
    EXPECT_GE(results.size(), 1);
}

// Test 8: Empty search (topk=0)
TEST_F(HnswIndexTest, SearchWithZeroTopk) {
    auto emb = randomEmbedding(128);
    hnsw_index::add(emb, 0.0f, 5.0f, "test");
    
    auto results = hnsw_index::search(emb, 0);
    
    EXPECT_EQ(results.size(), 0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}