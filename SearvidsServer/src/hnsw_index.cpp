/*
 * Copyright (c) 2025 Hyeonjun Park (phjun7150@gmail.com)
 * All rights reserved.
 */

#include "hnsw_index.h"
#include <hnswlib.h>
#include <mutex>
#include <stdexcept>
#include <vector>
#include <string>
#include <memory>

namespace hnsw_index {

class HnswIndexImpl {
public:
    HnswIndexImpl(int dim, const std::string& space = "cosine")
        : dim_(dim), space_(space), next_id_(0)
    {
        if (space_ == "cosine") {
            space_l2_ = std::make_unique<hnswlib::InnerProductSpace>(dim_);
        } else {
            space_l2_ = std::make_unique<hnswlib::L2Space>(dim_);
        }
        index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(space_l2_.get(), 10000);
    }

    int add(const std::vector<float>& embedding, float start, float end, const std::string& caption) {
        std::lock_guard<std::mutex> lock(mutex_);
        int id = next_id_++;
        index_->addPoint(embedding.data(), id);
        // use TimelineEntry from header
        entries_.push_back(TimelineEntry{ id, start, end, caption });
        return id;
    }

    std::vector<TimelineEntry> search(const std::vector<float>& query, size_t topk = 5) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (entries_.empty()) return {};

        size_t k = std::min(topk, entries_.size());
        auto result = index_->searchKnn(query.data(), k);
        std::vector<TimelineEntry> out;
        while (!result.empty()) {
            int id = result.top().second;
            float dist = result.top().first;
            result.pop();
            // find the entry by id
            for (const auto& e : entries_) {
                if (e.id == id) {
                    TimelineEntry res = e;
                    // hnswlib InnerProduct distance is 1.0 - dot_product (if normalized)
                    // We want similarity (dot product), so 1.0 - dist
                    res.similarity = 1.0f - dist;
                    out.push_back(res);
                    break;
                }
            }
        }
        // Result is from worst to best (priority queue max heap), so reverse it
        std::reverse(out.begin(), out.end());
        return out;
    }

    size_t size() const { return entries_.size(); }

private:
    int dim_;
    std::string space_;
    int next_id_;
    std::unique_ptr<hnswlib::SpaceInterface<float>> space_l2_;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
    std::vector<TimelineEntry> entries_;
    mutable std::mutex mutex_;
};

static std::unique_ptr<HnswIndexImpl> g_index;

void create(int dim, const std::string& space) {
    g_index = std::make_unique<HnswIndexImpl>(dim, space);
}

int add(const std::vector<float>& embedding, float start, float end, const std::string& caption) {
    if (!g_index) throw std::runtime_error("Index not created");
    return g_index->add(embedding, start, end, caption);
}

std::vector<TimelineEntry> search(const std::vector<float>& query, size_t topk) {
    if (!g_index) throw std::runtime_error("Index not created");
    return g_index->search(query, topk);
}

size_t size() {
    if (!g_index) return 0;
    return g_index->size();
}

} // namespace hnsw_index