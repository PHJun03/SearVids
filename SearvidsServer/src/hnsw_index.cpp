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
#include <algorithm>
#include <iostream>

namespace hnsw_index {

class SearchFilter : public hnswlib::BaseFilterFunctor {
public:
    SearchFilter(const std::vector<TimelineEntry>& entries, const std::string& video_id, const std::string& type)
        : entries_(entries), video_id_(video_id), type_(type) {}

    bool operator()(hnswlib::labeltype label) override {
        if (label >= entries_.size()) return false;
        const auto& entry = entries_[label];
        
        if (!video_id_.empty() && entry.video_id != video_id_) return false;
        
        if (!type_.empty()) {
            bool is_visual = (entry.caption.empty() || entry.caption == "[Visual]" || entry.caption == "visual_frame");
            if (type_ == "visual" && !is_visual) return false;
            if (type_ == "audio" && is_visual) return false;
        }
        
        return true;
    }

private:
    const std::vector<TimelineEntry>& entries_;
    std::string video_id_;
    std::string type_;
};

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

    int add(const std::vector<float>& embedding, const std::string& video_id, float start, float end, const std::string& caption) {
        std::lock_guard<std::mutex> lock(mutex_);
        int id = next_id_++;
        index_->addPoint(embedding.data(), id);
        // use TimelineEntry from header
        entries_.push_back(TimelineEntry{ id, video_id, start, end, caption });
        return id;
    }

    std::vector<TimelineEntry> search(const std::vector<float>& query, size_t topk = 5, const std::string& video_id_filter = "", const std::string& type_filter = "") {
        std::lock_guard<std::mutex> lock(mutex_);
        if (entries_.empty()) return {};

        size_t k = std::min(topk, entries_.size());
        
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result;
        if (!video_id_filter.empty() || !type_filter.empty()) {
            SearchFilter filter(entries_, video_id_filter, type_filter);
            try {
                result = index_->searchKnn(query.data(), k, &filter);
            } catch (...) {
                // Ignore errors
            }
        } else {
            result = index_->searchKnn(query.data(), k);
        }
        
        std::vector<TimelineEntry> out;
        while (!result.empty()) {
            int id = result.top().second;
            float dist = result.top().first;
            result.pop();
            // find the entry by id
            if (id < entries_.size()) {
                TimelineEntry res = entries_[id];
                res.similarity = 1.0f - dist;
                out.push_back(res);
            }
        }
        // Result is from worst to best (priority queue max heap), so reverse it
        std::reverse(out.begin(), out.end());
        return out;
    }

    size_t size() const { return entries_.size(); }

    void remove_video(const std::string& video_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& entry : entries_) {
            if (entry.video_id == video_id) {
                // Clear data to "delete" it
                entry.video_id = "";
                entry.caption = "";
                entry.start_time = 0;
                entry.end_time = 0;
            }
        }
    }

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

int add(const std::vector<float>& embedding, const std::string& video_id, float start, float end, const std::string& caption) {
    if (!g_index) throw std::runtime_error("Index not created");
    return g_index->add(embedding, video_id, start, end, caption);
}

void remove_video(const std::string& video_id) {
    if (g_index) {
        g_index->remove_video(video_id);
    }
}

std::vector<TimelineEntry> search(const std::vector<float>& query, size_t topk, const std::string& video_id_filter, const std::string& type_filter) {
    if (!g_index) throw std::runtime_error("Index not created");
    return g_index->search(query, topk, video_id_filter, type_filter);
}

size_t size() {
    if (!g_index) return 0;
    return g_index->size();
}

} // namespace hnsw_index