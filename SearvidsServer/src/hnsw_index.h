/*
 * Copyright (c) 2025 Hyeonjun Park (phjun7150@gmail.com)
 * All rights reserved.
 */

#pragma once
#include <vector>
#include <string>

namespace hnsw_index {

struct TimelineEntry {
    int id;
    std::string video_id;
    float start_time;
    float end_time;
    std::string caption;
    float similarity;
};

void create(int dim, const std::string& space = "cosine");
int add(const std::vector<float>& embedding, const std::string& video_id, float start, float end, const std::string& caption);
void remove_video(const std::string& video_id);
std::vector<TimelineEntry> search(const std::vector<float>& query, size_t topk = 5, const std::string& video_id_filter = "", const std::string& type_filter = "");
size_t size();

} // namespace hnsw_index