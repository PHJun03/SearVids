#pragma once
#include <vector>
#include <string>

namespace hnsw_index {

struct TimelineEntry {
    int id;
    float start_time;
    float end_time;
    std::string caption;
    float similarity;
};

void create(int dim, const std::string& space = "cosine");
int add(const std::vector<float>& embedding, float start, float end, const std::string& caption);
std::vector<TimelineEntry> search(const std::vector<float>& query, size_t topk = 5);
size_t size();

} // namespace hnsw_index