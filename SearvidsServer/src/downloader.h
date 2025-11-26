#pragma once
#include <string>
#include <vector>

namespace downloader {

struct DownloadProgress {
    long long downloaded_bytes;
    long long total_bytes;
    float progress_percent;  // 0.0 ~ 100.0
};

// Callback function type for progress updates
using ProgressCallback = void(*)(const DownloadProgress& progress);

/**
 * Download file from URL to local path
 * @param url: source URL (http/https)
 * @param output_path: local file path to save
 * @param callback: optional progress callback (nullptr to skip)
 * @return true if successful, false otherwise
 */
bool download(const std::string& url, const std::string& output_path, ProgressCallback callback = nullptr);

/**
 * Download file with retry logic
 * @param url: source URL
 * @param output_path: local file path to save
 * @param max_retries: number of retries on failure
 * @param callback: optional progress callback
 * @return true if successful after retries
 */
bool download_with_retry(const std::string& url, const std::string& output_path, 
                         int max_retries = 3, ProgressCallback callback = nullptr);

/**
 * Check if URL is valid (http/https)
 * @param url: URL to validate
 * @return true if valid
 */
bool is_valid_url(const std::string& url);

/**
 * Extract filename from URL
 * @param url: source URL
 * @return filename (or empty string if invalid)
 */
std::string extract_filename(const std::string& url);

} // namespace downloader