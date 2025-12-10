/*
 * Copyright (c) 2025 Hyeonjun Park (phjun7150@gmail.com)
 * All rights reserved.
 */

#pragma once
#include <string>
#include <functional>

namespace downloader {

struct DownloadProgress {
    size_t downloaded_bytes;
    size_t total_bytes;
    float progress_percent;
};

using ProgressCallback = std::function<void(const DownloadProgress&)>;

/**
 * Check if URL is valid (starts with http:// or https://)
 */
bool is_valid_url(const std::string& url);

/**
 * Check if URL is a YouTube URL
 */
bool is_youtube_url(const std::string& url);

/**
 * Extract filename from URL
 * For YouTube URLs, extracts video ID
 */
std::string extract_filename(const std::string& url);

/**
 * Download file from URL
 * Uses yt-dlp for YouTube URLs, curl for others
 */
bool download(const std::string& url, const std::string& output_path, ProgressCallback callback = nullptr);

/**
 * Download with retry mechanism
 */
bool download_with_retry(const std::string& url, const std::string& output_path, 
                         int max_retries = 3, ProgressCallback callback = nullptr);

/**
 * Check if yt-dlp is available in PATH
 */
bool is_ytdlp_available();

} // namespace downloader