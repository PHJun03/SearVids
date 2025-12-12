/*
 * Copyright (c) 2025 Hyeonjun Park (phjun7150@gmail.com)
 * All rights reserved.
 */

#include "downloader.h"
#include <curl/curl.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <filesystem>
#include <cstdlib>

namespace downloader {

// CURL write callback
static size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// CURL progress callback
static int progress_callback(void* clientp, curl_off_t dltotal, curl_off_t dlnow, 
                             curl_off_t ultotal, curl_off_t ulnow) {
    if (dltotal <= 0) return 0;
    
    DownloadProgress progress;
    progress.downloaded_bytes = dlnow;
    progress.total_bytes = dltotal;
    progress.progress_percent = (dlnow / (float)dltotal) * 100.0f;
    
    auto* callback = (ProgressCallback*)clientp;
    if (callback && *callback) {
        (*callback)(progress);
    }
    
    return 0;  // return non-zero to abort
}

bool is_valid_url(const std::string& url) {
    if (url.length() < 7) return false;
    
    // Convert to lowercase for case-insensitive comparison
    std::string url_lower = url;
    std::transform(url_lower.begin(), url_lower.end(), url_lower.begin(), ::tolower);
    
    return url_lower.substr(0, 7) == "http://" || url_lower.substr(0, 8) == "https://";
}

bool is_youtube_url(const std::string& url) {
    std::string url_lower = url;
    std::transform(url_lower.begin(), url_lower.end(), url_lower.begin(), ::tolower);
    
    return url_lower.find("youtube.com") != std::string::npos || 
           url_lower.find("youtu.be") != std::string::npos;
}

std::string extract_filename(const std::string& url) {
    // For YouTube URLs, extract video ID
    if (is_youtube_url(url)) {
        // Handle youtube.com/watch?v=VIDEO_ID format
        size_t v_pos = url.find("v=");
        if (v_pos != std::string::npos) {
            std::string video_id = url.substr(v_pos + 2);
            size_t amp_pos = video_id.find('&');
            if (amp_pos != std::string::npos) {
                video_id = video_id.substr(0, amp_pos);
            }
            return video_id + ".mp4";
        }
        
        // Handle youtu.be/VIDEO_ID format
        size_t slash_pos = url.rfind('/');
        if (slash_pos != std::string::npos) {
            std::string video_id = url.substr(slash_pos + 1);
            size_t question_pos = video_id.find('?');
            if (question_pos != std::string::npos) {
                video_id = video_id.substr(0, question_pos);
            }
            return video_id + ".mp4";
        }
    }
    
    // For regular URLs, extract filename from path
    size_t last_slash = url.find_last_of("/");
    if (last_slash == std::string::npos) return "";
    
    std::string filename = url.substr(last_slash + 1);
    
    // Remove query parameters if present
    size_t question_mark = filename.find('?');
    if (question_mark != std::string::npos) {
        filename = filename.substr(0, question_mark);
    }
    
    // If filename is generic or empty, return default
    if (filename.empty() || filename == "watch") {
        return "video.mp4";
    }
    
    return filename;
}

bool is_ytdlp_available() {
#ifdef _WIN32
    int result = std::system("where yt-dlp >nul 2>&1");
#else
    int result = std::system("which yt-dlp >/dev/null 2>&1");
#endif
    return result == 0;
}

static bool download_with_ytdlp(const std::string& url, const std::string& output_path, 
                                ProgressCallback callback) {
    std::cout << "Downloading YouTube video with yt-dlp..." << std::endl;
    
    // Create directory if it doesn't exist
    std::filesystem::path path(output_path);
    std::filesystem::create_directories(path.parent_path());
    
    // Build yt-dlp command
    // -f "best[ext=mp4]/best" - prefer mp4 format
    // --no-playlist - don't download playlists
    // --no-warnings - suppress warnings
    // -o output_path - output file path
    std::ostringstream cmd;
    cmd << "yt-dlp -f \"best[ext=mp4]/best\" --no-playlist --no-warnings -o \"" 
        << output_path << "\" \"" << url << "\"";
    
#ifdef _WIN32
    cmd << " >nul 2>&1";  // Suppress output on Windows
#else
    cmd << " >/dev/null 2>&1";  // Suppress output on Unix
#endif
    
    std::string command = cmd.str();
    std::cout << "Executing: yt-dlp (output suppressed)" << std::endl;
    
    // Execute command
    int result = std::system(command.c_str());
    
    if (result != 0) {
        std::cerr << "yt-dlp download failed with code " << result << std::endl;
        std::cerr << "Make sure yt-dlp is installed: winget install yt-dlp" << std::endl;
        return false;
    }
    
    // Check if file exists
    if (!std::filesystem::exists(output_path)) {
        std::cerr << "Download completed but output file not found: " << output_path << std::endl;
        return false;
    }
    
    // Get file size for progress callback
    if (callback) {
        auto file_size = std::filesystem::file_size(output_path);
        DownloadProgress progress;
        progress.downloaded_bytes = file_size;
        progress.total_bytes = file_size;
        progress.progress_percent = 100.0f;
        callback(progress);
    }
    
    std::cout << "Download completed: " << output_path << std::endl;
    return true;
}

static bool download_with_curl(const std::string& url, const std::string& output_path, 
                               ProgressCallback callback) {
    std::cout << "Downloading with curl: " << url << std::endl;
    
    // Create directory if it doesn't exist
    std::filesystem::path path(output_path);
    std::filesystem::create_directories(path.parent_path());
    
    // Initialize CURL
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Failed to initialize CURL" << std::endl;
        return false;
    }
    
    std::ofstream outfile(output_path, std::ios::binary);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open output file: " << output_path << std::endl;
        curl_easy_cleanup(curl);
        return false;
    }
    
    // Set CURL options
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300L);  // 5 minutes timeout
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);  // For development only
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);  // For development only
    
    // Write callback
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    std::string buffer;
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void*)&buffer);
    
    // Progress callback (if provided)
    if (callback) {
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress_callback);
        curl_easy_setopt(curl, CURLOPT_XFERINFODATA, (void*)&callback);
    }
    
    // Perform download
    CURLcode res = curl_easy_perform(curl);
    
    // Write to file
    outfile.write(buffer.c_str(), buffer.size());
    outfile.close();
    
    // Cleanup
    curl_easy_cleanup(curl);
    
    if (res != CURLE_OK) {
        std::cerr << "Download failed: " << curl_easy_strerror(res) << std::endl;
        // Remove incomplete file
        std::filesystem::remove(output_path);
        return false;
    }
    
    std::cout << "Download completed: " << output_path << std::endl;
    return true;
}

bool download(const std::string& url, const std::string& output_path, ProgressCallback callback) {
    // Validate URL
    if (!is_valid_url(url)) {
        std::cerr << "Invalid URL: " << url << std::endl;
        return false;
    }
    
    // Use yt-dlp for YouTube URLs
    if (is_youtube_url(url)) {
        if (!is_ytdlp_available()) {
            std::cerr << "yt-dlp is not available. Please install it: winget install yt-dlp" << std::endl;
            return false;
        }
        return download_with_ytdlp(url, output_path, callback);
    }
    
    // Use curl for regular URLs
    return download_with_curl(url, output_path, callback);
}

bool download_with_retry(const std::string& url, const std::string& output_path, 
                         int max_retries, ProgressCallback callback) {
    for (int attempt = 1; attempt <= max_retries; ++attempt) {
        std::cout << "Download attempt " << attempt << "/" << max_retries << std::endl;
        
        if (download(url, output_path, callback)) {
            return true;
        }
        
        if (attempt < max_retries) {
            std::cout << "Retrying in 2 seconds..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }
    
    std::cerr << "Download failed after " << max_retries << " attempts" << std::endl;
    return false;
}

int64_t get_duration(const std::string& url) {
    if (!is_ytdlp_available()) return -1;

    std::string cmd = "yt-dlp --print duration --no-warnings \"" + url + "\"";
    FILE* pipe = 
#ifdef _WIN32
        _popen(cmd.c_str(), "r");
#else
        popen(cmd.c_str(), "r");
#endif
    
    if (!pipe) return -1;
    
    char buffer[128];
    std::string result = "";
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    
#ifdef _WIN32
    _pclose(pipe);
#else
    pclose(pipe);
#endif

    try {
        if (result.empty()) return -1;
        return std::stoll(result);
    } catch (...) {
        return -1;
    }
}

bool download_section(const std::string& url, const std::string& output_path, int start_sec, int end_sec) {
    if (!is_ytdlp_available()) return false;

    std::cout << "Downloading section " << start_sec << "-" << end_sec << "s..." << std::endl;
    
    std::filesystem::path path(output_path);
    std::filesystem::create_directories(path.parent_path());
    
    // yt-dlp --download-sections "*start-end"
    std::ostringstream cmd;
    cmd << "yt-dlp -f \"best[ext=mp4]/best\" --no-playlist --no-warnings "
        << "--download-sections \"*" << start_sec << "-" << end_sec << "\" "
        << "-o \"" << output_path << "\" \"" << url << "\"";
    
#ifdef _WIN32
    cmd << " >nul 2>&1";
#else
    cmd << " >/dev/null 2>&1";
#endif
    
    int result = std::system(cmd.str().c_str());
    
    if (result != 0 || !std::filesystem::exists(output_path)) {
        return false;
    }
    return true;
}

} // namespace downloader