#include "downloader.h"
#include <curl/curl.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <thread>
#include <chrono>

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
    
    auto callback = (ProgressCallback)clientp;
    if (callback) {
        callback(progress);
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

std::string extract_filename(const std::string& url) {
    size_t last_slash = url.find_last_of("/");
    if (last_slash == std::string::npos) return "";
    
    std::string filename = url.substr(last_slash + 1);
    
    // Remove query parameters if present
    size_t question_mark = filename.find('?');
    if (question_mark != std::string::npos) {
        filename = filename.substr(0, question_mark);
    }
    
    return filename;
}

bool download(const std::string& url, const std::string& output_path, ProgressCallback callback) {
    // Validate URL
    if (!is_valid_url(url)) {
        std::cerr << "Invalid URL: " << url << std::endl;
        return false;
    }
    
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
        curl_easy_setopt(curl, CURLOPT_XFERINFODATA, (void*)callback);
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
        std::remove(output_path.c_str());
        return false;
    }
    
    std::cout << "Download completed: " << output_path << std::endl;
    return true;
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

} // namespace downloader