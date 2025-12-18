# Copyright (c) 2025 Hyeonjun Park (phjun7150@gmail.com)
# All rights reserved.

# ============================================================
# CMakeLists for SearvidsServer
# ============================================================
cmake_minimum_required(VERSION 3.15)
project(SearvidsServer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)


# ========== Add SigLIP ==========
set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnxruntime")

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_c_api.h
    HINTS "${ONNXRUNTIME_ROOT}/include"
    NO_DEFAULT_PATH
    REQUIRED
)

find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime libonnxruntime.so
    HINTS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib64"
    NO_DEFAULT_PATH
    REQUIRED
)

find_package(Threads REQUIRED)

find_package(OpenCV REQUIRED)

add_library(siglip_onnx STATIC
    src/siglip_onnx.cpp
    src/siglip_onnx.h
)

if(USE_TENSORRT)
    target_compile_definitions(siglip_onnx PRIVATE USE_TENSORRT)
endif()

target_include_directories(siglip_onnx PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})

target_link_libraries(siglip_onnx PUBLIC
    ${ONNXRUNTIME_LIB}
    Threads::Threads
    ${CMAKE_DL_LIBS}
    ${OpenCV_LIBS}
)

# ========== Main Executable ==========
set(SOURCES
    src/server_main.cpp
)

add_executable(SearvidsServer ${SOURCES})

target_include_directories(SearvidsServer PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(SearvidsServer PRIVATE utils ffmpeg_decoder whisper_wrapper siglip_onnx)


# ========== Add Utils ==========
add_library(utils STATIC
    src/utils.cpp
    src/utils.h
)

# ========== Add FFmpeg ==========
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET libavformat libavcodec libavutil libswscale libswresample libavfilter)

add_library(ffmpeg_decoder STATIC
    src/ffmpeg_decoder.cpp
    src/ffmpeg_decoder.h
)

set(FFMPEG_INCLUDE_DIRS ${FFMPEG_INCLUDE_DIRS} ${PkgConfig_FFMPEG_INCLUDE_DIRS})
target_include_directories(ffmpeg_decoder PRIVATE ${FFMPEG_INCLUDE_DIRS})

set(FFMPEG_LIBRARIES ${FFMPEG_LIBRARIES} PkgConfig::FFMPEG)
target_link_libraries(ffmpeg_decoder PRIVATE ${FFMPEG_LIBRARIES})

# ========== Add Whisper ==========
add_library(whisper_wrapper STATIC
    src/whisper_wrapper.cpp
    src/whisper_wrapper.h
)


# ========== Add Hnsw ==========
# create an INTERFACE target for header-only hnswlib
add_library(hnswlib INTERFACE)
target_include_directories(hnswlib INTERFACE 
    ${CMAKE_SOURCE_DIR}/third_party/hnswlib
    ${CMAKE_SOURCE_DIR}/third_party/hnswlib/hnswlib
)

add_library(hnsw_index STATIC
    src/hnsw_index.cpp
    src/hnsw_index.h
)

target_link_libraries(hnsw_index PRIVATE hnswlib)


# ========== Add Downloader ==========
find_package(CURL REQUIRED)

add_library(downloader STATIC
    src/downloader.cpp
    src/downloader.h
)
target_link_libraries(downloader PRIVATE CURL::libcurl)


# ========== Add Server API ==========
# Crow
find_package(Crow REQUIRED)

# nlohmann_json
find_package(nlohmann_json 3.2.0 REQUIRED)

add_library(server_api STATIC
    src/server_api.cpp
    src/server_api.h
)

# OpenCV
find_package(OpenCV REQUIRED)

target_include_directories(server_api PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(server_api PRIVATE
    Crow::Crow
    hnsw_index
    downloader
    ffmpeg_decoder
    siglip_onnx
    whisper_wrapper
    nlohmann_json::nlohmann_json
    ${OpenCV_LIBS}
)


# ========== Main Linking ==========
target_link_libraries(SearvidsServer PRIVATE
    utils
    ffmpeg_decoder
    whisper_wrapper
    hnsw_index
    downloader
    server_api
    siglip_onnx   # place after dependents(server_api)
    Crow::Crow
    nlohmann_json::nlohmann_json
)

#[[
# ========== Tests ==========
enable_testing()
add_subdirectory(tests)
]]