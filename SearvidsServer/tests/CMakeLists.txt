# ============================================================
# GoogleTest setup for SearVids Server
# ============================================================

cmake_minimum_required(VERSION 3.16)

# Create main test target
add_executable(SearvidsTests
    test_utils.cpp
    test_ffmpeg_decoder.cpp
    test_whisper_wrapper.cpp
    test_clip_onnx.cpp

    # Do not add under tests at the same time to avoid main function redefinition error
    # test_hnsw_index.cpp
    # test_downloader.cpp
    test_server_api.cpp
)

# Link GTest
find_package(GTest CONFIG REQUIRED)
target_link_libraries(SearvidsTests
    PRIVATE
        GTest::gtest
        GTest::gtest_main
)

# Include all project headers
target_include_directories(SearvidsTests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

# Link project modules (if they are static libs)
target_link_libraries(SearvidsTests
    PRIVATE
        utils
        ffmpeg_decoder
        whisper_wrapper
        clip_onnx
        hnsw_index
        downloader
        server_api
)

# Enable test discovery
enable_testing()
add_test(NAME AllTests COMMAND SearvidsTests)