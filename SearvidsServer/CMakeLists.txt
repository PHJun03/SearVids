# ============================================================
# CMakeLists for SearvidsServer
# ============================================================
cmake_minimum_required(VERSION 3.15)
project(SearvidsServer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Build Optimization (LTO / IPO)
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()


# ========== Main Executable ==========
set(SOURCES
    src/server_main.cpp
)

add_executable(SearvidsServer ${SOURCES})

target_include_directories(SearvidsServer PRIVATE ${PROJECT_SOURCE_DIR}/src)


# ========== Add Utils ==========
add_library(utils STATIC
    src/utils.cpp
    src/utils.h
)

# ========== Add FFmpeg ==========
find_package(FFMPEG REQUIRED COMPONENTS AVCODEC AVFORMAT AVUTIL AVDEVICE SWSCALE SWRESAMPLE)

add_library(ffmpeg_decoder STATIC
    src/ffmpeg_decoder.cpp
    src/ffmpeg_decoder.h
)

target_include_directories(ffmpeg_decoder PRIVATE ${FFMPEG_INCLUDE_DIRS})

target_link_libraries(ffmpeg_decoder PRIVATE ${FFMPEG_LIBRARIES})

# ========== Add Whisper ==========
# whisper.cpp
add_subdirectory(third_party/whisper.cpp EXCLUDE_FROM_ALL)

add_library(whisper_wrapper STATIC
    src/whisper_wrapper.cpp
    src/whisper_wrapper.h
)

target_include_directories(whisper_wrapper PRIVATE ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/include)

# ========== Add ONNX ==========
add_library(clip_onnx STATIC
    src/clip_onnx.cpp
    src/clip_onnx.h
)

set(ONNXRUNTIME_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/onnxruntime/include")
include_directories(${ONNXRUNTIME_INCLUDE_DIR})

set(ONNXRUNTIME_LIB_DIR "${CMAKE_SOURCE_DIR}/third_party/onnxruntime/lib")
target_link_libraries(clip_onnx PRIVATE ${ONNXRUNTIME_LIB_DIR}/onnxruntime.lib)


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
# Crow(HTTP Framework)
find_package(Crow REQUIRED)

# nlohmann_json
find_package(nlohmann_json 3.2.0 REQUIRED)

add_library(server_api STATIC
    src/server_api.cpp
    src/server_api.h
)

target_include_directories(server_api PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(server_api PRIVATE
    Crow::Crow
    hnsw_index
    downloader
    ffmpeg_decoder
    clip_onnx
    whisper_wrapper
    nlohmann_json::nlohmann_json
)


# ========== Main Linking ==========
target_link_libraries(SearvidsServer PRIVATE
    utils
    ffmpeg_decoder
    whisper_wrapper
    clip_onnx
    hnsw_index
    downloader
    server_api
    Crow::Crow
    nlohmann_json::nlohmann_json
)


# ========== Tests ==========
enable_testing()
add_subdirectory(tests)


#[[
# ============================================================
# Options
# ============================================================
option(BUILD_WHISPERCPP "Automatically build whisper.cpp submodule" ON)
option(DOWNLOAD_ONNX "Automatically download ONNX Runtime prebuilt" ON)
option(BUILD_FFMPEG "Automatically download/build FFmpeg" ON)

# ============================================================
# Build Optimization (LTO / IPO)
# ============================================================
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
if(ipo_supported)
    message(STATUS "✅ IPO (Link Time Optimization) enabled for Release build")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
else()
    message(WARNING "⚠️ IPO (LTO) not supported on this platform: ${ipo_error}")
endif()

# ============================================================
# Include & Dependencies
# ============================================================
include(ExternalProject)
include(FetchContent)

# --- Fetch Crow (HTTP Framework) ---
FetchContent_Declare(
    crow
    GIT_REPOSITORY https://github.com/CrowCpp/Crow.git
    GIT_TAG v1.2.0
)
FetchContent_MakeAvailable(crow)
include_directories(${crow_SOURCE_DIR}/include)

include_directories(${CMAKE_SOURCE_DIR}/hnswlib)

# ============================================================
# FFmpeg
# ============================================================
if(WIN32)
    if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
        set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE FILEPATH "")
    endif()

    find_package(FFMPEG REQUIRED COMPONENTS AVCODEC AVFORMAT AVUTIL SWSCALE SWRESAMPLE)
    
    target_include_directories(SearvidsServer PRIVATE ${FFMPEG_INCLUDE_DIRS})
    target_link_libraries(SearvidsServer PRIVATE ${FFMPEG_LIBRARIES})
    
    message(STATUS "🔍 FFmpeg (vcpkg) configured")
    
else()
    set(FFMPEG_DIR "${CMAKE_BINARY_DIR}/ffmpeg")
    if(BUILD_FFMPEG)
        message(STATUS "⬇️ Configuring FFmpeg build from source...")
        ExternalProject_Add(
            ffmpeg_ep
            PREFIX ${FFMPEG_DIR}
            GIT_REPOSITORY https://git.ffmpeg.org/ffmpeg.git
            GIT_TAG n5.1
            CONFIGURE_COMMAND ./configure --disable-static --enable-shared --prefix=${FFMPEG_DIR}/install
            BUILD_COMMAND $(MAKE)
            INSTALL_COMMAND make install
            UPDATE_DISCONNECTED TRUE
        )
        include_directories(${FFMPEG_DIR}/install/include)
        link_directories(${FFMPEG_DIR}/install/lib)
    endif()
endif()

# ============================================================
# whisper.cpp
# ============================================================
set(WHISPER_DIR "${CMAKE_SOURCE_DIR}/third_party/whisper.cpp")
if(BUILD_WHISPERCPP)
    if(EXISTS "${WHISPER_DIR}/Makefile" OR EXISTS "${WHISPER_DIR}/CMakeLists.txt")
        if(WIN32)
            ExternalProject_Add(
                whispercpp_ep
                SOURCE_DIR ${WHISPER_DIR}
                BINARY_DIR ${CMAKE_BINARY_DIR}/whisper_build
                CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release
                BUILD_BYPRODUCTS ${CMAKE_BINARY_DIR}/whisper_build/Release/whisper.lib
                INSTALL_COMMAND ""
                UPDATE_DISCONNECTED TRUE
            )
            add_library(whisper STATIC IMPORTED GLOBAL)
            set_target_properties(whisper PROPERTIES
                IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/whisper_build/Release/whisper.lib"
                INTERFACE_INCLUDE_DIRECTORIES "${WHISPER_DIR}"
            )
        else()
            ExternalProject_Add(
                whispercpp_ep
                SOURCE_DIR ${WHISPER_DIR}
                BINARY_DIR ${WHISPER_DIR}
                CONFIGURE_COMMAND ""
                BUILD_COMMAND ${CMAKE_MAKE_PROGRAM} libwhisper.a -C ${WHISPER_DIR} GGML_NATIVE=1
                BUILD_BYPRODUCTS ${WHISPER_DIR}/libwhisper.a
                INSTALL_COMMAND ""
                UPDATE_COMMAND ""
                UPDATE_DISCONNECTED TRUE
            )
            add_library(whisper STATIC IMPORTED GLOBAL)
            set_target_properties(whisper PROPERTIES
                IMPORTED_LOCATION "${WHISPER_DIR}/libwhisper.a"
                INTERFACE_INCLUDE_DIRECTORIES "${WHISPER_DIR}"
            )
        endif()
        add_dependencies(whisper whispercpp_ep)
        include_directories(${WHISPER_DIR})
    else()
        message(FATAL_ERROR "❌ whisper.cpp submodule missing. Please run 'git submodule update --init --recursive'")
    endif()
endif()

# ============================================================
# ONNX Runtime
# ============================================================
set(ONNX_VERSION "1.16.0")
set(ONNXRUNTIME_DIR "${CMAKE_BINARY_DIR}/onnxruntime")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_DIR}/lib")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include")

if(DOWNLOAD_ONNX AND NOT EXISTS "${ONNXRUNTIME_DIR}")
    message(STATUS "⬇️ Downloading ONNX Runtime prebuilt...")
    if(WIN32)
        set(ONNX_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-win-x64-${ONNX_VERSION}.zip")
        set(ONNX_ZIP "${CMAKE_BINARY_DIR}/onnxruntime.zip")
        file(DOWNLOAD ${ONNX_URL} ${ONNX_ZIP} SHOW_PROGRESS)
        execute_process(COMMAND powershell -Command "Expand-Archive -Path ${ONNX_ZIP} -DestinationPath ${CMAKE_BINARY_DIR}")
    else()
        set(ONNX_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-${ONNX_VERSION}.tgz")
        set(ONNX_TGZ "${CMAKE_BINARY_DIR}/onnxruntime.tgz")
        file(DOWNLOAD ${ONNX_URL} ${ONNX_TGZ} SHOW_PROGRESS)
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xfz ${ONNX_TGZ}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        )
    endif()
    # Normalize folder name (handle nested extracted directory)
    file(GLOB ONNX_EXTRACTED_DIR "${CMAKE_BINARY_DIR}/onnxruntime-*")
    file(RENAME "${ONNX_EXTRACTED_DIR}" "${ONNXRUNTIME_DIR}")
endif()

include_directories(${ONNXRUNTIME_INCLUDE_DIR})
link_directories(${ONNXRUNTIME_LIB_DIR})

# ============================================================
# Sources
# ============================================================
set(SOURCES
    src/server_main.cpp
    src/server_api.cpp
    src/downloader.cpp
    src/ffmpeg_decoder.cpp
    src/clip_onnx.cpp
    src/whisper_wrapper.cpp
    src/hnsw_index.cpp
    src/utils.cpp
)

add_executable(SearvidsServer ${SOURCES})

add_subdirectory(tests)

# ============================================================
# Linking
# ============================================================
if(WIN32)
    target_link_libraries(SearvidsServer whisper avcodec avformat avutil swscale swresample ${ONNXRUNTIME_LIB_DIR}/onnxruntime.lib)
else()
    target_link_libraries(SearvidsServer whisper ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so)
endif()

set_target_properties(SearvidsServer PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
message(STATUS "✅ Searvids Server configuration complete")
]]