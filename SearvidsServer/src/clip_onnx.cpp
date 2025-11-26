// src/clip_onnx.cpp
#include "clip_onnx.h"

#include <array>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <filesystem>

// stb_image for simple image loading; make sure stb_image.h is available in include paths.
// Define implementation in this translation unit.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace std::string_literals;

namespace clip_onnx {

static std::wstring to_wstring_if_windows(const std::string& s) {
#ifdef _WIN32
    return std::wstring(s.begin(), s.end());
#else
    return std::wstring();
#endif
}

/**
 * Constructor: loads two ONNX sessions (text + vision).
 * If device_gpu is true, attempts to enable CUDA execution provider.
 */
ClipOnnx::ClipOnnx(const std::string& text_model_path,
                   const std::string& vision_model_path,
                   bool device_gpu,
                   int image_size)
    : env_(ORT_LOGGING_LEVEL_WARNING, "clip_onnx"),
      use_gpu_(device_gpu),
      image_size_(image_size)
{
    try {
        // session options common settings
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // If GPU requested, try to append CUDA EP.
        if (use_gpu_) {
#ifdef USE_CUDA  // optional compile-time guard if you add it
            OrtCUDAProviderOptions cuda_opts;
            session_options_.AppendExecutionProvider_CUDA(cuda_opts);
            std::cout << "[clip_onnx] Using CUDA Execution Provider\n";
#else
            // Try to append CUDA provider at runtime using Ort C++ API (works if onnxruntime built with CUDA)
            try {
                OrtCUDAProviderOptions cuda_options;
                session_options_.AppendExecutionProvider_CUDA(cuda_options);
                std::cout << "[clip_onnx] CUDA EP appended (if available)\n";
            } catch (...) {
                std::cerr << "[clip_onnx] Warning: CUDA EP append failed; falling back to CPU\n";
                use_gpu_ = false;
            }
#endif
        } else {
            std::cout << "[clip_onnx] Using CPU Execution Provider\n";
        }

        // Create sessions: use wide-string path on Windows (some Ort builds expect wchar_t)
#ifdef _WIN32
        std::wstring wtext = to_wstring_if_windows(text_model_path);
        std::wstring wvision = to_wstring_if_windows(vision_model_path);
        text_session_ = std::make_unique<Ort::Session>(env_, wtext.c_str(), session_options_);
        vision_session_ = std::make_unique<Ort::Session>(env_, wvision.c_str(), session_options_);
#else
        text_session_ = std::make_unique<Ort::Session>(env_, text_model_path.c_str(), session_options_);
        vision_session_ = std::make_unique<Ort::Session>(env_, vision_model_path.c_str(), session_options_);
#endif

        std::cout << "[clip_onnx] Loaded text model: " << text_model_path << "\n";
        std::cout << "[clip_onnx] Loaded vision model: " << vision_model_path << "\n";
    } catch (const Ort::Exception& e) {
        throw std::runtime_error(std::string("ONNX Runtime error during session creation: ") + e.what());
    } catch (const std::exception& e) {
        throw;
    }
}

ClipOnnx::~ClipOnnx() = default;

/* ---------------------------
   Text / Vision runners
   --------------------------- */

/**
 * Run text session. Accepts token ids and returns L2-normalized embedding.
 *
 * Note: This implementation is minimal — many real models require attention_mask,
 * token_type_ids, etc. Here we assume a single input "input_ids" and that model returns
 * a single float output that can be averaged over sequence dimension if needed.
 */
std::vector<float> ClipOnnx::runTextSession(const std::vector<int64_t>& input_ids,
                                            const std::vector<int64_t>& input_shape)
{
    if (!text_session_) throw std::runtime_error("Text session not initialized");

    // Create CPU memory info
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Create tensor for input_ids
    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info,
        const_cast<int64_t*>(input_ids.data()),
        static_cast<size_t>(input_ids.size()),
        input_shape.data(),
        input_shape.size());

    // If model requires additional inputs (attention_mask) you should add them here.
    const char* input_names[] = { text_input_name_.c_str() };
    const char* output_names[] = { text_output_name_.c_str() };

    try {
        auto output_tensors = text_session_->Run(Ort::RunOptions{nullptr},
                                                 input_names, &input_tensor, 1,
                                                 output_names, 1);
        if (output_tensors.empty()) throw std::runtime_error("Text model produced no outputs");

        Ort::Value& out = output_tensors.front();
        float* out_data = out.GetTensorMutableData<float>();
        size_t out_count = out.GetTensorTypeAndShapeInfo().GetElementCount();

        std::vector<float> vec(out_data, out_data + out_count);

        // If output is 3D (batch, seq, dim) we try to collapse by averaging sequence axis.
        auto shape = out.GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() == 3) {
            // shape: [1, seq, dim]
            int64_t seq = static_cast<int64_t>(shape[1]);
            int64_t dim = static_cast<int64_t>(shape[2]);
            std::vector<float> avg(dim, 0.0f);
            for (int64_t i = 0; i < seq; ++i) {
                for (int64_t j = 0; j < dim; ++j) {
                    avg[j] += vec[i * dim + j];
                }
            }
            for (int64_t j = 0; j < dim; ++j) avg[j] /= static_cast<float>(seq);
            return l2Normalize(avg);
        }

        return l2Normalize(vec);
    } catch (const Ort::Exception& e) {
        throw std::runtime_error(std::string("ONNX Runtime error in text run: ") + e.what());
    }
}

/**
 * Run vision session. Accepts preprocessed image tensor (NCHW float array) and returns L2-normalized embedding.
 */
std::vector<float> ClipOnnx::runVisionSession(const std::vector<float>& image_tensor,
                                              const std::vector<int64_t>& input_shape)
{
    if (!vision_session_) throw std::runtime_error("Vision session not initialized");

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        const_cast<float*>(image_tensor.data()),
        static_cast<size_t>(image_tensor.size()),
        input_shape.data(),
        input_shape.size());

    const char* input_names[] = { vision_input_name_.c_str() };
    const char* output_names[] = { vision_output_name_.c_str() };

    try {
        auto output_tensors = vision_session_->Run(Ort::RunOptions{nullptr},
                                                   input_names, &input_tensor, 1,
                                                   output_names, 1);
        if (output_tensors.empty()) throw std::runtime_error("Vision model produced no outputs");

        Ort::Value& out = output_tensors.front();
        float* out_data = out.GetTensorMutableData<float>();
        size_t out_count = out.GetTensorTypeAndShapeInfo().GetElementCount();
        std::vector<float> vec(out_data, out_data + out_count);

        // If output is 4D (N,C,H,W) or (N,seq,dim) try reduce to embeddings
        auto shape = out.GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() == 4) {
            // e.g., [1, C, H, W] -> average spatial dims
            int64_t C = static_cast<int64_t>(shape[1]);
            int64_t H = static_cast<int64_t>(shape[2]);
            int64_t W = static_cast<int64_t>(shape[3]);
            std::vector<float> avg(C, 0.0f);
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t y = 0; y < H; ++y) {
                    for (int64_t x = 0; x < W; ++x) {
                        size_t idx = static_cast<size_t>(c * H * W + y * W + x);
                        avg[c] += vec[idx];
                    }
                }
                avg[c] /= static_cast<float>(H * W);
            }
            return l2Normalize(avg);
        }

        if (shape.size() == 3) {
            // e.g., [1, seq, dim] -> average seq
            int64_t seq = static_cast<int64_t>(shape[1]);
            int64_t dim = static_cast<int64_t>(shape[2]);
            std::vector<float> avg(dim, 0.0f);
            for (int64_t i = 0; i < seq; ++i) {
                for (int64_t j = 0; j < dim; ++j) {
                    avg[j] += vec[i * dim + j];
                }
            }
            for (int64_t j = 0; j < dim; ++j) avg[j] /= static_cast<float>(seq);
            return l2Normalize(avg);
        }

        return l2Normalize(vec);
    } catch (const Ort::Exception& e) {
        throw std::runtime_error(std::string("ONNX Runtime error in vision run: ") + e.what());
    }
}

/* ---------------------------
   Public wrapper functions
   --------------------------- */

std::vector<float> ClipOnnx::encodeText(const std::string& text) {
    // Tokenize (naive); production code should use proper tokenizer to produce ids and attention mask
    auto ids = naiveTokenize(text, 64);
    if (ids.empty()) ids.push_back(0);
    std::vector<int64_t> shape = {1, static_cast<int64_t>(ids.size())};
    return runTextSession(ids, shape);
}

std::vector<float> ClipOnnx::encodeImageFromFile(const std::string& image_path) {
    auto img_tensor = loadAndPreprocessImage(image_path);
    std::vector<int64_t> shape = {1, 3, image_size_, image_size_};
    return runVisionSession(img_tensor, shape);
}

/* ---------------------------
   Utilities
   --------------------------- */

std::vector<float> ClipOnnx::l2Normalize(const std::vector<float>& v) {
    double sumsq = 0.0;
    for (float x : v) sumsq += static_cast<double>(x) * static_cast<double>(x);
    double norm = std::sqrt(sumsq);
    std::vector<float> out(v.size());
    if (norm <= 1e-12) return out;
    for (size_t i = 0; i < v.size(); ++i) out[i] = static_cast<float>(v[i] / norm);
    return out;
}

std::vector<int64_t> ClipOnnx::naiveTokenize(const std::string& text, size_t max_tokens) {
    // Very simple char-based tokenizer: map bytes to small ids.
    // Replace with proper tokenizer for real use.
    std::vector<int64_t> tok;
    tok.reserve(std::min(max_tokens, text.size()));
    for (char ch : text) {
        if (ch == ' ') continue;
        tok.push_back(static_cast<int64_t>(static_cast<unsigned char>(ch) % 256));
        if (tok.size() >= max_tokens) break;
    }
    if (tok.empty()) tok.push_back(0);
    return tok;
}

/**
 * Load an image using stb_image and preprocess:
 *  - force 3 channels
 *  - nearest-neighbor resize to image_size_
 *  - normalize (mean/std) and produce NCHW float tensor
 */
std::vector<float> ClipOnnx::loadAndPreprocessImage(const std::string& path) {
    int w, h, c;
    unsigned char* img = stbi_load(path.c_str(), &w, &h, &c, 3);
    if (!img) {
        throw std::runtime_error("Failed to load image: " + path);
    }

    // allocate resized buffer (CHW)
    std::vector<float> resized(3 * image_size_ * image_size_);

    // simple nearest-neighbor resize and normalization
    for (int y = 0; y < image_size_; ++y) {
        int src_y = (y * h) / image_size_;
        if (src_y >= h) src_y = h - 1;
        for (int x = 0; x < image_size_; ++x) {
            int src_x = (x * w) / image_size_;
            if (src_x >= w) src_x = w - 1;
            for (int ch = 0; ch < 3; ++ch) {
                int src_idx = (src_y * w + src_x) * 3 + ch;
                float v = static_cast<float>(img[src_idx]) / 255.0f;
                v = (v - mean_[ch]) / std_[ch];
                // NCHW ordering: channel-major
                size_t dst_idx = static_cast<size_t>(ch) * image_size_ * image_size_ + y * image_size_ + x;
                resized[dst_idx] = v;
            }
        }
    }

    stbi_image_free(img);
    return resized;
}

/* ---------------------------
   Node name setters
   --------------------------- */
void ClipOnnx::setTextInputName(const std::string& name) { text_input_name_ = name; }
void ClipOnnx::setTextOutputName(const std::string& name) { text_output_name_ = name; }
void ClipOnnx::setVisionInputName(const std::string& name) { vision_input_name_ = name; }
void ClipOnnx::setVisionOutputName(const std::string& name) { vision_output_name_ = name; }

/* ---------------------------
   Static helpers
   --------------------------- */
float ClipOnnx::cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) throw std::runtime_error("cosineSimilarity: vector size mismatch");
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    double denom = std::sqrt(na) * std::sqrt(nb);
    if (denom <= 1e-12) return 0.0f;
    return static_cast<float>(dot / denom);
}

} // namespace clip_onnx