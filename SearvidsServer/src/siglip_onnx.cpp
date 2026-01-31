/*
 * Copyright (c) 2025 Hyeonjun Park (phjun7150@gmail.com)
 * All rights reserved.
 */

// src/siglip_onnx.cpp
#include "siglip_onnx.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <stdexcept>

#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#endif

using namespace std::string_literals;

namespace {
// SigLIP / Optimum ONNX names
static constexpr const char *TEXT_INPUT = "input_ids";
static constexpr const char *TEXT_OUTPUT =
    "pooler_output"; // or "last_hidden_state" if pooler not present
static constexpr const char *VISION_INPUT = "pixel_values";
static constexpr const char *VISION_OUTPUT =
    "pooler_output"; // or "image_embeds"
} // namespace

namespace siglip_onnx {

static std::wstring to_wstring_if_windows(const std::string &s) {
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
SiglipOnnx::SiglipOnnx(const std::string &text_model_path,
                       const std::string &vision_model_path, bool device_gpu,
                       int image_size)
    : env_(ORT_LOGGING_LEVEL_WARNING, "siglip_onnx"), use_gpu_(device_gpu),
      image_size_(image_size), text_model_path_(text_model_path),
      vision_model_path_(vision_model_path) {
  // Try to enable TensorRT initially if GPU is requested
  reloadSession(use_gpu_);
}

SiglipOnnx::~SiglipOnnx() = default;

void SiglipOnnx::reloadSession(bool use_tensorrt) {
  if (text_session_)
    text_session_.reset();
  if (vision_session_)
    vision_session_.reset();

  // Refresh session options
  session_options_ = Ort::SessionOptions();
  session_options_.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);

  tensorrt_enabled_ = use_tensorrt;

  if (use_gpu_) {
    // Always try CUDA first
    try {
      OrtCUDAProviderOptions cuda_opts;
      session_options_.AppendExecutionProvider_CUDA(cuda_opts);
      std::cout << "[siglip_onnx] Appended CUDA Provider\n";

      if (tensorrt_enabled_) {
#ifdef USE_TENSORRT
        try {
          Ort::TensorRTProviderOptions trt_options;
          std::unordered_map<std::string, std::string> options;
          options["device_id"] = "0";
          options["trt_fp16_enable"] =
              "0"; // Disable FP16 to debug negative scores
          options["trt_engine_cache_enable"] = "1";
          options["trt_engine_cache_path"] = "tensorrt_cache";
          trt_options.Update(options);
          session_options_.AppendExecutionProvider_TensorRT_V2(*trt_options);
          std::cout << "[siglip_onnx] Appended TensorRT Provider\n";
        } catch (const std::exception &e) {
          std::cerr << "[siglip_onnx] Failed to append TensorRT: " << e.what()
                    << "\n";
          tensorrt_enabled_ = false;
        }
#else
        std::cerr << "[siglip_onnx] TensorRT requested but not compiled in.\n";
        tensorrt_enabled_ = false;
#endif
      }
    } catch (...) {
      std::cerr << "[siglip_onnx] Error appending GPU providers, falling back "
                   "to CPU\n";
      use_gpu_ = false;
    }
  }

  try {
#ifdef _WIN32
    std::wstring wtext = to_wstring_if_windows(text_model_path_);
    std::wstring wvision = to_wstring_if_windows(vision_model_path_);
    text_session_ =
        std::make_unique<Ort::Session>(env_, wtext.c_str(), session_options_);
    vision_session_ =
        std::make_unique<Ort::Session>(env_, wvision.c_str(), session_options_);
#else
    text_session_ = std::make_unique<Ort::Session>(
        env_, text_model_path_.c_str(), session_options_);
    vision_session_ = std::make_unique<Ort::Session>(
        env_, vision_model_path_.c_str(), session_options_);
#endif
    // Re-detect input/output names if needed (omitted for brevity as they are
    // likely constant per model) Ideally we should re-run the detection logic
    // here or store it once. For fallback safety, we should re-run it.

    Ort::AllocatorWithDefaultOptions allocator;
    if (text_session_->GetInputCount() > 0) {
      text_input_name_ =
          text_session_->GetInputNameAllocated(0, allocator).get();
    }
    if (text_session_->GetOutputCount() > 0) {
      // Re-detect 2D output
      auto find_2d = [&](Ort::Session *sess, std::string &out_name) {
        for (size_t i = 0; i < sess->GetOutputCount(); ++i) {
          auto shape =
              sess->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
          if (shape.size() == 2) {
            out_name = sess->GetOutputNameAllocated(i, allocator).get();
            return;
          }
        }
        out_name = sess->GetOutputNameAllocated(0, allocator).get();
      };
      find_2d(text_session_.get(), text_output_name_);
      find_2d(vision_session_.get(), vision_output_name_);
    }

    // Log available outputs
    std::cout << "[siglip_onnx] Vision Outputs: ";
    for (size_t i = 0; i < vision_session_->GetOutputCount(); ++i) {
      auto name = vision_session_->GetOutputNameAllocated(i, allocator);
      std::cout << name.get() << " ["
                << vision_session_->GetOutputTypeInfo(i)
                       .GetTensorTypeAndShapeInfo()
                       .GetShape()
                       .size()
                << "D] ";
    }
    std::cout << "\n";
    std::cout << "[siglip_onnx] Text Outputs: ";
    for (size_t i = 0; i < text_session_->GetOutputCount(); ++i) {
      auto name = text_session_->GetOutputNameAllocated(i, allocator);
      auto shape = text_session_->GetOutputTypeInfo(i)
                       .GetTensorTypeAndShapeInfo()
                       .GetShape();
      std::cout << name.get() << " [";
      for (size_t k = 0; k < shape.size(); ++k)
        std::cout << (k > 0 ? "x" : "") << shape[k];
      std::cout << "] ";

      std::string n_str = name.get();
      bool is_pooler = (n_str.find("pooler_output") != std::string::npos ||
                        n_str.find("last_hidden_state") != std::string::npos);

      if (is_pooler) {
        text_output_name_ = n_str;
      } else if (shape.size() == 2 &&
                 text_output_name_ ==
                     "text_embeddings") { // Only default if not already set by
                                          // priority
        int64_t dim = shape[1];
        if (dim > 0 && dim < 4096) {
          text_output_name_ = n_str;
        }
      }
    }
    std::cout << "\n[siglip_onnx] Selected Text Output: " << text_output_name_
              << "\n";
    std::cout << "[siglip_onnx] Selected Vision Output: " << vision_output_name_
              << "\n";

    std::cout << "[siglip_onnx] Sessions loaded. TRT="
              << (tensorrt_enabled_ ? "ON" : "OFF") << "\n";

  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to load sessions: "s + e.what());
  }
}

/* ---------------------------
   Text / Vision runners
   --------------------------- */

std::vector<float>
SiglipOnnx::processTextOutput(std::vector<Ort::Value> &output_tensors,
                              Ort::Value &output_tensor) {
  Ort::Value &out = output_tensor;
  float *out_data = out.GetTensorMutableData<float>();
  size_t out_count = out.GetTensorTypeAndShapeInfo().GetElementCount();
  std::vector<float> vec(out_data, out_data + out_count);

  // If output is 3D (batch, seq, dim) we try to collapse by averaging sequence
  // axis.
  auto shape = out.GetTensorTypeAndShapeInfo().GetShape();
  if (shape.size() == 3) {
    // shape: [1, seq, dim]
    int64_t seq = static_cast<int64_t>(shape[1]);
    int64_t dim = static_cast<int64_t>(shape[2]);

    // Use EOS token (last token) embedding
    // SigLIP uses the last token (EOS) for representation
    // Testing confirmed: EOS (0.019) > CLS (-0.009)
    if (seq > 0) {
      int64_t last_idx = seq - 1; // EOS token
      std::vector<float> eos_emb(dim);
      for (int64_t j = 0; j < dim; ++j) {
        eos_emb[j] = vec[last_idx * dim + j];
      }
      return l2Normalize(eos_emb);
    }

    /* Average Pooling (Disabled)
    std::vector<float> avg(dim, 0.0f);
    for (int64_t i = 0; i < seq; ++i) {
      for (int64_t j = 0; j < dim; ++j) {
        avg[j] += vec[i * dim + j];
      }
    }
    for (int64_t j = 0; j < dim; ++j)
      avg[j] /= static_cast<float>(seq);
    return l2Normalize(avg);
    */
    return l2Normalize(vec); // Fallback if seq=0?
  }
  return l2Normalize(vec);
}

std::vector<float>
SiglipOnnx::processVisionOutput(std::vector<Ort::Value> &output_tensors,
                                Ort::Value &output_tensor) {
  Ort::Value &out = output_tensor;
  float *out_data = out.GetTensorMutableData<float>();
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
    for (int64_t j = 0; j < dim; ++j)
      avg[j] /= static_cast<float>(seq);
    return l2Normalize(avg);
  }

  return l2Normalize(vec);
}

/**
 * Run text session. Accepts token ids and returns L2-normalized embedding.
 *
 * token_type_ids, etc. Here we assume a single input "input_ids" and that model
 * returns a single float output that can be averaged over sequence dimension if
 * needed.
 */
std::vector<float>
SiglipOnnx::runTextSession(const std::vector<int64_t> &input_ids,
                           const std::vector<int64_t> &input_shape) {
  if (!text_session_)
    throw std::runtime_error("Text session not initialized");

  // Create CPU memory info
  Ort::MemoryInfo mem_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // Create tensor for input_ids
  Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
      mem_info, const_cast<int64_t *>(input_ids.data()),
      static_cast<size_t>(input_ids.size()), input_shape.data(),
      input_shape.size());

  std::vector<int64_t> attention_mask(input_ids.size(), 1);

  Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
      mem_info, attention_mask.data(), attention_mask.size(),
      input_shape.data(), input_shape.size());

  std::vector<const char *> input_names;
  input_names.reserve(2);
  input_names.push_back(text_input_name_.c_str());

  std::vector<Ort::Value> input_tensors;
  input_tensors.reserve(2);
  input_tensors.push_back(std::move(input_tensor));

  if (text_has_attention_mask_) {
    input_names.push_back("attention_mask");
    input_tensors.push_back(std::move(attention_mask_tensor));
  }

  const char *output_names[] = {text_output_name_.c_str()};

  try {
    auto output_tensors = text_session_->Run(
        Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(),
        input_names.size(), output_names, 1);
    if (output_tensors.empty())
      throw std::runtime_error("Text model produced no outputs");

    return processTextOutput(output_tensors, output_tensors.front());
  } catch (const Ort::Exception &e) {
    if (use_gpu_ && tensorrt_enabled_) {
      std::cerr << "[siglip_onnx] TensorRT failed during Text Run (" << e.what()
                << "). Fallback to CUDA...\n";
      reloadSession(false); // disable TRT
      // Retry
      return runTextSession(input_ids, input_shape);
    }
    throw std::runtime_error(std::string("ONNX Runtime error in text run: ") +
                             e.what());
  }
}

/**
 * Run vision session. Accepts preprocessed image tensor (NCHW float array) and
 * returns L2-normalized embedding.
 */
std::vector<float>
SiglipOnnx::runVisionSession(const std::vector<float> &image_tensor,
                             const std::vector<int64_t> &input_shape) {
  if (!vision_session_)
    throw std::runtime_error("Vision session not initialized");

  Ort::MemoryInfo mem_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      mem_info, const_cast<float *>(image_tensor.data()),
      static_cast<size_t>(image_tensor.size()), input_shape.data(),
      input_shape.size());

  const char *input_names[] = {vision_input_name_.c_str()};
  const char *output_names[] = {vision_output_name_.c_str()};

  try {
    auto output_tensors =
        vision_session_->Run(Ort::RunOptions{nullptr}, input_names,
                             &input_tensor, 1, output_names, 1);
    if (output_tensors.empty())
      throw std::runtime_error("Vision model produced no outputs");

    return processVisionOutput(output_tensors, output_tensors.front());
  } catch (const Ort::Exception &e) {
    if (use_gpu_ && tensorrt_enabled_) {
      std::cerr << "[siglip_onnx] TensorRT failed during Vision Run ("
                << e.what() << "). Fallback to CUDA...\n";
      reloadSession(false); // disable TRT
      // Retry
      return runVisionSession(image_tensor, input_shape);
    }
    throw std::runtime_error(std::string("ONNX Runtime error in vision run: ") +
                             e.what());
  }
}

/* ---------------------------
   Public wrapper functions
   --------------------------- */

std::vector<float> SiglipOnnx::encodeText(const std::string &text) {
  // Use Python tokenizer for better accuracy
  auto ids = pythonTokenize(text);
  if (ids.empty())
    ids.push_back(0);
  std::vector<int64_t> shape = {1, static_cast<int64_t>(ids.size())};
  auto res = runTextSession(ids, shape);

  std::cout << "[siglip_onnx] Text Embedding (First 5): ";
  for (size_t k = 0; k < 5 && k < res.size(); ++k)
    std::cout << res[k] << " ";
  std::cout << "\n";

  return res;
}

std::vector<float>
SiglipOnnx::encodeImageFromFile(const std::string &image_path) {
  auto img_tensor = loadAndPreprocessImage(image_path);
  std::vector<int64_t> shape = {1, 3, image_size_, image_size_};
  auto res = runVisionSession(img_tensor, shape);
  return res;
}

/* ---------------------------
   Utilities
   --------------------------- */

std::vector<float> SiglipOnnx::l2Normalize(const std::vector<float> &v) {
  double sumsq = 0.0;
  for (float x : v)
    sumsq += static_cast<double>(x) * static_cast<double>(x);
  double norm = std::sqrt(sumsq);
  std::vector<float> out(v.size());
  if (norm <= 1e-12)
    return out;
  for (size_t i = 0; i < v.size(); ++i)
    out[i] = static_cast<float>(v[i] / norm);
  return out;
}

std::vector<int64_t> SiglipOnnx::pythonTokenize(const std::string &text) {
  // Escape quotes in text
  std::string escaped_text;
  for (char c : text) {
    if (c == '"')
      escaped_text += "\\\"";
    else
      escaped_text += c;
  }

  // Call python script
  // Assuming tokenizer.py is in /app/tokenizer.py in Docker
  std::string cmd = "python3 /app/tokenizer.py \"" + escaped_text + "\"";

  std::array<char, 128> buffer;
  std::string result;

  // Use popen to run command and capture output
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"),
                                                pclose);
  if (!pipe) {
    std::cerr << "[siglip_onnx] Error: popen() failed for tokenizer\n";
    return {}; // Return empty on failure
  }

  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }

  // Parse result
  std::vector<int64_t> tokens;
  std::stringstream ss(result);
  int64_t token;
  while (ss >> token) {
    tokens.push_back(token);
  }

  if (tokens.empty()) {
    std::cerr << "[siglip_onnx] Warning: Python tokenizer returned empty.\n";
    return {};
  }

  std::cout << "[siglip_onnx] Tokens: ";
  for (auto t : tokens)
    std::cout << t << " ";
  std::cout << "\n";

  return tokens;
}

/**
 * Load an image using OpenCV and preprocess:
 *  - force 3 channels (RGB)
 *  - resize to image_size_ (bilinear/area)
 *  - normalize (mean/std) and produce NCHW float tensor
 */
std::vector<float> SiglipOnnx::loadAndPreprocessImage(const std::string &path) {
  cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
  if (img.empty()) {
    throw std::runtime_error("Failed to load image: " + path);
  }
  // OpenCV loads as BGR, convert to RGB
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

  return preprocessRGBBuffer(
      std::vector<uint8_t>(img.data, img.data + (img.total() * img.elemSize())),
      img.cols, img.rows);
}

/**
 * Preprocess RGB buffer (HWC format) to NCHW float tensor
 * Input: RGB24 buffer (width * height * 3), width, height
 * Output: Normalized NCHW float tensor (3 * image_size_ * image_size_)
 */
std::vector<float>
SiglipOnnx::preprocessRGBBuffer(const std::vector<uint8_t> &rgb_data, int width,
                                int height) {
  if (rgb_data.empty() || width <= 0 || height <= 0) {
    throw std::runtime_error("Invalid image data");
  }

  // Wrap buffer in cv::Mat (no copy)
  cv::Mat src(height, width, CV_8UC3, const_cast<uint8_t *>(rgb_data.data()));

  cv::Mat resized;
  // Use INTER_AREA for downscaling (e.g. 4K -> 224), INTER_LINEAR for upscaling
  int interpolation = (width > image_size_ && height > image_size_)
                          ? cv::INTER_AREA
                          : cv::INTER_LINEAR;
  cv::resize(src, resized, cv::Size(image_size_, image_size_), 0, 0,
             interpolation);

  // Normalize and convert to NCHW
  std::vector<float> output(3 * image_size_ * image_size_);

  // Iterate over pixels
  // Optimized loop could use split() and forEach, but this is clear enough
  for (int y = 0; y < image_size_; ++y) {
    for (int x = 0; x < image_size_; ++x) {
      cv::Vec3b pixel = resized.at<cv::Vec3b>(y, x);
      for (int c = 0; c < 3; ++c) {
        // pixel[c] is R, G, B (since input was RGB)
        float v = static_cast<float>(pixel[c]) / 255.0f;
        v = (v - mean_[c]) / std_[c];

        // NCHW: [channel][y][x]
        output[c * image_size_ * image_size_ + y * image_size_ + x] = v;
      }
    }
  }

  return output;
}

/**
 * Encode image from RGB buffer (wrapper for preprocessRGBBuffer +
 * runVisionSession)
 */
std::vector<float> SiglipOnnx::encodeImage(const std::vector<uint8_t> &rgb_data,
                                           int width, int height) {
  // 1) Preprocess RGB buffer to NCHW tensor
  auto img_tensor = preprocessRGBBuffer(rgb_data, width, height);

  // 2) Run vision session
  std::vector<int64_t> shape = {1, 3, image_size_, image_size_};
  auto res = runVisionSession(img_tensor, shape);
  return res;
}

std::vector<std::vector<float>>
SiglipOnnx::encodeBatch(const std::vector<std::vector<uint8_t>> &batch_rgb,
                        int width, int height) {
  size_t batch_size = batch_rgb.size();
  if (batch_size == 0)
    return {};

  // 1. Preprocess all images
  std::vector<float> input_batch;
  input_batch.reserve(batch_size * 3 * image_size_ * image_size_);

  for (const auto &rgb : batch_rgb) {
    auto chw = preprocessRGBBuffer(rgb, width, height);
    input_batch.insert(input_batch.end(), chw.begin(), chw.end());
  }

  // 2. Prepare Tensor
  std::vector<int64_t> input_shape = {static_cast<int64_t>(batch_size), 3,
                                      image_size_, image_size_};

  Ort::MemoryInfo mem_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      mem_info, input_batch.data(), input_batch.size(), input_shape.data(),
      input_shape.size());

  const char *input_names[] = {vision_input_name_.c_str()};
  const char *output_names[] = {vision_output_name_.c_str()};

  // 3. Run Session
  try {
    if (!vision_session_)
      throw std::runtime_error("Vision session not initialized");

    auto output_tensors =
        vision_session_->Run(Ort::RunOptions{nullptr}, input_names,
                             &input_tensor, 1, output_names, 1);

    Ort::Value &out = output_tensors.front();
    auto shape = out.GetTensorTypeAndShapeInfo().GetShape();
    float *out_data = out.GetTensorMutableData<float>();

    // Handle Output Shape [N, D]
    // Currently assuming [N, D]. If strict SigLIP pooler_output, it is [N, D].
    size_t dim = 0;
    if (shape.size() == 2) {
      dim = shape[1];
    } else if (shape.size() == 3) {
      // [N, seq, dim] -> average?
      // Logic in runVisionSession handles this for N=1.
      // For batch, we'll assume user model outputs pooler [N, D].
      dim = shape[2];
    }

    if (dim == 0) {
      // Fallback: try element count / batch_size
      size_t total = out.GetTensorTypeAndShapeInfo().GetElementCount();
      dim = total / batch_size;
    }

    std::vector<std::vector<float>> results;
    results.reserve(batch_size);

    if (shape.size() == 3) {
      // [Batch, Seq, Dim]
      int64_t seq = static_cast<int64_t>(shape[1]);
      int64_t dim = static_cast<int64_t>(shape[2]);

      for (size_t i = 0; i < batch_size; ++i) {
        std::vector<float> avg(dim, 0.0f);
        // Calculate offset for this batch item
        size_t offset = i * seq * dim;

        for (int64_t s = 0; s < seq; ++s) {
          for (int64_t d = 0; d < dim; ++d) {
            avg[d] += out_data[offset + s * dim + d];
          }
        }
        for (int64_t d = 0; d < dim; ++d) {
          avg[d] /= static_cast<float>(seq);
        }
        results.push_back(l2Normalize(avg));
      }
    } else {
      // Assume [Batch, Dim]
      if (dim == 0)
        dim = out.GetTensorTypeAndShapeInfo().GetElementCount() / batch_size;

      for (size_t i = 0; i < batch_size; ++i) {
        std::vector<float> vec(out_data + i * dim, out_data + (i + 1) * dim);
        auto normalized = l2Normalize(vec);

        // Log first 5 values
        std::cout << "[siglip_onnx] Embedding[" << i << "] (First 5): ";
        for (int k = 0; k < 5 && k < dim; ++k)
          std::cout << normalized[k] << " ";
        std::cout << "\n";

        results.push_back(normalized);
      }
    }

    return results;

  } catch (const std::exception &e) {
    std::cerr << "Batch inference error: " << e.what() << std::endl;
    return {};
  }
}

/* ---------------------------
   Node name setters
   --------------------------- */
void SiglipOnnx::setTextInputName(const std::string &name) {
  text_input_name_ = name;
}
void SiglipOnnx::setTextOutputName(const std::string &name) {
  text_output_name_ = name;
}
void SiglipOnnx::setVisionInputName(const std::string &name) {
  vision_input_name_ = name;
}
void SiglipOnnx::setVisionOutputName(const std::string &name) {
  vision_output_name_ = name;
}

/* ---------------------------
   Static helpers
   --------------------------- */
float SiglipOnnx::cosineSimilarity(const std::vector<float> &a,
                                   const std::vector<float> &b) {
  if (a.size() != b.size())
    throw std::runtime_error("cosineSimilarity: vector size mismatch");
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
    nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  double denom = std::sqrt(na) * std::sqrt(nb);
  if (denom <= 1e-12)
    return 0.0f;
  return static_cast<float>(dot / denom);
}

} // namespace siglip_onnx