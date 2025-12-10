#pragma once

#include <string>
#include <vector>
#include <memory>

#include <onnxruntime_cxx_api.h>

namespace clip_onnx {

/**
 * CLIP ONNX wrapper for separated text & vision encoder models.
 */
class ClipOnnx {
public:
    /**
     * Construct wrapper and load two ONNX models:
     *  - text_model_path: text encoder
     *  - vision_model_path: vision encoder
     */
    ClipOnnx(const std::string& text_model_path,
             const std::string& vision_model_path,
             bool device_gpu = false,
             int image_size = 224);

    ~ClipOnnx();

    // Encode text -> embedding
    std::vector<float> encodeText(const std::string& text);

    // Encode image -> embedding
    std::vector<float> encodeImageFromFile(const std::string& image_path);

    // Encode image from RGB buffer -> embedding
    std::vector<float> encodeImage(const std::vector<uint8_t>& rgb_data, 
                                    int width, 
                                    int height);

    // Cosine similarity utility
    static float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b);

    // Override node names (if needed)
    void setTextInputName(const std::string& name);
    void setTextOutputName(const std::string& name);
    void setVisionInputName(const std::string& name);
    void setVisionOutputName(const std::string& name);

private:
    // Internal session runners
    std::vector<float> runTextSession(const std::vector<int64_t>& input_ids,
                                      const std::vector<int64_t>& input_shape);

    std::vector<float> runVisionSession(const std::vector<float>& image_tensor,
                                        const std::vector<int64_t>& input_shape);

    // Utils
    std::vector<float> l2Normalize(const std::vector<float>& v);
    std::vector<int64_t> pythonTokenize(const std::string& text);
    std::vector<float> loadAndPreprocessImage(const std::string& path);

    // Preprocess RGB buffer
    std::vector<float> preprocessRGBBuffer(const std::vector<uint8_t>& rgb_data,
                                            int width,
                                            int height);

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::AllocatorWithDefaultOptions allocator_;

    std::unique_ptr<Ort::Session> text_session_;
    std::unique_ptr<Ort::Session> vision_session_;

    bool use_gpu_;
    int image_size_;

    // Node names (defaults for CLIP text/vision models)
    std::string text_input_name_ = "input_ids";
    std::string text_output_name_ = "text_embeddings";

    std::string vision_input_name_ = "pixel_values";
    std::string vision_output_name_ = "image_embeddings";

    // CLIP normalization parameters
    const std::array<float, 3> mean_{0.48145466f, 0.4578275f, 0.40821073f};
    const std::array<float, 3> std_{0.26862954f, 0.26130258f, 0.27577711f};
};

} // namespace clip_onnx
