#include <gtest/gtest.h>
#include <filesystem>
#include <memory>
#include <iostream>

#include "../src/clip_onnx.h"
using clip_onnx::ClipOnnx;

namespace fs = std::filesystem;

static const fs::path TEXT_MODEL  = "../../models/clip_text_sim.onnx";
static const fs::path VISION_MODEL = "../../models/clip_vision_sim.onnx";
static const fs::path TEST_IMAGE = "assets/test.jpg";   // test image (optional)

class ClipOnnxCpuTest : public ::testing::Test {
protected:
    std::unique_ptr<ClipOnnx> clip;

    void SetUp() override {
        std::cout << "TEXT_MODEL: " << TEXT_MODEL << " exists? " 
                  << fs::exists(TEXT_MODEL) << std::endl;
        std::cout << "VISION_MODEL: " << VISION_MODEL << " exists? "
                  << fs::exists(VISION_MODEL) << std::endl;

        ASSERT_TRUE(fs::exists(TEXT_MODEL)) 
            << "Text model not found: " << TEXT_MODEL.string();
        ASSERT_TRUE(fs::exists(VISION_MODEL)) 
            << "Vision model not found: " << VISION_MODEL.string();

        clip = std::make_unique<ClipOnnx>(
            TEXT_MODEL.string(),
            VISION_MODEL.string(),
            false,    // CPU mode
            false     // TensorRT disabled
        );
    }
};
