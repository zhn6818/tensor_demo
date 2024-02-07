#pragma once
#include <iostream>
#include <vector>

struct Config
{
    std::string onnx_file;
    std::string engine_file;
    int max_keypoints{};
    double keypoint_threshold{};
    int remove_borders{};
    int dla_core{};
    std::vector<std::string> input_tensor_names;
    std::vector<std::string> output_tensor_names;
};

