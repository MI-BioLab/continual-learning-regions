
#pragma once

#ifndef UTILS_H
#define UTILS_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

namespace rtabmap
{
    torch::Tensor image_to_tensor(const cv::Mat &image, const int64_t &image_width, const int64_t &image_height);
    torch::Tensor compute_simple_weights(const torch::Tensor &samples_per_class);
    torch::Tensor compute_effective_weights(const torch::Tensor &samples_per_class, float beta = 0.999);
    void save_tensor_serialized(const std::string &file_path, const c10::IValue &ivalue);
    c10::IValue load_tensor_deserialized(const std::string &file_path);
}

#endif