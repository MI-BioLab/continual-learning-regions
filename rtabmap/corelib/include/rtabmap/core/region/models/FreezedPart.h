#pragma once

#ifndef FREEZED_PART_H
#define FREEZED_PART_H

#include <torch/torch.h>
#include <torch/script.h>
#include "rtabmap/core/region/models/BasicBlock.h"

namespace rtabmap
{
    struct FreezedPartImpl : torch::nn::Module
    {
        int64_t inplanes = 64;

        torch::nn::Conv2d conv1;
        torch::nn::BatchNorm2d bn1;
        torch::nn::ReLU relu;
        torch::nn::MaxPool2d maxpool;
        torch::nn::Sequential layer1;
        torch::nn::Sequential layer2;
        torch::nn::Sequential layer3;
        
        FreezedPartImpl();

        torch::Tensor forward(const torch::Tensor &input);
    };

    TORCH_MODULE(FreezedPart);

}

#endif