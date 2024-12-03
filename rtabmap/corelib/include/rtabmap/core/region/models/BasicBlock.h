#pragma once

#ifndef BASIC_BLOCK_H
#define BASIC_BLOCK_H

#include <torch/torch.h>

namespace rtabmap
{
    torch::nn::Conv2dOptions conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
                                          int64_t stride = 1, int64_t padding = 0, bool with_bias = false);

    struct BasicBlock : torch::nn::Module
    {

        static const int expansion;

        torch::nn::Conv2d conv1;
        torch::nn::BatchNorm2d bn1;
        torch::nn::ReLU relu;
        torch::nn::Conv2d conv2;
        torch::nn::BatchNorm2d bn2;
        torch::nn::Sequential downsample;

        BasicBlock(int64_t inplanes, int64_t planes, int64_t stride = 1,
                   torch::nn::Sequential downsample = torch::nn::Sequential());

        torch::Tensor forward(torch::Tensor x);
    };

    torch::nn::Sequential make_layer(int64_t planes, int64_t inplanes, int64_t blocks, int64_t stride = 1);

}

#endif