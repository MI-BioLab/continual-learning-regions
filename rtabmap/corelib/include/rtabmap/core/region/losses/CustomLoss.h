#pragma once

#ifndef CUSTOM_LOSS_H
#define CUSTOM_LOSS_H

#include <torch/torch.h>

namespace rtabmap
{

    struct CustomLossImpl : public torch::nn::CrossEntropyLossImpl
    {
        CustomLossImpl(torch::nn::CrossEntropyLossOptions options = {}) : torch::nn::CrossEntropyLossImpl(options) {}

        virtual torch::Tensor compute(const torch::Tensor &input, const torch::Tensor &target) = 0;
    };

    TORCH_MODULE(CustomLoss);

}
#endif