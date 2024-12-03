#pragma once

#ifndef FOCAL_LOSS_H
#define FOCAL_LOSS_H

#include <torch/torch.h>
#include "rtabmap/core/region/losses/CustomLoss.h"

namespace rtabmap
{

    struct FocalLossImpl : public CustomLossImpl
    {
        torch::Scalar gamma;

        FocalLossImpl(float gamma);
                      
        virtual torch::Tensor compute(const torch::Tensor &input, const torch::Tensor &target);


    };

    TORCH_MODULE(FocalLoss);

}
#endif