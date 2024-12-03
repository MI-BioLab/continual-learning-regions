#include "rtabmap/core/region/models/TrainablePart.h"

namespace rtabmap
{
    TrainablePartImpl::TrainablePartImpl() : layer4(make_layer(512, 256, 1, 2)),
                                             avgpool(torch::nn::AdaptiveAvgPool2dOptions({1, 1}))

    {
        register_module("layer4", layer4);
        register_module("avgpool", avgpool);
    }

    torch::Tensor TrainablePartImpl::forward(const torch::Tensor &input)
    {
        torch::Tensor x = this->layer4->forward(input);
        x = this->avgpool->forward(x);
        return x;
    }

}