#include "rtabmap/core/region/models/FreezedPart.h"

namespace rtabmap
{
    FreezedPartImpl::FreezedPartImpl() : conv1(conv_options(3, 64, 7, 2, 3)),
                                         bn1(64),
                                         relu(torch::nn::ReLUOptions().inplace(true)),
                                         maxpool(torch::nn::MaxPool2dOptions(3).stride(2).padding(1)),
                                         layer1(make_layer(64, 64, 1)),
                                         layer2(make_layer(128, 64, 1, 2)),
                                         layer3(make_layer(256, 128, 1, 2))

    {
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("relu", relu);
        register_module("maxpool", maxpool);
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);
    }

    torch::Tensor FreezedPartImpl::forward(const torch::Tensor &input)
    {
        torch::Tensor x = this->conv1->forward(input);
        x = this->bn1->forward(x);
        x = this->relu->forward(x);
        x = this->maxpool->forward(x);
        x = this->layer1->forward(x);
        x = this->layer2->forward(x);
        x = this->layer3->forward(x);
        return x;
    }

}