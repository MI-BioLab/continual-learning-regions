#include <torch/torch.h>
#include "rtabmap/core/region/models/BasicBlock.h"

namespace rtabmap
{

    torch::nn::Conv2dOptions conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
                                          int64_t stride, int64_t padding, bool with_bias)
    {
        torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size).stride(stride).padding(padding).bias(with_bias);
        return conv_options;
    }

    BasicBlock::BasicBlock(int64_t inplanes, int64_t planes, int64_t stride,
                           torch::nn::Sequential downsample)
        : conv1(conv_options(inplanes, planes, 3, stride, 1)),
          bn1(planes),
          relu(torch::nn::ReLUOptions().inplace(true)),
          conv2(conv_options(planes, planes, 3, 1, 1)),
          bn2(planes),
          downsample(downsample)

    {
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("relu", relu);
        register_module("conv2", conv2);
        register_module("bn2", bn2);
        if (!downsample->is_empty())
        {
            register_module("downsample", downsample);
        }
    }

    torch::Tensor BasicBlock::forward(torch::Tensor x)
    {
        at::Tensor residual(x.clone());

        x = conv1->forward(x);
        x = bn1->forward(x);
        x = relu->forward(x);

        x = conv2->forward(x);
        x = bn2->forward(x);

        if (!downsample->is_empty())
        {
            residual = downsample->forward(residual);
        }

        x += residual;
        x = relu->forward(x);

        return x;
    }

    torch::nn::Sequential make_layer(int64_t planes, int64_t inplanes, int64_t blocks, int64_t stride)
    {
        torch::nn::Sequential downsample;
        if (stride != 1 or inplanes != planes)
        {
            downsample = torch::nn::Sequential(
                torch::nn::Conv2d(conv_options(inplanes, planes, 1, stride)),
                torch::nn::BatchNorm2d(planes));
        }
        torch::nn::Sequential layers;
        layers->push_back(BasicBlock(inplanes, planes, stride, downsample));
        inplanes = planes;
        for (int64_t i = 0; i < blocks; i++)
        {
            layers->push_back(BasicBlock(inplanes, planes));
        }

        return layers;
    }
}