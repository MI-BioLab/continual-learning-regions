#include "rtabmap/core/region/models/FeatureExtractor.h"
#include "rtabmap/utilite/ULogger.h"

namespace rtabmap
{
    FeatureExtractorImpl::FeatureExtractorImpl()
    {
        register_module("freezed_part", freezed_part);
        register_module("trainable_part", trainable_part);
    }

    torch::Tensor FeatureExtractorImpl::extract_freezed_features(const torch::Tensor &input)
    {
        {
            torch::NoGradGuard no_grad;
            return this->freezed_part->forward(input);
        }
    }

    torch::Tensor FeatureExtractorImpl::extract_trainable_features(const torch::Tensor &input)
    {
        torch::Tensor x = this->trainable_part->forward(input);
        x = torch::flatten(x, 1);
        return x;
    }

    void FeatureExtractorImpl::train(bool on)
    {
        torch::nn::Module::train(on);
        this->freezed_part->eval();
    }

}