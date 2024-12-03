#include "rtabmap/core/region/losses/FocalLoss.h"

namespace rtabmap
{
    FocalLossImpl::FocalLossImpl(float gamma) : gamma(torch::Scalar((double)gamma)),
                                                CustomLossImpl(torch::nn::CrossEntropyLossOptions().reduction(torch::kNone))
                                                                            
    {}

    torch::Tensor FocalLossImpl::compute(const torch::Tensor &input,
                                         const torch::Tensor &target)
    {
        torch::Tensor weighted_cross_entropy = torch::nn::functional::cross_entropy(input, target, this->options);
        torch::Tensor confidence = torch::exp(-weighted_cross_entropy);
        torch::Tensor focal_loss = torch::pow((1 - confidence), this->gamma) * weighted_cross_entropy;
        return torch::mean(focal_loss);
    
    }


}