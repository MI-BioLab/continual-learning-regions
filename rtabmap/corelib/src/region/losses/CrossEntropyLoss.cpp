#include "rtabmap/core/region/losses/CrossEntropyLoss.h"

namespace rtabmap
{
    torch::Tensor CrossEntropyLossImpl::compute(const torch::Tensor &input, const torch::Tensor &target)
    {
        return this->forward(input, target);
    }
}