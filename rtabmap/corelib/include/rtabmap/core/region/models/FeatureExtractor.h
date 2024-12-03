#pragma once

#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <torch/torch.h>
#include <torch/script.h>
#include "rtabmap/core/region/models/FreezedPart.h"
#include "rtabmap/core/region/models/TrainablePart.h"

namespace rtabmap
{
    struct FeatureExtractorImpl : torch::nn::Module
    {

        FeatureExtractorImpl();

        FreezedPart freezed_part;
        TrainablePart trainable_part;

        torch::Tensor extract_freezed_features(const torch::Tensor &input);
        torch::Tensor extract_trainable_features(const torch::Tensor &input);

        void train(bool on = true) override;
    };

    TORCH_MODULE(FeatureExtractor);
}

#endif