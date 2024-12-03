#pragma once

#ifndef EXPERIENCE_DATASET_H
#define EXPERIENCE_DATASET_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

namespace rtabmap
{

    class ExperienceDataset : public torch::data::Dataset<ExperienceDataset>
    {
    public:
        ExperienceDataset(const std::vector<size_t> &ids,
                          const std::vector<cv::Mat> &images,
                          const std::vector<size_t> &labels,
                          const int64_t &image_width = 224,
                          const int64_t &image_height = 224);

        inline torch::optional<size_t> size() const override
        {
            return this->_ids.size();
        }

        torch::data::Example<> get(size_t index) override;

        inline const std::vector<size_t> &ids() const { return this->_ids; }
        inline const torch::Tensor &images() const { return this->_images; }
        inline const torch::Tensor &labels() const { return this->_labels; }

    private:
        std::vector<size_t> _ids;
        torch::Tensor _images;
        torch::Tensor _labels;
    };

}

#endif