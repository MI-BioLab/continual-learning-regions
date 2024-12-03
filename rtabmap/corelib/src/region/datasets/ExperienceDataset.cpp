#include "rtabmap/core/region/datasets/ExperienceDataset.h"
#include "rtabmap/core/region/utils.h"

namespace rtabmap
{

    ExperienceDataset::ExperienceDataset(const std::vector<size_t> &ids,
                                         const std::vector<cv::Mat> &images,
                                         const std::vector<size_t> &labels,
                                         const int64_t &target_width,
                                         const int64_t &target_height) : _ids(ids) /* : IdDataset::IdDataset(ids) */
    {
        std::vector<torch::Tensor> images_tensor(images.size());
        std::vector<torch::Tensor> labels_tensor(labels.size());

        for (int i = 0; i < images.size(); i++)
        {
            images_tensor[i] = image_to_tensor(images[i], target_width, target_height);
            labels_tensor[i] = torch::tensor(static_cast<float>(labels[i]));
        }

        this->_images = torch::cat(images_tensor);
        this->_labels = torch::stack(labels_tensor);
    }

    torch::data::Example<> ExperienceDataset::get(size_t index)
    {
        return {this->_images[index], this->_labels[index]};
    }
}
