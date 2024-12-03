#pragma once

#ifndef LATENT_DATASET_H
#define LATENT_DATASET_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

namespace rtabmap
{
    class LatentDataset : public torch::data::Dataset<LatentDataset>
    {

    public:
        using Map = std::unordered_map<size_t, size_t>;

        LatentDataset();

        LatentDataset(const std::vector<size_t> &ids,
                      const torch::Tensor &features,
                      const torch::Tensor &labels);

        inline torch::optional<size_t> size() const override
        {
            return this->_ids.size();
        }
        ExampleType get(size_t index) override;

        inline const Map &id_index() const { return this->_id_index; }
        inline const std::vector<size_t> &ids() const { return this->_ids; }
        inline const torch::Tensor &features() const { return this->_features; }
        inline const torch::Tensor &labels() const { return this->_labels; }
        inline const torch::Tensor &classes_in_dataset() const { return this->_classes_in_dataset; }
        inline const torch::Tensor &samples_per_class() const { return this->_samples_per_class; }

        void update_label(size_t id, const torch::Tensor &new_label);

        // std::shared_ptr<LatentDataset> concat(const std::shared_ptr<LatentDataset> &other);
        // std::shared_ptr<LatentDataset> subset(const at::Tensor &indices);

        void concat(const std::shared_ptr<LatentDataset> &other, std::shared_ptr<LatentDataset> &concat_dataset);
        void subset(const at::Tensor &indices, std::shared_ptr<LatentDataset> &subset_dataset);

        static void concat_datasets(const std::vector<std::shared_ptr<LatentDataset>> &datasets, std::shared_ptr<LatentDataset> &concat_dataset);

    private:
        Map _id_index;
        std::vector<size_t> _ids;
        torch::Tensor _features;
        torch::Tensor _labels;

        torch::Tensor _classes_in_dataset;
        torch::Tensor _samples_per_class;
    };
}

#endif