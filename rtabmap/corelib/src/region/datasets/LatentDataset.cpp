#include "rtabmap/core/region/datasets/LatentDataset.h"
#include "rtabmap/utilite/ULogger.h"
#include "rtabmap/utilite/UProcessInfo.h"

namespace rtabmap
{
    LatentDataset::LatentDataset()
    {
    }

    LatentDataset::LatentDataset(const std::vector<size_t> &ids,
                                 const torch::Tensor &features,
                                 const torch::Tensor &labels) : _ids(ids),
                                                                _features(features),
                                                                _labels(labels)
    {
        for (int i = 0; i < ids.size(); i++)
        {
            this->_id_index.insert({ids[i], i});
        }

        std::tuple<at::Tensor, at::Tensor, at::Tensor> unique_values = at::_unique2(this->_labels, true, false, true);
        this->_classes_in_dataset = std::get<0>(unique_values);
        this->_samples_per_class = std::get<2>(unique_values);
    }

    torch::data::Example<torch::Tensor, torch::Tensor> LatentDataset::get(size_t index)
    {
        return {this->_features[index], this->_labels[index]};
    }

    void LatentDataset::update_label(size_t id, const torch::Tensor &new_label)
    {
        if (this->_id_index.count(id))
        {
            this->_labels[this->_id_index[id]] = new_label;
            ULOGGER_DEBUG("Label at id=%d updated with new_label=%d", (int)id, new_label.item().toInt());
            return;
        }
        ULOGGER_DEBUG("Label not updated, id=%d", (int)id);
    }

    // std::shared_ptr<LatentDataset> LatentDataset::concat(const std::shared_ptr<LatentDataset> &other)
    // {
    //     ULOGGER_DEBUG("LatendDataset::concat, this size=%d, other size=%d", (int)this->size(), (int)other->size())
    //     std::vector<size_t> new_ids;
    //     std::vector<torch::Tensor> new_features;
    //     std::vector<torch::Tensor> new_labels;

    //     for (const auto &i : this->_id_index)
    //     {
    //         new_ids.emplace_back(i.first);
    //         auto data = this->get(i.second);
    //         new_features.emplace_back(data.data);
    //         new_labels.emplace_back(data.target);
    //     }

    //     for (const auto &i : other->id_index())
    //     {
    //         if (!this->_id_index.count(i.first))
    //         {
    //             new_ids.emplace_back(i.first);
    //             auto data = other->get(i.second);
    //             new_features.emplace_back(data.data);
    //             new_labels.emplace_back(data.target);
    //         }
    //     }
    //     ULOGGER_DEBUG("LatendDataset::concat, new size=%d", (int)new_ids.size());
    //     return std::make_shared<LatentDataset>(new_ids, torch::stack(new_features), torch::stack(new_labels));
    // }

    // std::shared_ptr<LatentDataset> LatentDataset::subset(const at::Tensor &indices)
    // {
    //     ULOGGER_DEBUG("LatendDataset::subset, indices size=%d", (int)indices.size(0));
    //     std::vector<size_t> new_ids(indices.size(0));
    //     std::vector<torch::Tensor> new_features(indices.size(0));
    //     std::vector<torch::Tensor> new_labels(indices.size(0));

    //     //cannot vectorize since ids is vector
    //     for (int i = 0; i < indices.size(0); i++)
    //     {
    //         new_ids[i] = this->_ids[indices[i].item().toLong()];
    //         new_freezed_features[i] = this->_features[indices[i].item().toLong()];
    //         new_labels[i] = this->_labels[indices[i].item().toLong()];
    //     }

    //     return std::make_shared<LatentDataset>(new_ids, torch::stack(new_features), torch::stack(new_labels));
    // }

    void LatentDataset::concat(const std::shared_ptr<LatentDataset> &other, std::shared_ptr<LatentDataset> &concat_dataset)
    {
        ULOGGER_DEBUG("This size=%d, other size=%d", (int)this->size().value_or(0), (int)other->size().value_or(0));
        ULOGGER_DEBUG("RAM usage before concat=%ld", UProcessInfo::getMemoryUsage());
        std::vector<size_t> new_ids;
        std::vector<torch::Tensor> new_features;
        std::vector<torch::Tensor> new_labels;

        for (const auto &i : this->_id_index)
        {
            new_ids.emplace_back(i.first);
            auto data = this->get(i.second);
            new_features.emplace_back(data.data);
            new_labels.emplace_back(data.target);
        }

        for (const auto &i : other->id_index())
        {
            if (!this->_id_index.count(i.first))
            {
                new_ids.emplace_back(i.first);
                auto data = other->get(i.second);
                new_features.emplace_back(data.data);
                new_labels.emplace_back(data.target);
            }
            else
            {
                ULOGGER_DEBUG("Found identical id=%d", (int)i.first);
            }
        }
        ULOGGER_DEBUG("New size=%d", (int)new_ids.size());
        concat_dataset = std::make_shared<LatentDataset>(new_ids, torch::stack(new_features), torch::stack(new_labels));
        ULOGGER_DEBUG("RAM usage after concat=%ld", UProcessInfo::getMemoryUsage());
    }

    void LatentDataset::subset(const at::Tensor &indices, std::shared_ptr<LatentDataset> &subset_dataset)
    {
        ULOGGER_DEBUG("LatendDataset::subset, indices size=%d", (int)indices.size(0));
        ULOGGER_DEBUG("RAM usage before subset=%ld", UProcessInfo::getMemoryUsage());
        std::vector<size_t> new_ids(indices.size(0));
        std::vector<torch::Tensor> new_features(indices.size(0));
        std::vector<torch::Tensor> new_labels(indices.size(0));

        // cannot vectorize since ids is vector
        for (int i = 0; i < indices.size(0); i++)
        {
            new_ids[i] = this->_ids[indices[i].item().toLong()];
            new_features[i] = this->_features[indices[i].item().toLong()];
            new_labels[i] = this->_labels[indices[i].item().toLong()];
        }

        subset_dataset = std::make_shared<LatentDataset>(new_ids, torch::stack(new_features), torch::stack(new_labels));
        ULOGGER_DEBUG("RAM usage after subset=%ld", UProcessInfo::getMemoryUsage());
    }

    void LatentDataset::concat_datasets(const std::vector<std::shared_ptr<LatentDataset>> &datasets, std::shared_ptr<LatentDataset> &concat_dataset)
    {
        if (datasets.size() == 0)
        {
            concat_dataset = std::make_shared<LatentDataset>();
            return;
        }
        concat_dataset = datasets[0];
        for (size_t i = 1; i < datasets.size(); i++)
        {
            concat_dataset->concat(datasets[i], concat_dataset);
        }
    }

}
