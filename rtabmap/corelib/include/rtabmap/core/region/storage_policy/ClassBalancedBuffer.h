#pragma once

#ifndef CLASS_BALANCED_BUFFER_H
#define CLASS_BALANCED_BUFFER_H

#include "rtabmap/core/region/storage_policy/Buffer.h"
#include "rtabmap/core/region/storage_policy/ReservoirSamplingBuffer.h"
#include "rtabmap/core/region/datasets/LatentDataset.h"
#include <torch/torch.h>

namespace rtabmap
{
    class ClassBalancedBuffer : public Buffer
    {
    public:
        ClassBalancedBuffer(size_t max_size);

        virtual void update(const std::shared_ptr<LatentDataset> &new_data) override;
        virtual void resize(size_t new_size) override; // TODO expandible?
        virtual const std::unordered_map<size_t, std::shared_ptr<ReservoirSamplingBuffer>> &buffer_groups() const;
        virtual void get_ids_in_memory(std::unordered_set<int> &ids_in_memory) const;
        
        // virtual void updateLabels(std::unordered_map<int, std::pair<int, int>> &signatures_moved);

    private:
        void _update_buffer();
        size_t _get_group_length(); // TODO can be tuned on total example for each class

        std::unordered_map<size_t, std::shared_ptr<ReservoirSamplingBuffer>> _buffer_groups;
        std::set<size_t> _seen_classes;
    };
}

#endif