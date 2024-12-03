#pragma once

#ifndef BASE_BUFFER_H
#define BASE_BUFFER_H

#include "rtabmap/core/region/datasets/LatentDataset.h"

namespace rtabmap
{
    class Buffer
    {
    public:
        Buffer(size_t max_size);

        inline const std::shared_ptr<LatentDataset> &buffer() const { return this->_buffer; }
        void buffer(const std::shared_ptr<LatentDataset> &new_buffer) { this->_buffer = new_buffer; }

        virtual void update(const std::shared_ptr<LatentDataset> &new_data) = 0;
        virtual void resize(size_t new_size) = 0;
        virtual void get_ids_in_memory(std::unordered_set<int> &ids_in_memory) const = 0;
        // virtual void updateLabels(std::unordered_map<int, std::pair<int, int>> &signatures_moved) = 0;

    protected:
        size_t _max_size;
        std::shared_ptr<LatentDataset> _buffer;
    };
}

#endif