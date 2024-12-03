#include "rtabmap/core/region/storage_policy/ReservoirSamplingBuffer.h"

#include "rtabmap/utilite/ULogger.h"
#include "rtabmap/utilite/UProcessInfo.h"

namespace rtabmap
{
    ReservoirSamplingBuffer::ReservoirSamplingBuffer(size_t max_size) : Buffer::Buffer(max_size)
    {
    }

    void ReservoirSamplingBuffer::update(const std::shared_ptr<LatentDataset> &new_data)
    {
        ULOGGER_DEBUG("Buffer update on new data of size=%d", (int)new_data->size().value_or(0));
        ULOGGER_DEBUG("RAM usage before buffer update=%ld", UProcessInfo::getMemoryUsage());
        at::Tensor new_weights = at::rand({static_cast<int64_t>(new_data->size().value())});
        at::Tensor cat_weights = at::cat({new_weights, this->_buffer_weights});
        std::shared_ptr<LatentDataset> cat_data;
        new_data->concat(this->_buffer, cat_data);
        std::tuple<at::Tensor, at::Tensor> sorted_data = cat_weights.sort(0, true);
        at::Tensor sorted_weights = std::get<0>(sorted_data);
        at::Tensor sorted_indices = std::get<1>(sorted_data);
        at::Tensor buffer_indices = sorted_indices.slice(0, 0, this->_max_size);
        cat_data->subset(buffer_indices, this->_buffer);
        this->_buffer_weights = sorted_weights.slice(0, 0, this->_max_size);
        ULOGGER_DEBUG("Buffer size after update=%d", (int)this->_buffer->size().value_or(0));
        ULOGGER_DEBUG("RAM usage after buffer update=%ld", UProcessInfo::getMemoryUsage());
    }

    void ReservoirSamplingBuffer::resize(size_t new_size)
    {
        ULOGGER_DEBUG("Buffer resizing of dimension=%d", (int)new_size);
        ULOGGER_DEBUG("RAM usage before buffer resizing=%ld", UProcessInfo::getMemoryUsage());
        this->_max_size = new_size;
        if (this->_buffer->size().value_or(0) <= this->_max_size)
        {
            return;
        }
        this->_buffer->subset(torch::arange(torch::Scalar(static_cast<int64_t>(this->_max_size))), this->_buffer);
        this->_buffer_weights = this->_buffer_weights.slice(0, 0, this->_max_size);
        ULOGGER_DEBUG("Buffer size after resize=%d", (int)this->_buffer->size().value_or(0));
        ULOGGER_DEBUG("RAM usage after buffer resizing=%ld", UProcessInfo::getMemoryUsage());
    }

    const at::Tensor &ReservoirSamplingBuffer::buffer_weights() const
    {
        return this->_buffer_weights;
    }   

    void ReservoirSamplingBuffer::get_ids_in_memory(std::unordered_set<int> &ids_in_memory) const
    {
        ids_in_memory.insert(this->_buffer->ids().begin(), this->_buffer->ids().end());
    }

    // void ReservoirSamplingBuffer::updateLabels(std::unordered_map<int, std::pair<int, int>> &signatures_moved)
    // {
    //     for(const auto &id_region : signatures_moved)
    //     {
    //         this->_buffer->update_label(id_region.first, id_region.second.second);
    //     }
        
    // }

}