#include "rtabmap/core/region/samplers/ReplaySampler.h"
#include "rtabmap/utilite/ULogger.h"

namespace rtabmap
{
    ReplaySampler::ReplaySampler(size_t current_experience_size,
                                 size_t replay_memory_size,
                                 size_t batch_size,
                                 size_t replay_memory_batch_size,
                                 torch::Dtype index_dtype) : _current_experience_size(current_experience_size),
                                                             _replay_memory_size(replay_memory_size),
                                                             _batch_size(batch_size),
                                                             _replay_memory_batch_size(replay_memory_batch_size),
                                                             _index_dtype(index_dtype)
    {
        UASSERT_MSG(this->_current_experience_size > 0, "Current experience size should be > 0");
        this->reset();
    }

    void ReplaySampler::reset(torch::optional<size_t> new_size)
    {
        // Initialize shuffled indices for each dataset
        torch::Tensor current_experience_indices = torch::randperm(this->_current_experience_size, torch::dtype(this->_index_dtype));
        torch::Tensor replay_memory_indices = torch::randperm(this->_replay_memory_size, torch::dtype(this->_index_dtype)) + static_cast<int64_t>(this->_current_experience_size);


        this->_current_experience_indices = std::vector<size_t>(current_experience_indices.data_ptr<int64_t>(), current_experience_indices.data_ptr<int64_t>() + static_cast<int64_t>(this->_current_experience_size));
        this->_replay_memory_indices = std::vector<size_t>(replay_memory_indices.data_ptr<int64_t>(), replay_memory_indices.data_ptr<int64_t>() + static_cast<int64_t>(this->_replay_memory_size));
        
        ULOGGER_DEBUG("Current experience indices size=%d", (int)this->_current_experience_indices.size());
        ULOGGER_DEBUG("Replay memory indices size=%d", (int)this->_replay_memory_indices.size());

        this->_index = 0;
        this->_max_index = std::max(std::ceil(this->_current_experience_size / this->_batch_size), std::ceil(this->_replay_memory_size / this->_replay_memory_batch_size));
    }

    torch::optional<std::vector<size_t>> ReplaySampler::next(size_t batch_size)
    {
        
        if(this->_index > this->_max_index)
        {
            ULOGGER_DEBUG("Sampler index > max index (%d >= %d)", (int)this->_index, (int)this->_max_index);
            return {};
        }
        ULOGGER_DEBUG("Sampler index=%d", (int)this->_index);
        std::vector<size_t> batch_indices;
        size_t current_experience_start_index = this->_index * this->_batch_size;
        // size_t current_experience_stop = (this->_index == this->_max_index) && ((current_experience_start_index + this->_batch_size) % this->_current_experience_size > this->_batch_size) ? this->_batch_size;
        ULOGGER_DEBUG("Current experience from idx %d to idx %d", (int)current_experience_start_index, (int)(current_experience_start_index + this->_batch_size));
        for (size_t i = 0; i < this->_batch_size; i++)
        {
            batch_indices.push_back(this->_current_experience_indices[(current_experience_start_index + i) % this->_current_experience_size]);
        }

        if (this->_replay_memory_size > 0)
        {
            size_t replay_memory_start_index = this->_index * this->_replay_memory_batch_size;
            // size_t replay_memory_stop = this->_index == this->_max_index ? this->_replay_memory_size - (replay_memory_start_index % this->_replay_memory_size) : this->_replay_memory_batch_size;
            ULOGGER_DEBUG("Replay memory from idx %d to idx %d", (int)replay_memory_start_index, (int)(replay_memory_start_index + this->_replay_memory_batch_size));

            for (size_t i = 0; i < this->_replay_memory_batch_size; i++)
            {
                batch_indices.push_back(this->_replay_memory_indices[(replay_memory_start_index + i) % this->_replay_memory_size]);
            }
        }

        this->_index++;
        return torch::optional<std::vector<size_t>>(batch_indices);
    }

    void ReplaySampler::save(torch::serialize::OutputArchive &archive) const
    {
    }

    void ReplaySampler::load(torch::serialize::InputArchive &archive)
    {
    }

    size_t ReplaySampler::index() const noexcept
    {
        return this->_index;
    }

}