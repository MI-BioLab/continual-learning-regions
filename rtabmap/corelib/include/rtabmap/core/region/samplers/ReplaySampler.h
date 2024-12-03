#pragma once

#ifndef REPLAY_SAMPLER_H
#define REPLAY_SAMPLER_H

#include <torch/torch.h>

namespace rtabmap
{

    class ReplaySampler : public torch::data::samplers::Sampler<>
    {
    public:
        explicit ReplaySampler(size_t current_experience_size,
                               size_t replay_memory_size,
                               size_t batch_size,
                               size_t replay_memory_batch_size,
                               torch::Dtype index_dtype = torch::kInt64);

        /// Resets the `ReplaySampler` to a new set of indices.
        void reset(torch::optional<size_t> new_size = torch::nullopt) override;

        /// Returns the next batch of indices.
        torch::optional<std::vector<size_t>> next(size_t batch_size) override;

        /// Serializes the `ReplaySampler` to the `archive`.
        void save(torch::serialize::OutputArchive &archive) const override;

        /// Deserializes the `ReplaySampler` from the `archive`.
        void load(torch::serialize::InputArchive &archive) override;

        /// Returns the current index of the `ReplaySampler`.
        size_t index() const noexcept;

        // inline void current_experience_size(int64_t current_experience_size) { this->_current_experience_size = current_experience_size; }
        // inline void replay_memory_size(int64_t replay_memory_size) { this->_replay_memory_size = replay_memory_size; }
        // inline void batch_size(int64_t batch_size) { this->_batch_size = batch_size; }
        // inline void replay_memory_batch_size(int64_t replay_memory_batch_size) { this->_replay_memory_batch_size = replay_memory_batch_size; }

    private:
        size_t _current_experience_size;
        size_t _replay_memory_size;
        size_t _batch_size;
        size_t _replay_memory_batch_size;
        std::vector<size_t> _current_experience_indices;
        std::vector<size_t> _replay_memory_indices;
        size_t _index = 0;
        size_t _max_index;
        torch::Dtype _index_dtype;
    };
}

#endif