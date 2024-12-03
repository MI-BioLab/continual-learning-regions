#pragma once

#ifndef RESERVOIR_SAMPLING_BUFFER_H
#define RESERVOIR_SAMPLING_BUFFER_H

#include <torch/torch.h>
#include "rtabmap/core/region/storage_policy/Buffer.h"
#include "rtabmap/core/region/datasets/LatentDataset.h"

namespace rtabmap
{
class ReservoirSamplingBuffer : public Buffer
{
public:
    ReservoirSamplingBuffer(size_t max_size);

    virtual void update(const std::shared_ptr<LatentDataset> &new_data) override;
    virtual void resize(size_t new_size) override;
    virtual const at::Tensor &buffer_weights() const;
    virtual void get_ids_in_memory(std::unordered_set<int> &ids_in_memory) const;
    // virtual void updateLabels(std::unordered_map<int, std::pair<int, int>> &signatures_moved);

private:
    at::Tensor _buffer_weights;
};
}
#endif