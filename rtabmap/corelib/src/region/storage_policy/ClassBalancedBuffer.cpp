#include "rtabmap/core/region/storage_policy/ClassBalancedBuffer.h"

#include "rtabmap/utilite/ULogger.h"
#include "rtabmap/utilite/UProcessInfo.h"

namespace rtabmap
{
    ClassBalancedBuffer::ClassBalancedBuffer(size_t max_size) : Buffer::Buffer(max_size)
    {
        ULOGGER_DEBUG("New class balanced buffer creation of size %d", (int)_max_size);
    }

    void ClassBalancedBuffer::update(const std::shared_ptr<LatentDataset> &new_data)
    {
        ULOGGER_DEBUG("Buffer update on new data of size=%d", (int)new_data->size().value_or(0));
        ULOGGER_DEBUG("RAM usage before buffer update=%ld", UProcessInfo::getMemoryUsage());
        at::Tensor labels = new_data->labels();
        at::Tensor classes_in_new_data = new_data->classes_in_dataset();

        std::vector<int64_t> classes_in_new_data_vec(classes_in_new_data.data<int64_t>(), classes_in_new_data.data<int64_t>() + classes_in_new_data.numel());
        ULOGGER_DEBUG("Number of new classes in new data=%d", (int)classes_in_new_data_vec.size());
        this->_seen_classes.insert(classes_in_new_data_vec.begin(), classes_in_new_data_vec.end());
        ULOGGER_DEBUG("Number of total classes in replay memory=%d", (int)this->_seen_classes.size());
        size_t group_length = this->_get_group_length();
        if(group_length < 40)
        {
            this->resize(this->_seen_classes.size() * 60);
        }
        group_length = this->_get_group_length();
        ULOGGER_DEBUG("Size of each %d bucket=%d", (int)this->_seen_classes.size(), (int)group_length);

        for (const auto &current_class : this->_seen_classes)
        {
            at::Tensor indices = at::nonzero(labels == at::tensor((int64_t)current_class));

            if (this->_buffer_groups.count(current_class)) // already seen
            {
                if (indices.numel() != 0) // indices in new_data
                {
                    indices = indices.squeeze(-1);
                    std::shared_ptr<LatentDataset> new_data_class;
                    new_data->subset(indices, new_data_class);
                    this->_buffer_groups[current_class]->update(new_data_class);
                }
                // for new_data and not new_data
                this->_buffer_groups[current_class]->resize(group_length);
                ULOGGER_DEBUG("Buffer for class %d updated. New buffer size=%d", (int)current_class, (int)this->_buffer_groups[current_class]->buffer()->size().value_or(0));
            }
            else
            {
                indices = indices.squeeze(-1);
                std::shared_ptr<LatentDataset> new_data_class;
                new_data->subset(indices, new_data_class);
                this->_buffer_groups[current_class] = std::make_shared<ReservoirSamplingBuffer>(group_length);
                this->_buffer_groups[current_class]->update(new_data_class);
                ULOGGER_DEBUG("Buffer for class %d created. New data size=%d", (int)current_class, (int)this->_buffer_groups[current_class]->buffer()->size().value_or(0));
            }
        }

        // for (size_t i = 0; i < classes_in_new_data.size(0); i++)
        // {
        //     at::Tensor indices = at::nonzero(labels == classes_in_new_data[i]).squeeze(-1);
        //     std::shared_ptr<LatentDataset> new_data_class;
        //     new_data->subset(indices, new_data_class);
        //     size_t current_class = static_cast<size_t>(classes_in_new_data[i].item().toLong());

        //     if (this->_buffer_groups.count(current_class))
        //     {
        //         this->_buffer_groups[current_class]->update(new_data_class);
        //         this->_buffer_groups[current_class]->resize(group_length);
        //         ULOGGER_DEBUG("Buffer for class %d updated. New buffer size=%d", (int)current_class, (int)this->_buffer_groups[current_class]->buffer()->size().value_or(0));
        //     }
        //     else
        //     {
        //         this->_buffer_groups[current_class] = std::make_shared<ReservoirSamplingBuffer>(group_length);
        //         this->_buffer_groups[current_class]->update(new_data_class);
        //         ULOGGER_DEBUG("Buffer for class %d created. New data size=%d", (int)current_class, (int)this->_buffer_groups[current_class]->buffer()->size().value_or(0));
        //     }
        // }
        this->_update_buffer();

        ULOGGER_DEBUG("RAM usage after buffer update=%ld", UProcessInfo::getMemoryUsage());
    }

    void ClassBalancedBuffer::resize(size_t new_size)
    {
        ULOGGER_DEBUG("Bucket resizing of dimension=%d", (int)new_size);
        this->_max_size = new_size;
        size_t group_length = this->_get_group_length();
        for (const auto &b : this->_buffer_groups)
        {
            b.second->resize(group_length);
        }
    }

    const std::unordered_map<size_t, std::shared_ptr<ReservoirSamplingBuffer>> &ClassBalancedBuffer::buffer_groups() const
    {
        return this->_buffer_groups;
    }

    void ClassBalancedBuffer::get_ids_in_memory(std::unordered_set<int> &ids_in_memory) const
    {
        for (const auto &b : this->_buffer_groups)
        {
            ids_in_memory.insert(b.second->buffer()->ids().begin(), b.second->buffer()->ids().end());
        }
    }

    // void ClassBalancedBuffer::updateLabels(std::unordered_map<int, std::pair<int, int>> &signatures_moved)
    // {
    //     for(const auto &id_region : signatures_moved)
    //     {
    //         this->_buffer->update_label(id_region.first, id_region.second.second);
    //         for(const auto &b : this->_buffer_groups)
    //         {
    //             if(id_region.second == b.first)
    //             {
    //                 this->_buffer_groups[b.first]
    //                 b.second->buffer()->update_label(id_region.first, id_region.second.second);
    //             }
    //         }
    //     }

    // }

    void ClassBalancedBuffer::_update_buffer()
    {
        this->_buffer = std::make_shared<LatentDataset>();
        for (const auto &b : this->_buffer_groups)
        {
            this->_buffer->concat(b.second->buffer(), this->_buffer);
        }
        ULOGGER_DEBUG("Buffer of total size=%d", (int)this->_buffer->size().value_or(0));
    }

    size_t ClassBalancedBuffer::_get_group_length()
    {
        return static_cast<size_t>(this->_max_size / this->_seen_classes.size());
    }
}