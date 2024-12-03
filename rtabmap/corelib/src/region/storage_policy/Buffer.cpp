#include "rtabmap/core/region/storage_policy/Buffer.h"

#include "rtabmap/utilite/ULogger.h"

namespace rtabmap
{
    Buffer::Buffer(size_t max_size) : _max_size(max_size)
    {
        this->_buffer = std::make_shared<LatentDataset>();
        ULOGGER_DEBUG("New buffer creation of size %d", (int)_max_size);
    }
    
}