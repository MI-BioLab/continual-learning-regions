from avalanche.training.storage_policy import ReservoirSamplingBuffer, ClassBalancedBuffer


def get_storage_policy(value, **kwargs):
    match value:
        case 0:
            return ReservoirSamplingBuffer(kwargs["replay_memory_size"])
        case 1:
            return ClassBalancedBuffer(kwargs["replay_memory_size"])
        
    raise ValueError("get_storage_policy: invalid storage policy")


__all__ = ["get_storage_policy"]
