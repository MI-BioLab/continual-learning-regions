import torch

def get_weighting_function(value, **kwargs):
    match value:
        case 0:
            return _compute_simple_weights
        case 1:
            return lambda x : _compute_effective_number_weights(x, beta=kwargs["beta"])
    raise ValueError("get_weighting_function: invalid weighting function")

def _compute_simple_weights(samples_per_class: torch.Tensor):
    return 1.0 / samples_per_class


def _compute_effective_number_weights(samples_per_class: torch.Tensor, beta=0.999):
    """Utility function to compute the weights as explained in https://arxiv.org/abs/1901.05555.

    The effective number Eₙ = (1-βⁿ) where n is the number of samples for each class.
    The weights w = (1-β) / Eₙ are then normalized and multiplied for the total number of classes.

    Args:
        labels (torch.Tensor): the labels of the dataset.
        beta (float, optional): a weighting parameter. Defaults to 0.999.

    Returns:
        torch.Tensor: a tensor containing the weights for each class.
    """
    num_classes = torch.count_nonzero(samples_per_class)
    effective_num = 1.0 - torch.pow(beta, samples_per_class)
    effective_num[samples_per_class == 0] = 1e10
    weights = (1.0 - beta) / effective_num
    weights = weights / torch.sum(weights) * num_classes
    return weights

__all__ = ["get_weighting_function"]