from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.CrossEntropyLoss):
    def __init__(
        self,
        gamma=2,
        weight: Optional[torch.Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super(FocalLoss, self).__init__(weight, size_average, ignore_index,
                                        reduce, reduction, label_smoothing)
        self.gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        cross_entropy = F.cross_entropy(input,
                                        target,
                                        reduction="none",
                                        weight=self.weight)
        pt = torch.exp(-cross_entropy)
        focal_loss = (1 - pt)**self.gamma * cross_entropy
        return (
            torch.mean(focal_loss) if self.reduction == "mean" else
            torch.sum(focal_loss) if self.reduction == "sum" else focal_loss)


__all__ = ["FocalLoss"]
