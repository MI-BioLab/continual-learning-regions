from typing import Sequence, Callable

import torch
from torch.utils.data import DataLoader
from sklearn import metrics
import numpy as np
from tqdm import tqdm
import sys

sys.path.insert(1, "src/")
from setting_strategies import BaseStrategy
def loop(
    strategy: BaseStrategy,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    accuracy_fn: Callable,
    num_classes: int,
    optimizer = None,
    top_k: Sequence[int] = [1, 3, 5],
    train: bool = True,
    **kwargs
):        
    strategy.model.train(train)

    total_loss = 0.0
    total_predictions = []
    total_targets = []

    if len(top_k) > 0:
        top_k_accuracy = torch.zeros(len(top_k)).to(strategy.model.device, non_blocking=True)

    for batch_id, batch in enumerate(tqdm(dataloader)):  # (ids, im_0, im_1, ..., im_N, label)       
        
        with torch.set_grad_enabled(train):
            input = strategy.before_feature_extraction(batch).to(strategy.model.device, non_blocking=True)
            features = strategy.extract_freezed_features(input) 
            features = strategy.after_freezed_features_extraction(features)
            features = strategy.extract_trainable_features(features)
            features = strategy.after_trainable_features_extraction(features)
            predictions = strategy.classify(features)
            predictions = strategy.after_classification(predictions)
            targets = batch[-1].to(strategy.model.device, non_blocking=True)
           
            one_hot_targets = torch.nn.functional.one_hot(targets, num_classes).float()
            
            total_predictions.append(predictions.detach().cpu().numpy())
            total_targets.append(targets.detach().cpu().numpy())
            loss = loss_fn(predictions, one_hot_targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            

        with torch.no_grad():
            for i in range(len(top_k)):
                top_k_accuracy[i] += accuracy_fn(
                    predictions, targets, num_classes=num_classes, top_k=top_k[i]
                )

    with torch.no_grad():
        total_targets = np.concatenate(total_targets)
        total_predictions = np.argmax(np.concatenate(total_predictions), axis=-1)
        confusion_matrix = metrics.confusion_matrix(
            total_targets,
            total_predictions
        )
    return total_loss / (batch_id + 1), top_k_accuracy / (batch_id + 1), confusion_matrix