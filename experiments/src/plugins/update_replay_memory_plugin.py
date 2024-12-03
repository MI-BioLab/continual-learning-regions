from typing import  TYPE_CHECKING

import copy
import torch
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from torch.utils.data.dataset import TensorDataset
from avalanche.benchmarks.utils import make_classification_dataset, AvalancheDataset
from avalanche.training.storage_policy import ReservoirSamplingBuffer, ClassBalancedBuffer

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate
    
from .replay_plugin import ReplayPlugin 
from memory_profiler import profile
import globals

class UpdateReplayMemoryPlugin(SupervisedPlugin):
    """
    Plugin to weigh the loss function depending on the classes distribution in the dataset.
    """
    def __init__(
        self,
        storage_policy_type: int,
        remove_unused=False,
        update=False        
    ):
        super().__init__()
        self.storage_policy_type = storage_policy_type 
        self.remove_unused = remove_unused
        self.update = update
     
    @profile(stream=globals.memory_profiler_file)  
    def _remove_buffer_unused_examples(self,
                                       buffer: AvalancheDataset):
        ids = []
        features = []
        targets = []
        task_labels = []
        if len(buffer) > 0:
            for id, feature, target, task_label in buffer:
                ids.append(id)
                features.append(feature)
                targets.append(target)
                task_labels.append(task_label)
                
            ids = torch.tensor(ids)
            targets = torch.tensor(targets)
            task_labels = torch.tensor(task_labels)
            features = torch.stack(features)
            
            dataset = TensorDataset(ids, features, targets)
            return make_classification_dataset(dataset, task_labels=task_labels, targets=targets)
            
            
        
        
    @profile(stream=globals.memory_profiler_file)
    def _remove_reservoir_unused_examples(self,
                                          storage_policy: ReservoirSamplingBuffer):
        storage_policy.buffer = self._remove_buffer_unused_examples(storage_policy.buffer)
        
    @profile(stream=globals.memory_profiler_file)
    def _remove_class_balanced_unused_examples(self, 
                                               storage_policy: ClassBalancedBuffer):
        for buffer_group in storage_policy.buffer_groups.values():
            buffer_group.buffer = self._remove_buffer_unused_examples(buffer_group.buffer)   
            
        
    @profile(stream=globals.memory_profiler_file)
    def _update_reservoir_sampling_examples(self,
                                            storage_policy: ReservoirSamplingBuffer,
                                            nodes_moved_in_this_experience
                                            ):
        changed_idx = []  
        for i in range(len(storage_policy.buffer)):
            id, _, _, _ = storage_policy.buffer[i] #ids, freezed_features, targets, task_labels
            if id.item() in nodes_moved_in_this_experience:
                changed_idx.append(i)
                storage_policy.buffer[i][2] = nodes_moved_in_this_experience[id.item()][1] #change targets
        
        new_weights = torch.rand(len(changed_idx))
        storage_policy.buffer[changed_idx] = new_weights
        if globals.debug:
            with open(globals.debug_file, "a") as file:
                file.write(f"Updating reservoir sampling buffer\n")
                file.write(f"New buffer weights={new_weights.tolist()}\n")
    
    @profile(stream=globals.memory_profiler_file)
    def _update_class_balanced_examples(self,
                                        storage_policy: ClassBalancedBuffer,
                                        nodes_moved_in_this_experience):
        if globals.debug:
            with open(globals.debug_file, "a") as file:
                file.write(f"Updating class balanced buffer\n")
        changed_buffers = dict()
        for id, changes in nodes_moved_in_this_experience.items():
            if changes[0] not in changed_buffers:
                changed_buffers[changes[0]] = [[], [], []] #ids, freezed_features, targets
            if changes[1] not in changed_buffers:
                changed_buffers[changes[1]] = [[], [], []] #ids, freezed_features, targets
            
        for buffer_group_id in changed_buffers:
            if buffer_group_id in storage_policy.buffer_groups:
                idx_to_keep = []
                for i in range(len(storage_policy.buffer_groups[buffer_group_id].buffer)):
                    id, freezed_feature, _, _ = storage_policy.buffer_groups[buffer_group_id].buffer[i]
                    if id.item() not in nodes_moved_in_this_experience:
                        idx_to_keep.append(i)
                    elif nodes_moved_in_this_experience[id.item()][0] == buffer_group_id: #change group
                        new_group_id = nodes_moved_in_this_experience[id.item()][1]                        
                        changed_buffers[new_group_id][0].append(id)
                        changed_buffers[new_group_id][1].append(freezed_feature)
                        changed_buffers[new_group_id][2].append(torch.tensor(new_group_id)) 
                
                storage_policy.buffer_groups[buffer_group_id].buffer = storage_policy.buffer_groups[buffer_group_id].buffer.subset(idx_to_keep)
                storage_policy.buffer_groups[buffer_group_id]._buffer_weights = storage_policy.buffer_groups[buffer_group_id]._buffer_weights[idx_to_keep]
                
        for buffer_group_id, buffer in changed_buffers.items():
            if len(buffer[0]) > 0: #some nodes moved in this experience may not be present in any buffer because of the max_size of the buffer
                
                ids = copy.deepcopy(buffer[0])
                freezed_features = copy.deepcopy(buffer[1])
                targets = copy.deepcopy(buffer[2])
            
                if buffer_group_id in storage_policy.buffer_groups:
                    for id, freezed_feature, target, _  in storage_policy.buffer_groups[buffer_group_id].buffer:
                        ids.append(id)
                        freezed_features.append(freezed_feature)
                        targets.append(target)
                else:
                    storage_policy.buffer_groups[buffer_group_id] = ReservoirSamplingBuffer(len(ids)) #max_size?                
                
                ids = torch.stack(ids)
                freezed_features = torch.stack(freezed_features)
                targets = torch.stack(targets)
                    
                experience_dataset = TensorDataset(ids, freezed_features, targets)
   
               
                storage_policy.buffer_groups[buffer_group_id].buffer = make_classification_dataset(experience_dataset, task_labels=[0] * len(experience_dataset), targets=targets)
                storage_policy.buffer_groups[buffer_group_id]._buffer_weights = \
                    torch.cat([torch.rand(len(buffer[0])), storage_policy.buffer_groups[buffer_group_id]._buffer_weights])
                
                if globals.debug:
                    with open(globals.debug_file, "a") as file:
                        file.write(f"New buffer len for class {buffer_group_id} = {len(storage_policy.buffer_groups[buffer_group_id].buffer)}\n")
                        file.write(f"New buffer weights for class {buffer_group_id} = {storage_policy.buffer_groups[buffer_group_id]._buffer_weights.tolist()}\n")
      
    @profile(stream=globals.memory_profiler_file)     
    def before_training_exp(self, 
                            strategy: "SupervisedTemplate", 
                            **kwargs):
        assert isinstance(strategy.plugins[0], ReplayPlugin), "Cannot call update_replay_memory_plugin if replay memory is not used!"
        
        if self.update:
            nodes_moved_in_this_experience = strategy.nodes_moved_in_each_experience[strategy.experience.current_experience]

            match self.storage_policy_type:
                case 0:
                    self._update_reservoir_sampling_examples(strategy.plugins[0].storage_policy, nodes_moved_in_this_experience)
                case 1:
                    self._update_class_balanced_examples(strategy.plugins[0].storage_policy, nodes_moved_in_this_experience)


    @profile(stream=globals.memory_profiler_file)
    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        assert isinstance(strategy.plugins[0], ReplayPlugin), "Cannot call update_replay_memory_plugin if replay memory is not used!"
        
        if self.remove_unused:
            match self.storage_policy_type:
                case 0:
                    
                    self._remove_reservoir_unused_examples(strategy.plugins[0].storage_policy)
                case 1:
                    self._remove_class_balanced_unused_examples(strategy.plugins[0].storage_policy)
        
    