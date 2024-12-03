import torch
from torch import nn
from collections import OrderedDict

def _get_named_children_until(model, layer_name):
    layers = OrderedDict()
    for name, child in model.named_children():
        if name != layer_name:
            layers[name] = child
        else:
            break
    return layers

def _get_resnet(model):
    layers = _get_named_children_until(model, "fc")
    layers["flatten"] = nn.Flatten()
    return nn.Sequential(layers)

def _get_resnet_places(state_dict_file: str, 
                      name: str):
    model = torch.hub.load("pytorch/vision:v0.10.0", name, pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 365)
    model.load_state_dict(torch.load(state_dict_file))
    return _get_resnet(model), model.fc.in_features

class FeatureExtractor(nn.Module):
    def __init__(self, 
                 model_name: str = "resnet18",
                 state_dict_file: str = None,
                 trainable_from_layer: str = None,
                 device: str = "cuda",
                 **kwargs):
        super(FeatureExtractor, self).__init__()
        base_model, self.output_features = _get_resnet_places(state_dict_file, model_name)#, 512#_get_resnet(torch.hub.load("pytorch/vision:v0.10.0", model_name, pretrained=False))#
        
        self.freezed_part = OrderedDict()
        self.trainable_part = OrderedDict()

        if trainable_from_layer is not None:
            found = False
            
            #RESNET_18
            for name, child in base_model.named_children():
                if name == trainable_from_layer:
                    found = True
                if found:
                    self.trainable_part[name] = child
                else:
                    self.freezed_part[name] = child            
                    
            self.freezed_part = nn.Sequential(self.freezed_part).to(device, non_blocking=True) if len(self.freezed_part) > 0 else None      
        
        else:
            self.freezed_part = base_model    
            
        self.trainable_part = nn.Sequential(self.trainable_part).to(device, non_blocking=True) if len(self.trainable_part) > 0 else None

        assert self.trainable_part is not None or self.freezed_part is not None, "Feature extractor must have at least one model (freezed or trainable)"

        if self.freezed_part is not None:
            for param in self.freezed_part.parameters():
                param.requires_grad = False
            self.freezed_part.eval() 
    
    @torch.jit.export
    def extract_freezed_features(self, x):
        if self.freezed_part is not None:
            with torch.no_grad():
                x = self.freezed_part(x)
        return x
    
    @torch.jit.export
    def extract_trainable_features(self, x):
        if self.trainable_part is not None:
            x = self.trainable_part(x)
        return x
        
    def train(self, mode: bool = True):
        self.training = mode
        if self.trainable_part is not None:
            return self.trainable_part.train(mode)
        return self
        

    
