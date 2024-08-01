from torch import nn

"""
The entire model: CNN + MLP.
"""
class Model(nn.Module):
    def __init__(self, 
                 feature_extractor: nn.Module, 
                 classifier: nn.Module, 
                 device: str = "cuda"):
        super(Model, self).__init__()
        self.device = device
        self.feature_extractor = feature_extractor
        self.classifier = classifier
    
    #never call forward directly here
    def forward(self, x):
        pass
    
    def train(self, mode: bool = True):
        self.feature_extractor.train(mode)
        self.classifier.train(mode)
        
