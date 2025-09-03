"""
Backbone networks for SCNN lane detection
Currently supports VGG16 backbone
"""
import torch
import torch.nn as nn
import torchvision.models as models

class VGG16Backbone(nn.Module):
    """
    VGG16 backbone for SCNN lane detection
    Modified to output feature maps at specific resolution
    """
    def __init__(self, pretrained=True):
        super(VGG16Backbone, self).__init__()
        
        # VGG16 feature layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            
            # Block 5 (modified for lane detection)
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Feature dimension reduction for SCNN
        self.feature_reduce = nn.Conv2d(512, 128, 1)
        
        if pretrained:
            self._load_pretrained_weights()
    
    def _load_pretrained_weights(self):
        """Load pretrained VGG16 weights"""
        pretrained_vgg = models.vgg16(pretrained=True)
        
        # Map pretrained weights to our architecture
        pretrained_dict = pretrained_vgg.features.state_dict()
        model_dict = self.features.state_dict()
        
        # Filter matching layers
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and model_dict[k].shape == v.shape}
        
        model_dict.update(pretrained_dict)
        self.features.load_state_dict(model_dict)
        print("Loaded pretrained VGG16 weights")
    
    def forward(self, x):
        features = self.features(x)
        features = self.feature_reduce(features)
        return features


class ResNet18Backbone(nn.Module):
    """
    ResNet18 backbone alternative for SCNN
    """
    def __init__(self, pretrained=True):
        super(ResNet18Backbone, self).__init__()
        
        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=pretrained)
        
        # Remove the last two layers (avgpool and fc)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Feature dimension reduction for SCNN
        self.feature_reduce = nn.Conv2d(512, 128, 1)
    
    def forward(self, x):
        features = self.features(x)
        features = self.feature_reduce(features)
        return features
