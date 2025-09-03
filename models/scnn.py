"""
Complete SCNN model for lane detection
"""
import torch
import torch.nn as nn
from .backbone import VGG16Backbone, ResNet18Backbone
from .spatial_conv import SpatialConv

class SCNNLaneDetection(nn.Module):
    """
    Complete SCNN model for lane detection
    """
    def __init__(self, num_classes=5, backbone='vgg16'):
        super(SCNNLaneDetection, self).__init__()
        
        self.num_classes = num_classes
        
        # Backbone network
        if backbone == 'vgg16':
            self.backbone = VGG16Backbone(pretrained=True)
        elif backbone == 'resnet18':
            self.backbone = ResNet18Backbone(pretrained=True)
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented")
        
        # Spatial convolution module
        self.spatial_conv = SpatialConv(num_channels=128)
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)
        )
        
        # Existence prediction branch
        self.existence_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes - 1)  # Background class excluded
        )
    
    def forward(self, x):
        # Extract backbone features
        features = self.backbone(x)
        
        # Apply spatial convolution
        spatial_features = self.spatial_conv(features)
        
        # Lane segmentation prediction
        seg_output = self.classifier(spatial_features)
        
        # Upsample to original input size
        seg_output = nn.functional.interpolate(
            seg_output, 
            size=(x.shape[2], x.shape[3]), 
            mode='bilinear', 
            align_corners=True
        )
        
        # Lane existence prediction
        exist_output = self.existence_classifier(spatial_features)
        
        return {
            'segmentation': seg_output,
            'existence': exist_output,
            'features': spatial_features
        }
