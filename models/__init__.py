"""
Models package initialization
"""
from .scnn import SCNNLaneDetection
from .backbone import VGG16Backbone
from .spatial_conv import SpatialConv

__all__ = ['SCNNLaneDetection', 'VGG16Backbone', 'SpatialConv']
